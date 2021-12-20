"""
    Code to build spatially-dependent network.

    There's two sides to it:
    - Use correct density of different cell types and place them in space
    - Connect them according to spatially-dependent radial connectivity profiles

    For the connectivity part we need to evaluate all-to-all-distances which is O(n^2).
    Instead, take advantage of the profiles decaying with distance and use a space partitioning structure
    so we only compute distances between reachable cell pairs.
"""
import logging
import numpy as np
import pandas as pd
from collections import namedtuple
import numba
from tctx.util.profiling import log_time
from tctx.util import parallel


##########################################################################################
# Density


def assign_area(layers_desc, target_cell_count):
    return target_cell_count / layers_desc.density_per_mm2.sum()


def assign_cell_counts(density_per_mm2, size_mm2):
    layers_cell_counts = (size_mm2 * density_per_mm2).round().astype(np.int)
    return layers_cell_counts


def assign_cell_positions_cube(height_um, layers_cell_counts, side_um):
    cells = pd.DataFrame({
        'layer': np.repeat(height_um.index, layers_cell_counts)
    }, dtype='category')

    total_cell_count = layers_cell_counts.sum()

    cells['x'] = np.random.rand(total_cell_count) * side_um
    cells['y'] = np.random.rand(total_cell_count) * side_um

    layer_z_offsets = np.concatenate([[0], height_um.cumsum().values[:-1]])

    cells['z'] = np.concatenate([
        np.random.rand(count) * height + offset
        for count, height, offset in zip(layers_cell_counts, height_um, layer_z_offsets)
    ])

    return cells


def define_space(layers_desc, target_cell_count) -> pd.Series:
    """
    given statistics of the cortical layers and the total cell count,
    generate the total volume of the circuit.
    """

    size_mm2 = assign_area(layers_desc, target_cell_count)
    mm_to_mu = 1000.
    side_um = np.sqrt(size_mm2) * mm_to_mu

    return pd.Series({'size_mm2': size_mm2, 'side_um': side_um, 'height_um': layers_desc.height_um.sum()})


##########################################################################################
# Spacially-dependent connectivity


def get_cell_count_in_area(densities, radius_um):
    """in mm2"""
    assert isinstance(densities, pd.Series)

    um_to_mm = 0.001
    sampling_radius_mm = radius_um * um_to_mm
    sampling_area_mm2 = np.pi * sampling_radius_mm ** 2

    return densities * sampling_area_mm2


@numba.njit
def wrapped_a2a_distance(p, q, m):
    diff = np.expand_dims(p, axis=1) - np.expand_dims(q, axis=0)
    diff = np.abs(diff)
    return np.sqrt(np.sum(np.square(np.minimum(diff, m - diff)), axis=-1))


BinPartitioning = namedtuple('BinPartitioning', 'bin_edges, bin_distances, total_bin_count, bin_dims')
CellPartitioning = namedtuple('CellPartitioning', 'cell_to_bin_idx, bin_cells_mask')


# @numba.jit
def build_partitioning_bins(side_um, bin_size_um):
    bin_edges = np.arange(0, side_um + bin_size_um, bin_size_um)
    bin_dims = (len(bin_edges) - 1, len(bin_edges) - 1)
    total_bin_count = (len(bin_edges) - 1) ** 2

    bin_coords = np.array(np.unravel_index(np.arange(total_bin_count), bin_dims)).T
    bin_distances = wrapped_a2a_distance(bin_coords, bin_coords, len(bin_edges) - 1) * bin_size_um

    return BinPartitioning(bin_edges, bin_distances, total_bin_count, bin_dims)


def partition_cells(sorted_cells, partitioning):
    gids = sorted_cells.index.values

    cell_to_bin_idx, bin_cells_mask = _partition_cells(
        gids,
        sorted_cells['x'].values,
        sorted_cells['y'].values,
        partitioning.bin_edges, partitioning.bin_dims, partitioning.total_bin_count
    )

    return CellPartitioning(
        pd.Series(cell_to_bin_idx, index=gids),
        pd.DataFrame(bin_cells_mask, columns=gids)
    )


# @numba.jit
def _partition_cells(gids, xs, ys, bin_edges, bin_dims, total_bin_count):
    x_bin_idx = np.digitize(xs, bin_edges) - 1
    y_bin_idx = np.digitize(ys, bin_edges) - 1

    cell_to_bin_idx = np.ravel_multi_index((x_bin_idx, y_bin_idx), bin_dims)

    bin_cells_mask = np.zeros((total_bin_count, len(gids)), dtype=np.bool_)
    bin_cells_mask[cell_to_bin_idx, np.arange(len(cell_to_bin_idx))] = True

    return cell_to_bin_idx, bin_cells_mask


def wrap_around_dimension(values, full_side):
    """make sure values are wrapped around in a torus"""
    offset = values.copy()

    mask = offset < 0
    offset[mask] = offset[mask] + full_side

    mask = offset > full_side
    offset[mask] = offset[mask] - full_side

    return offset


def create_connections(cells, conn_fixed_counts, space_def, conn_profiles, bin_size_um, free_degree, proj_offsets):
    """
    :param cells: DataFrame with columns: x, y, z, ei_type
    :param conn_fixed_counts: a dict from string (e2e, e2i, i2e, i2i) to an integer
        representing the number of incoming connections
    :param space_def:
    :param conn_profiles:
    :param bin_size_um:
    :param free_degree: flag on whether to use free degrees or fixed in degrees
    :return:
    """
    # TODO merge conn_fixed_counts and free_degree params
    # we don't want to use distance in Z to determine connectivity
    logging.debug('Using %s degrees', 'free' if free_degree else 'fixed')
    logging.debug('proj offset\n%s', proj_offsets)

    # Note: because we have a potential bias, the virtual position of the cell is different
    # depending on whether it is a source/target and what kind of connection we are looking at.

    c_types = ['e2e', 'e2i', 'i2e', 'i2i']

    if proj_offsets is None:
        proj_offsets = {c_type: np.array([0., 0.]) for c_type in c_types}

    with log_time('partitioning space'):

        partition = build_partitioning_bins(space_def['side_um'], bin_size_um)
        source_position_by_c_type = {}
        source_partition_by_c_type = {}
        target_position_by_c_type = {}
        target_partition_by_c_type = {}

        for c_type in c_types:
            sources = cells[cells.ei_type == c_type[0]].copy()

            sources['x'] = wrap_around_dimension(sources['x'].values + proj_offsets[c_type][0], space_def['side_um'])
            sources['y'] = wrap_around_dimension(sources['y'].values + proj_offsets[c_type][1], space_def['side_um'])

            sorted_source_positions = sources.sort_values(['ei_type', 'x', 'y', 'z'])[['x', 'y']]
            source_position_by_c_type[c_type] = sorted_source_positions
            source_partition_by_c_type[c_type] = partition_cells(sorted_source_positions, partition)

            targets = cells[cells.ei_type == c_type[-1]]
            sorted_target_positions = targets.sort_values(['ei_type', 'x', 'y', 'z'])[['x', 'y']]
            target_position_by_c_type[c_type] = sorted_target_positions
            target_partition_by_c_type[c_type] = partition_cells(sorted_target_positions, partition)

        c_type_per_task, s_type_per_task = [], []

        all_params = []

        c_type_codes = pd.Series(np.arange(4), index=['e2e', 'e2i', 'i2e', 'i2i'], dtype=np.uint)
        s_type_codes = pd.Series(np.arange(2), index=['e2x', 'i2x'], dtype=np.uint)

    with log_time('prepare tasks'):

        for c_type in c_types:

            fixed_in_count = conn_fixed_counts[c_type]

            source_ei_type = c_type[0]
            s_type = f'{source_ei_type}2x'

            max_distance_um = conn_profiles[c_type].max_distance_um

            # noinspection PyTypeChecker
            reachable_a2a_bins: pd.Series = partition.bin_distances <= max_distance_um

            logging.debug('detecting %s connections in %d bins (max distance %.1f)',
                          c_type, np.product(partition.bin_dims), max_distance_um)

            source_all_positions = source_position_by_c_type[c_type]
            target_all_positions = target_position_by_c_type[c_type]

            num_conns = len(target_all_positions) * fixed_in_count
            logging.debug('creating %d*%d=%d %s connections', len(target_all_positions), fixed_in_count, num_conns, c_type)

            bin_per_target_cell = target_partition_by_c_type[c_type].cell_to_bin_idx
            source_bin_mask = source_partition_by_c_type[c_type].bin_cells_mask

            for bin_idx, target_positions_in_single_bin in target_all_positions.groupby(bin_per_target_cell):

                reachable_from_given_bin = reachable_a2a_bins[bin_idx]

                reachable_cells_mask = np.any(source_bin_mask[reachable_from_given_bin], axis=0)

                source_candidate_positions_over_multiple_bins = source_all_positions[reachable_cells_mask]

                c_type_per_task.append(c_type_codes[c_type])
                s_type_per_task.append(s_type_codes[s_type])

                params = (
                    c_type,
                    source_candidate_positions_over_multiple_bins,
                    target_positions_in_single_bin,
                    fixed_in_count if not free_degree else None,
                    max_distance_um,
                    conn_profiles[c_type].profile,
                    space_def,
                )

                all_params.append(params)

    with log_time('parallel tasks'):
        results = parallel.parallel_tasks(connect_cells, all_params, parallel_pool='process')

    with log_time('process task results'):
        c_types, s_types, sources, targets = [], [], [], []
        for i in sorted(results.keys()):
            bin_sources, bin_targets = results[i]
            sources.append(bin_sources)
            targets.append(bin_targets)

            c_types.append(np.repeat(c_type_per_task[i], len(bin_sources)))
            s_types.append(np.repeat(s_type_per_task[i], len(bin_sources)))

        sources = np.concatenate(sources)
        targets = np.concatenate(targets)

        c_types = np.concatenate(c_types)
        c_types = pd.Categorical.from_codes(c_types, categories=c_type_codes.index.values)

        s_types = np.concatenate(s_types)
        s_types = pd.Categorical.from_codes(s_types, categories=s_type_codes.index.values)

    connections = pd.DataFrame({
        'source': sources,
        'target': targets,
        'con_type': c_types,
        'syn_type': s_types,
    })

    connections.index.name = 'conn_idx'

    return connections


def connect_cells(
        c_type,
        source_positions,
        target_positions,
        fixed_in_count,
        max_distance_um, prob_pdf, space_def,
):

    with parallel.limit_mkl_threads(1):
        if fixed_in_count is not None:
            return _connect_cells_jit_fixed_degree(
                c_type,
                source_positions.index.values,
                source_positions.values,
                target_positions.index.values,
                target_positions.values,
                fixed_in_count,
                max_distance_um, prob_pdf, space_def['side_um']
            )
        else:
            return _connect_cells_jit_free_degree(
                c_type,
                source_positions.index.values,
                source_positions.values,
                target_positions.index.values,
                target_positions.values,
                max_distance_um, prob_pdf, space_def['side_um']
            )


@numba.jit
def _connect_cells_jit_fixed_degree(
        c_type,
        source_gids,
        source_positions,
        target_gids,
        target_positions,
        fixed_in_count,
        max_distance_um, prob_pdf, side_um: float):
    """this should be dataframe-free so we can jit it"""

    t2s_dists = wrapped_a2a_distance(target_positions, source_positions, side_um)
    reachable_sources = t2s_dists <= max_distance_um

    # probs: (TARGETS, SOURCES)
    probs = np.zeros_like(t2s_dists)
    probs[reachable_sources] = prob_pdf(t2s_dists[reachable_sources])

    if c_type[0] == c_type[-1]:
        autapses = target_gids[:, np.newaxis] == source_gids[np.newaxis, :]
        probs[autapses] = 0

    probs = probs / np.sum(probs, axis=1)[:, np.newaxis]

    sources = np.empty(len(target_gids) * fixed_in_count, dtype=np.int)
    sources_idcs = np.arange(len(source_gids))

    for target_idx, sources_probs in enumerate(probs):
        chosen_sources_idcs = np.random.choice(
            sources_idcs,
            p=sources_probs,
            size=fixed_in_count,
            replace=False
        )

        chosen_sources_gids = source_gids[chosen_sources_idcs]

        slice_in_summary_array = slice(
            target_idx * fixed_in_count,
            (target_idx + 1) * fixed_in_count)

        sources[slice_in_summary_array] = chosen_sources_gids

    targets = np.repeat(target_gids, fixed_in_count)

    return sources, targets


# @numba.jit
def _connect_cells_jit_free_degree(
        c_type,
        source_gids,
        source_positions,
        target_gids,
        target_positions,
        max_distance_um, prob_pdf, side_um: float):
    """this should be dataframe-free so we can jit it"""

    t2s_dists = wrapped_a2a_distance(target_positions, source_positions, side_um)
    reachable_sources = t2s_dists <= max_distance_um

    # probs: (TARGETS, SOURCES)
    probs = np.zeros_like(t2s_dists)
    probs[reachable_sources] = prob_pdf(t2s_dists[reachable_sources])

    if c_type[0] == c_type[-1]:
        autapses = target_gids[:, np.newaxis] == source_gids[np.newaxis, :]
        probs[autapses] = 0

    conn = np.random.random(probs.shape) < probs

    t_idcs, s_idcs = np.where(conn)

    sources = source_gids[s_idcs]
    targets = target_gids[t_idcs]

    return sources, targets
