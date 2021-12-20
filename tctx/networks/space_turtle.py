"""
turtle model with spatial component
"""

import pandas as pd
import numpy as np
import scipy.stats as st
import copy

from tctx.util.profiling import log_time
from tctx.util.networks import MutableNetwork, merge_dicts
import logging

from tctx.networks import space, turtle_data
import tctx.networks.base
import scipy.stats
import scipy.interpolate
from collections import namedtuple

from tctx.util import track

ConnProfile = namedtuple('ConnProfile', 'profile max_distance_um')

_DEFAULT_SPACE_BIN_SIZE_UM = 200


def get_network(extra_params=None):
    """
    :param extra_params: optional dict with a set of parameters
    :return: a MutableNetwork object
    """
    net = MutableNetwork(
        instantiate=instantiate,
        base_params=merge_dicts(
            _get_space_base_params_neuron(),
            turtle_data.get_base_params_network(),
            {'model': 'aeif_cond_exp'},
        ))

    if extra_params is not None:
        net = net.stack(extra_params)

    return net


def _get_space_base_params_neuron():
    params = turtle_data.get_base_params_neuron()
    params['std_scaling'] = 1.

    return params


def instantiate(params):
    """create the cells and connection dataframes"""

    original_params = copy.deepcopy(params)

    cell_count = params['N']

    with log_time(f'building', pre=True):

        layers_desc = turtle_data.get_layers_desc()
        space_def = space.define_space(layers_desc, cell_count)
        logging.debug('space def: %s', space_def)

        cells, connections = _build_network(params, layers_desc, space_def)

        return tctx.networks.base.NetworkInstance(
            merge_dicts(original_params, space_def),
            cells,
            connections
        )


def _assign_weights_and_delays(params: pd.Series, connections: pd.DataFrame):
    weight_dist = _create_weight_dist(params)
    delay_dist = st.uniform(params['delays_low'], params['delays_high']).rvs

    with log_time('assign weights and delays', pre=True):
        logging.debug('assigning weight_dist')
        connections['weight'] = weight_dist.sample(connections.syn_type)

        logging.debug('assigning delay_dist')
        connections['delay'] = tctx.networks.base.sample_delays(connections, delay_dist)


def _load_connection_profiles(params):
    """return distance-dependent profile of connection probability"""
    filename = track.ExperimentTracker().get_processed_data_path('connectivity_profiles.h5')

    connectivity_profile_name = params.get('connectivity_profile', 'gaussian_decay')
    logging.debug('Using %s connectivity profile', connectivity_profile_name)

    # noinspection PyTypeChecker
    profiles: pd.DataFrame = pd.read_hdf(filename, connectivity_profile_name + '_profiles')

    profiles: dict = {
        c_type:  ConnProfile(
            profile=scipy.interpolate.interp1d(
                profiles.index.values,
                profiles[c_type].values,
                fill_value='extrapolate',
            ),
            max_distance_um=profiles.index[-1] + (profiles.index[-1] - profiles.index[-2]) * .5
        )

        for c_type in profiles.columns
    }

    # noinspection PyTypeChecker
    fixed_in: pd.DataFrame = pd.read_hdf(filename, connectivity_profile_name + '_degrees')

    return profiles, fixed_in['in_degree']


def _build_network(params, layers_desc, space_def) -> (pd.DataFrame, pd.DataFrame):
    """create cells, connections """
    cells = _create_cells(layers_desc, space_def)

    with log_time('create connections', pre=True):
        conn_profiles, fixed_in = _load_connection_profiles(params)

        free_degree = params.get('free_degree', True)
        logging.info('using %s degrees', 'free' if free_degree else 'fixed')

        proj_offsets = _extract_proj_offsets(params)
        logging.info('using proj offsets: %s', proj_offsets)

        connections = space.create_connections(cells, fixed_in, space_def, conn_profiles,
                                               bin_size_um=_DEFAULT_SPACE_BIN_SIZE_UM,
                                               free_degree=free_degree,
                                               proj_offsets=proj_offsets)

    _assign_weights_and_delays(params, connections)

    return cells, connections


def _extract_proj_offsets(params):
    """
    We allow offsets in the X direction that are specific to the connection- or synapse-type.
    By default, no offset.
    """

    proj_offsets = {
        'e2e': np.array([0., 0.]),
        'e2i': np.array([0., 0.]),
        'i2e': np.array([0., 0.]),
        'i2i': np.array([0., 0.]),
    }

    # collect connection-type-specific x projection offset from params
    for c_type_k in proj_offsets.keys():
        param_key = f'{c_type_k}_proj_offset'
        if param_key in params:
            proj_offsets[c_type_k] = np.array([params[param_key], 0.])

    # collect synapse-type-specific x projection offset from params
    for ei_type in ['e', 'i']:
        param_key = f'{ei_type}_proj_offset'

        if param_key in params:
            for c_type_k in [f'{ei_type}2e', f'{ei_type}2i']:
                assert c_type_k not in params
                proj_offsets[c_type_k] = np.array([params[param_key], 0.])

    return proj_offsets


def _create_weight_dist(params):
    """create weights from a lognormal for excitatory connections and
    from a scaled version for inhibitory ones"""

    e_weights_range_low = params.get('e_weights_range_low', 0.)
    e_weights_range_high = params.get('e_weights_range_high', 1.)

    vmin, vmax = params.get('e_weights_vmin', 0), params['e_weights_vmax']

    vrange = vmax - vmin
    vmin = vmin + vrange * e_weights_range_low
    vmax = vmin + vrange * e_weights_range_high

    logging.info(f'weight distribution in [%f, %f]. That is [%f%% - %f%%]',
                 vmin, vmax, e_weights_range_low * 100., e_weights_range_high * 100.)

    e2x = tctx.networks.base.ClippedDist(
        dist=st.lognorm(
            loc=params['e_weights_loc'],
            s=params['e_weights_shape'],
            scale=params['e_weights_scale'],
        ).rvs,
        vmin=vmin,
        vmax=vmax,
    )

    if 'i_weights_loc' in params:
        logging.error('Attempting to use explicit inhibitory distribution')

    i2x = e2x.scale(params['g'])

    return tctx.networks.base.ByDist(e2x=e2x, i2x=i2x)


def _create_cells(layers_desc, space_def) -> pd.DataFrame:
    """randomly place neurons in the given space"""
    logging.debug('assign cell counts')
    layers_cell_counts = space.assign_cell_counts(layers_desc.density_per_mm2, space_def['size_mm2'])

    logging.debug('placing %d cells: %s', layers_cell_counts.sum(), layers_cell_counts.to_dict())
    cells = space.assign_cell_positions_cube(layers_desc.height_um, layers_cell_counts, space_def['side_um'])

    cells['ei_type'] = cells.layer.map({'L1': 'i', 'L2': 'e', 'L3': 'i'}).astype('category')
    cells = cells.sort_values('ei_type').reset_index(drop=True)

    return cells
