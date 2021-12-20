"""
The graph of traversed connections
"""
import numpy as np
import pandas as pd
import numba

from tctx.util import conn_check
from tctx.util.profiling import log_time

import collections


Binning = collections.namedtuple(
    'Binning',  # numba-friendly version of time-binning spikes
    [
        'tbins',  # np.ndarray, int, index of time bin that a spike belongs (sorted)
        'idcs',  # np.ndarray, int, index of spikes into a metadata table
        'times',  # np.ndarray, float, time of the spike
    ])


@numba.njit
def binning_make(idcs, times, bins):
    # digitize calls the bins by their RIGHT edge
    tbins = np.digitize(times, bins=bins) - 1

    sorting = np.argsort(tbins)
    return Binning(
        tbins[sorting],
        idcs[sorting],
        times[sorting],
    )


@numba.njit
def binning_find(binning, tbin):
    return slice(
        np.searchsorted(binning.tbins, tbin, side='left'),
        np.searchsorted(binning.tbins, tbin, side='right')
    )


@numba.njit
def binning_find_times(binning, tbin) -> np.ndarray:
    return binning.times[binning_find(binning, tbin)]


@numba.njit
def binning_find_idcs(binning, tbin) -> np.ndarray:
    return binning.idcs[binning_find(binning, tbin)]


@numba.njit
def extract_spike_to_spike_jumps_jit(
        source_idcs, source_times,
        target_idcs, target_times,
        thresh_ms, tbin_ms):
    """
        extract "effective" connections by detecting spikes that happen at least once
        sequentially within a small time window

        We use partitioning strategy. Classify spikes in time bins big enough that two spikes can only
        be below threshold if they are in the same or consecutive bins.

        The parameters source_spikes and target_spikes indicate candidates to be in jumps.
        For example: you may want to pass inhibitory spikes only as "target" candidates
        (because inhibitory ones don't propagate activity so can't be "source") and externally induced
        spikes only as "source".
    """
    assert thresh_ms[0] <= thresh_ms[1]

    tstart = min(np.min(source_times), np.min(target_times))
    tstop = max(np.max(source_times), np.max(target_times))
    tbins = np.arange(tstart, tstop + tbin_ms * .5, tbin_ms)

    source_binning = binning_make(source_idcs, source_times, tbins)
    target_binning = binning_make(target_idcs, target_times, tbins)

    effective_sources = []
    effective_targets = []
    all_bin_ids = np.arange(len(tbins) - 1)

    def pairwise_connections(i, j):
        """
            detect potential connections between all of the spikes in two bins

        :param i: source bin index
        :param j: target bin index
        """
        source_times = binning_find_times(source_binning, i)
        target_times = binning_find_times(target_binning, j)

        tdiffs = np.expand_dims(target_times, axis=1) - np.expand_dims(source_times, axis=0)

        right = thresh_ms[0] <= tdiffs
        left = tdiffs <= thresh_ms[1]
        classifies = np.logical_and(right, left)
        targets, sources = np.where(classifies)

        effective_sources.append(binning_find_idcs(source_binning, i)[sources])
        effective_targets.append(binning_find_idcs(target_binning, j)[targets])

    # each bin with itself
    for bin_idx0, bin_idx1 in zip(all_bin_ids, all_bin_ids):
        pairwise_connections(bin_idx0, bin_idx1)

    # each bin with the next one
    if thresh_ms[1] > 0:
        for bin_idx0, bin_idx1 in zip(all_bin_ids[:-1], all_bin_ids[1:]):
            pairwise_connections(bin_idx0, bin_idx1)

    # each bin with the previous one
    if thresh_ms[0] < 0:
        for bin_idx0, bin_idx1 in zip(all_bin_ids[1:], all_bin_ids[:-1]):
            pairwise_connections(bin_idx0, bin_idx1)

    return effective_sources, effective_targets


def extract_spike_to_spike_jumps_df(source_spikes, target_spikes, thresh_ms=(.5, 100.), tbin_ms=None):
    """
        extract "effective" connections by detecting spikes that happen at least once
        sequentially within a small time window

        We use partitioning strategy. Classify spikes in time bins big enough that two spikes can only
        be below threshold if they are in the same or consecutive bins.

        The parameters source_spikes and target_spikes indicate candidates to be in jumps.
        For example: you may want to pass inhibitory spikes only as "target" candidates
        (because inhibitory ones don't propagate activity so can't be "source") and externally induced
        spikes only as "source".

    :param source_spikes: DataFrame with columns: time
    :param target_spikes: DataFrame with columns: time
    :param thresh_ms: tuple representing a time difference window relative to spike 0 in which spike 1 must lie
        for the jump 0->1 to be considered
    :param tbin_ms:
    :return:
    """
    tbin_ms = tbin_ms if tbin_ms is not None else thresh_ms[1]
    assert tbin_ms >= thresh_ms[1]

    effective_sources, effective_targets = extract_spike_to_spike_jumps_jit(
        source_spikes.index.values, source_spikes['time'].values,
        target_spikes.index.values, target_spikes['time'].values,
        thresh_ms, tbin_ms)

    effective_conns = pd.DataFrame.from_dict({
        'source_spike': np.concatenate(effective_sources),
        'target_spike': np.concatenate(effective_targets),
        })

    effective_conns.drop_duplicates(inplace=True)
    effective_conns.index.name = 'jump_idx'

    return effective_conns


def map_detailed_spike_jumps(spike_jumps, source_spikes, target_spikes):
    """
    :param spike_jumps:
    :param source_spikes:
    :param target_spikes:
    :return: df looks like
                      source_gid  source_spike  source_time  target_gid  target_spike  target_time
        jump_idx
        0              96793        520022       1084.0       75034        539793       1085.3
        1              46437        120715       1083.1       75034        539793       1085.3
        2              79287        479575       1082.8       75034        539793       1085.3
        3                280        499566       1079.1       75034        539793       1085.3
        4             212366        299617       1079.5       75034        539793       1085.3
    """

    # merge source spike
    a = pd.merge(
        spike_jumps, source_spikes[['gid', 'time']], how='left',
        left_on='source_spike', right_index=True)

    # merge target spike
    detailed_jumps = pd.merge(
        a, target_spikes[['gid', 'time']], how='left',
        left_on='target_spike', right_index=True,
        suffixes=['_source', '_target'])

    # pandas only allows suffixes but it feels more natural to deal with prefix (source and target act as adjectives)
    return detailed_jumps.rename(columns=dict(
        gid_source='source_gid',
        gid_target='target_gid',
        time_source='source_time',
        time_target='target_time',
    )).sort_index(axis=1)


def map_detailed_spike_jumps_to_conns(detailed_jumps, conns):
    """
    Try to map each spike jump to a connection.
    Result is the detailed_jumps df with extra columns representing connection properties.
    A new conn_index property represents the index in the conns dataframe.
    For those spike jumps that do NOT actually match, connection properties will be nan.

    :param detailed_jumps:
    :param conns:
    :return: df looks like:

                      source_gid  source_spike  source_time  target_gid  target_spike  target_time  conn_index c_type syn_type     weight  delay
        jump_idx
        613623          6842        539961       1827.3        2710        459705       1835.3         NaN    NaN      NaN        NaN    NaN
        2105062          205        280466       2920.7      213397        439987       2921.6         NaN    NaN      NaN        NaN    NaN
        2195160        58828        260761       3071.0       62575        320085       3101.5         NaN    NaN      NaN        NaN    NaN
        3460865        10598        141281       3925.0      168147        400237       3930.9   4902849.0    e2e      e2x   6.054227    1.5
        4835094        98169        521252       4950.3       86620        581105       4960.8  14322135.0    e2e      e2x  67.272276    2.2
        6478792        82114         60481       1893.0       72279        479772       1940.2   7212089.0    e2e      e2x   3.377573    0.7

    """
    # prepare indices
    detailed_jumps.index.name = 'jump_idx'
    detailed_jumps = detailed_jumps.reset_index()

    conns = conns.reset_index().rename(columns=dict(source='source_gid', target='target_gid'))

    # merge connections
    mapped = pd.merge(detailed_jumps, conns, how='left')
    mapped = mapped.set_index('jump_idx')

    return mapped


@numba.njit
def extract_triggered_conns_jit(
        all_sources: np.ndarray, all_targets: np.ndarray,
        valid_sources: np.ndarray, valid_targets: np.ndarray):
    """
    Filter all connections returning only those where both the source AND the target are
    in source_gids and target_gids respectively.
    """
    def _in1d(element, test_elements):
        out = np.empty(element.shape[0], dtype=numba.boolean)
        test_elements_set = set(test_elements)

        for i in numba.prange(element.shape[0]):
            out[i] = element[i] in test_elements_set

        return out

    source_mask = _in1d(all_sources, valid_sources)
    target_mask = _in1d(all_targets, valid_targets)

    mask = np.logical_and(source_mask, target_mask)

    return mask


def extract_traversed_connectivity(
    all_conns, candidate_source_spikes, candidate_target_spikes, thresh_ms=(.5, 100.),
    remove_fake=True,
):
    """Extract the traversed connectivity
    This is the effective filtered connectivity: ie connections that do exist where source and target became active
    in a small time window
    """

    with log_time('extract triggered connectivity', pre=False):
        active_source_gids = np.unique(candidate_source_spikes.gid)
        active_target_gids = np.unique(candidate_target_spikes.gid)

        mask = extract_triggered_conns_jit(
            all_conns['source'].values,
            all_conns['target'].values,
            active_source_gids,
            active_target_gids)

        triggered_conns = all_conns[mask]

    with log_time('extract jumps', pre=False):
        spike_jumps = extract_spike_to_spike_jumps_df(
            candidate_source_spikes,
            candidate_target_spikes,
            thresh_ms=thresh_ms
        )

        detailed_spike_jumps = map_detailed_spike_jumps(
            spike_jumps,
            candidate_source_spikes,
            candidate_target_spikes,
        )

        if remove_fake:
            # drop connections that do not exist
            ct = conn_check.ConnectionTester(
                100_000,
                triggered_conns['source'].values,
                triggered_conns['target'].values,
            )

            exist = ct.check(
                detailed_spike_jumps['source_gid'].values,
                detailed_spike_jumps['target_gid'].values,
            )

            detailed_spike_jumps = detailed_spike_jumps.loc[exist]

    with log_time('map jumps to triggered connectivity', pre=False):
        c_jumps = map_detailed_spike_jumps_to_conns(detailed_spike_jumps, triggered_conns)

    # We can drop jumps that contain NaN for conn_idx
    # There will be a lot of jumps that we collected that can't be mapped to actual connections
    # because they are possibly bogus (or poly-synaptic) jumps.
    c_jumps = c_jumps.dropna()

    return c_jumps


def extract_spike_jumps(spikes: pd.DataFrame, conns: pd.DataFrame, thresh_ms=(.5, 100.)):
    """
    Build a detailed DF containing all of the spike jumps in this simulation.
    It ignores jumps onto spikes that are marked as induced.

    :param spikes:
    :param conns:
    :return: see _complement_spike_jumps
    """

    raw_conn_jumps = extract_traversed_connectivity(
        conns,
        candidate_source_spikes=spikes[spikes.ei_type == 'e'],
        candidate_target_spikes=spikes,
        thresh_ms=thresh_ms,
    )

    conn_jumps = drop_induced_spikes_jumps(raw_conn_jumps, spikes)

    return conn_jumps


def drop_induced_spikes_jumps(conn_jumps, detailed_spikes):
    """
    Remove the jumps TO detailed_spikes that were induced. These are spurious jumps.

    :param conn_jumps:
    :param detailed_spikes:
    :return:
    """
    induced_spikes = detailed_spikes[detailed_spikes.is_induced]
    conn_jumps['source_induced'] = conn_jumps.source_spike.isin(induced_spikes.index)

    # detected jumps TO induced detailed_spikes are not real propagation
    target_induced = conn_jumps.target_spike.isin(induced_spikes.index)
    # print(np.count_nonzero(target_induced), 'spurious jumps')

    conn_jumps = conn_jumps[~target_induced].copy()

    return conn_jumps
