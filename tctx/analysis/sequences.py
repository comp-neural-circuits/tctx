"""
Code to study the result of sequence experiments, where a randomly chosen cell is repeateadly activated.

The main function to post-process simulation results is:
    compute_sequence_details_batch
"""

import numpy as np
import pandas as pd
import scipy.stats as st
from pathlib import Path
from tqdm.auto import tqdm as pbar

from tctx.util import spike_trains as spt
import tctx.util.parallel
from tctx.analysis import simbatch as sb
from tctx.analysis.simbatch import CAT


MS_TO_S = 0.001
DEFAULT_FRM_THRESHOLD_PROB = 1. / 10 ** 7

DEFAULT_EFFECT_LENGTH_MS = 300


########################################################################################################################
# METRICS

def _compute_frm(ewins, cells, spikes):
    """
    compute, for every cell in the experiment, their firing rate change in Hz
    An entry per cell is guaranteed.

    :return: a DF that looks like

               effect   pre     frm
        gid
        0       0.1  0.10     0.00
        1       0.0  0.10    -0.10
        ...     ...   ...      ...
        99998   2.0  2.60    -0.60
        99999   3.1  2.75     0.35

    """
    counts = ewins.count_spikes(spikes)
    counts = counts.reindex(cells.index)
    counts.fillna(0, inplace=True)
    counts.index = cells.index
    counts.index.name = 'gid'

    total_time_s = ewins.get_length_by_cat() * MS_TO_S
    fr = counts / total_time_s

    fr.columns.name = ''
    fr.name = 'hz'

    fr['frm'] = fr['effect'] - fr['baseline']

    return fr


def _compute_delay_std(delays, min_spike_count):
    """
    Compute std of temporal delays of spikes classified in windows after an event.
    We use this as a metric for lack of precision.

    We could instead use the reciprocal of the variance (which is called "precision")
    but the scale then becomes problematic to visualize: maximum precision is infinite.

    Note that the std doesn't make sense for cells that spiked only once and is not
    representative for those that spiked very few times. That is why we first filter
    for cells with a minimum number of spikes. Those cells will have std "nan"

    :param delays: DF containing 'gid' and 'delay'
    :param min_spike_count: minimum acceptable number of spikes
    :return: pd.Series
    """

    spike_count = delays.groupby('gid')['delay_in_window'].count()

    mask = spike_count >= min_spike_count

    stds = delays.groupby('gid')['delay_in_window'].std()

    return pd.Series(stds[mask]).rename('delstd')


def _compute_spike_delays(sim_params, spikes, induced_spike_times):
    """
    the delay to the closest preceding induced spike even for the spikes that happened in a "baseline" window.
    :return: a pd.Series with the delay per spike
    """
    ind_wins = spt.ExclusiveWindows.build_between(induced_spike_times, sim_params.tstart, sim_params.tend)
    spikes_delay_from_induced = ind_wins.classify_spikes(spikes).delay
    return spikes_delay_from_induced


########################################################################################################################
# PROTOCOL

def get_trial_idx_from_win_idx(spikes, col='win_idx'):
    """
    We number 100 trials 0-99. Spikes outside of trials will get -1 or 100 (before or late).
    This relies on "win_idx" being present, which is computed in sequence analysis
    (windows being consecutive trial & baseline periods)
    """
    return np.ceil(spikes[col].dropna() / 2).astype(np.int) - 1


def define_experiment_windows(induced_times, start=None, stop=None, win=(0, +200)):
    """
    :param induced_times: np.array or series that represents the start of each trial
    :param start: (ms) beginning of experiment
    :param stop: (ms) end of experiment
    :param win: time pair that defines where we look for the effect relative to each induced spike
    :return: spt.ExclusiveWindows with two categories: 'baseline' and 'effect'.
    These may NOT cover the entire experiment if induced_times are too close,
    leaving gaps where spikes will be ignored. This is important for experimental data.
    """

    induced_windows_raw = spt.make_windows(induced_times, win)

    # anything else is "baseline"
    baseline_wins = spt.invert_windows(induced_windows_raw, start=start, stop=stop)
    assert spt.are_windows_exclusive(baseline_wins)

    # our induced_windows_raw may overlap and contain multiple induced spikes (if they are closer than "win")
    # discard those so we have "clean windows"
    # Note that we do this AFTER computing the baseline windows to avoid having induced spikes there
    # This means that our new "effect_wins" and "baseline_wins" may NOT cover the entire experiment
    effect_wins = spt.filter_windows_exclusive_ref(induced_windows_raw)
    assert spt.are_windows_exclusive(effect_wins)

    baseline_wins['cat'] = 'baseline'
    effect_wins['cat'] = 'effect'
    all_wins = pd.concat([baseline_wins, effect_wins], axis=0)
    all_wins.sort_values(['start', 'stop', 'ref'], inplace=True)
    all_wins.reset_index(drop=True, inplace=True)
    all_wins.index.name = 'win_idx'

    assert spt.are_windows_exclusive(all_wins)

    all_wins = spt.ExclusiveWindows(all_wins, by='cat')

    return all_wins


########################################################################################################################
# PROCESS


def _collect_induced_spikes(spikes, input_targeted_times, trial_length_ms, targeted_gid):
    """
    The targeted cell may fire multiple times due to recurrent excitatory connections.
    It may also fail to fire or fire with a random delay due to excitatory inhibitory connections.
    This tags spikes as "induced" only if they are the first per trial window and within a few milliseconds
    :return: a boolean series matching the spikes index
    """
    inter_induction_wins = spt.make_windows(input_targeted_times, (0, trial_length_ms))
    inter_induction_wins = spt.ExclusiveWindows(inter_induction_wins)
    targeted_spikes = spikes[spikes.gid == targeted_gid]
    targeted_spikes = inter_induction_wins.classify_spikes(targeted_spikes)

    targeted_spikes = targeted_spikes[targeted_spikes['delay'] < 10.]
    induced_spk_idcs = targeted_spikes.groupby('win_idx')['delay'].idxmin().values

    is_induced = pd.Series(np.zeros(len(spikes), dtype=np.bool_), index=spikes.index)
    is_induced.loc[induced_spk_idcs] = True

    return is_induced


def compute_sequence_details(
        sim_params, cells, spikes,
        effect_length_ms=DEFAULT_EFFECT_LENGTH_MS,
        delstd_min_spike_count=5,
        trial_times_col='input_targeted_times',
):
    """
    Compute multiple metrics for cells and for spikes, return as two DF
    """

    is_induced_spike = _collect_induced_spikes(
        spikes,
        sim_params[trial_times_col],
        sim_params['trial_length_ms'],
        sim_params['targeted_gid'])

    induced_spikes = spikes[is_induced_spike]

    exp_wins = define_experiment_windows(
        induced_spikes.time,
        sim_params.tstart, sim_params.tend,
        win=(0, effect_length_ms)
    )

    frm = _compute_frm(exp_wins, cells, spikes)
    frm = frm.rename(columns=dict(baseline='fr_baseline', effect='fr_effect'))

    delays = exp_wins.classify_spikes(spikes).rename(columns=dict(delay='delay_in_window'))

    delays['delay_from_induced'] = _compute_spike_delays(sim_params, spikes, induced_spikes.time)

    delstd = _compute_delay_std(delays[delays.cat == 'effect'], min_spike_count=delstd_min_spike_count)

    detailed_cells = pd.concat([cells, frm], sort=True, axis=1)
    detailed_cells['delstd'] = delstd
    detailed_cells['spike_count'] = spikes.groupby('gid')['time'].count()
    detailed_cells['spike_count'].fillna(0, inplace=True)

    # normalize delstd relative to the standard deviation of uniformly distributed delays
    # random_delstd = np.sqrt((effect_length_ms - 0) ** 2 / 12)
    # detailed_cells['delstd_norm'] = delstd / random_delstd

    # normalize frm relative to 1 spike per trial over 0 on non-trial
    plus_one_frm = 1. / (effect_length_ms * MS_TO_S)
    detailed_cells['frm_norm'] = detailed_cells['frm'] / plus_one_frm

    detailed_cells['is_targeted'] = detailed_cells.index == sim_params.targeted_gid

    detailed_spikes = pd.merge(spikes, delays.drop('gid', axis=1), left_index=True, right_index=True, how='left')
    detailed_spikes['is_induced'] = is_induced_spike

    detailed_spikes = pd.merge(
        detailed_spikes,
        detailed_cells[['ei_type']],
        left_on='gid', right_index=True, how='left',
    )

    detailed_spikes['trial_idx'] = get_trial_idx_from_win_idx(detailed_spikes, col='win_idx')

    return exp_wins, detailed_cells, detailed_spikes


def _collect_foll(all_detailed_cells, targeted_gids: pd.Series):
    """
    Collect all follower gids for each simulation, differentiating by ei_type

    :returns: df like:

                         e_foll_gids          i_foll_gids  e_foll_count  i_foll_count
        sim_gid
        0        (2118, 3486, 591...  (96852, 99575, 9...            42             3
        1        (553, 2118, 2240...  (93252, 93621, 9...            68            12
        2        (553, 2118, 2240...  (93359, 93621, 9...           125            21
        3        (5917, 24730, 48...                   ()             5             0
        4        (1162, 2240, 348...  (93213, 93621, 9...            80            21
        ...                      ...                  ...           ...           ...
        11032    (4379, 41169, 46...  (94603, 98130, 9...             4             3
        11033    (4379, 41169, 46...             (99221,)             4             1
        11034    (1882, 4589, 571...  (93164, 95562, 9...            62             6
        11035    (20517, 23404, 2...  (94550, 98253, 9...             7             3
        11036    (410, 3127, 5958...             (98615,)            18             1

    """
    all_foll_gids = {}

    for sim_gid, cells in pbar(all_detailed_cells.items(), total=len(all_detailed_cells), desc='sim'):
        targeted_gid = targeted_gids.loc[sim_gid]

        sb.CAT.add_cats_cells(cells)
        cells = cells.drop(targeted_gid)

        foll_ei_types = cells.loc[cells['frm_cat'] == 'foll', 'ei_type']

        all_foll_gids[sim_gid] = {
            f'{ei_type}_foll_gids': tuple(gids)
            for ei_type, gids in foll_ei_types.groupby(foll_ei_types).groups.items()}

    all_foll_gids = pd.DataFrame.from_dict(all_foll_gids, orient='index')

    # fillna doesn't like taking empty tuples
    for col, values in all_foll_gids.items():
        all_foll_gids.loc[all_foll_gids[col].isna(), col] = tuple()

    all_foll_gids = all_foll_gids.rename_axis(index='sim_gid')

    foll_counts = all_foll_gids.applymap(len)
    foll_counts.columns = [f'{col[0]}_foll_count'for col in foll_counts]

    all_foll_gids = pd.concat([all_foll_gids, foll_counts], axis=1)

    return all_foll_gids


def compute_sequence_details_batch(
        batch,
        batch_folder: str,
        effect_length_ms=DEFAULT_EFFECT_LENGTH_MS,
        delstd_min_spike_count=5,
        trial_times_col='forced_times',
        threshold_prob=DEFAULT_FRM_THRESHOLD_PROB,
        max_workers=None,
        exec_mode=None,
):
    """
    Compute the same as compute_sequence_details but for multiple experiments.
    Results are stored under the given folder and added to the batch registry.
    Batch should contain cells_raw and spikes_raw, which can be automatically added for new sims like:

        batch.register_raw()

    :return: a copy of the batch with references to the stored exp_wins, cells, spikes
    """
    batch_folder = Path(batch_folder)

    sim_gids = batch.reg.index

    res = tctx.util.parallel.independent_tasks(
        compute_sequence_details,
        [
            (
                batch.reg.loc[sim_gid],
                batch.stores['cells_raw'][sim_gid],
                batch.stores['spikes_raw'][sim_gid],
                effect_length_ms,
                delstd_min_spike_count,
                trial_times_col,

            )
            for sim_gid in pbar(sim_gids, desc='load')
        ],
        max_workers=max_workers,
        mode=exec_mode,
    )

    all_exp_wins = {}
    all_detailed_cells = {}
    all_detailed_spikes = {}

    for i, r in pbar(res.items(), desc='remove cats'):
        all_exp_wins[sim_gids[i]] = r[0]

        CAT.remove_cats_cells(r[1])
        all_detailed_cells[sim_gids[i]] = r[1]

        CAT.remove_cats_spikes(r[2])
        all_detailed_spikes[sim_gids[i]] = r[2]

    all_cmf = compute_frm_norm_cmf_multisim(
        sim_gids, all_exp_wins, all_detailed_spikes, all_detailed_cells, effect_length_ms)

    # this modifies each dataframe of 'all_detailed_cells' inplace
    classify_cells_by_frm_null_dist_multisim(sim_gids, all_cmf, all_detailed_cells, threshold_prob=threshold_prob)

    all_foll_gids = _collect_foll(all_detailed_cells, batch.reg['targeted_gid'])

    for col, values in all_foll_gids.items():
        batch.reg[col] = values

    batch.register_and_save(batch_folder, 'cells', all_detailed_cells)
    batch.register_and_save(batch_folder, 'spikes', all_detailed_spikes)
    batch.register_and_save(batch_folder, 'ewins', {sim_gid: ewin.windows for sim_gid, ewin in all_exp_wins.items()})

    return batch


########################################################################################################################
# NULL DISTRIBUTION


def _sample_from_null_frm_dist(mean_spike_count, total_baseline_time, total_effect_time, sample_size=10 ** 6):
    """
    Our null distribution is that which says that the firing rate of the cell is NOT affected
    by the injected spike.
    In that case, the firing RATE in the "baseline" and "effect" windows is the same.
    However, the actual COUNT of spikes in those windows will be different because of stochasticity
    and even the mean may differ if the windows have different lengths.
    Generate 2 independent poisson counts, convert them to rates and substract them.
    Notice this is similar to a Skellam distribution except that, because we conver to rates, we are
    scaling both poissons before subtracting. This makes the values not integer
    (although still discrete due to the integer-based underlying poisson) and non-contiguous.
    A poisson RV scaled is no longer poisson.
    """
    total_time = total_baseline_time + total_effect_time

    samples = (
            st.poisson(mean_spike_count * total_effect_time / total_time).rvs(sample_size) / total_effect_time
            -
            st.poisson(mean_spike_count * total_baseline_time / total_time).rvs(sample_size) / total_baseline_time
    )

    # convert 1/ms to 1/s (Hz)
    samples = samples / MS_TO_S

    return samples


def _estimate_frm_norm_cmf(
        baseline_total: float,
        effect_total: float,
        mean_spike_count: float,
        plus_one_frm: float,
        cmf_repeat_count=50, sample_size=10 ** 5
):
    cmf_bins = np.linspace(-1, 2, 1001)

    multiple_cmfs = []
    for i in range(cmf_repeat_count):
        null_samples = _sample_from_null_frm_dist(
            mean_spike_count,
            baseline_total, effect_total,
            sample_size,
        )

        null_samples = null_samples / plus_one_frm

        h = np.histogram(null_samples, cmf_bins)[0]
        multiple_cmfs.append(
            np.cumsum(h / np.sum(h))
        )

    cmf = np.mean(multiple_cmfs, axis=0)
    # noinspection PyUnresolvedReferences
    cmf = pd.Series(
        cmf,
        index=pd.IntervalIndex.from_breaks(cmf_bins)
    )

    return cmf


def compute_frm_norm_cmf_multisim(
        sim_gids,
        all_exp_wins, all_detailed_spikes, all_detailed_cells,
        effect_length_ms=DEFAULT_EFFECT_LENGTH_MS
):
    """
    Takes 1h 37 min for 3572 sims

    :return: pd.DataFrame that looks like:

        sim_gid          2188 335  964       2773 29
        ei_type             e    i    i         i    i
        frm_norm
        (-0.4, -0.397]    0.0  0.0  0.0  0.000221  0.0
        (-0.397, -0.394]  0.0  0.0  0.0  0.000256  0.0
        (-0.394, -0.391]  0.0  0.0  0.0  0.000256  0.0
        (-0.391, -0.388]  0.0  0.0  0.0  0.000298  0.0
        (-0.388, -0.385]  0.0  0.0  0.0  0.000345  0.0

    Every column is a Cumulative Mass Function of the null distribution of normalised Firing Rate Modulation
    of the given simulation and ei-population. These are generated by sampling from the weighted difference of
    two poissons (trial & baseline) assuming that the rate is the same although the length of the time windows
    may not.
    See sample_null_frm_dist
    """
    plus_one_frm = 1. / (effect_length_ms * MS_TO_S)

    params = []
    index = []

    for sim_gid in pbar(sim_gids, desc='null cmf'):
        for ei_type_code, ei_type_name in enumerate(CAT.category_types['ei_type'].categories):
            cells = all_detailed_cells[sim_gid]
            assert np.issubdtype(cells['ei_type'].dtype, np.number)

            spikes = all_detailed_spikes[sim_gid]
            assert np.issubdtype(spikes['ei_type'].dtype, np.number)

            spikes = spikes[spikes.ei_type == ei_type_code]
            cells = cells[cells.ei_type == ei_type_code]

            ewins = all_exp_wins[sim_gid]
            cat_times = ewins.get_length_by_cat()
            baseline_total, effect_total = cat_times['baseline'], cat_times['effect']

            spike_counts = spikes.groupby('gid')['time'].count().reindex(cells.index).fillna(0)

            assert len(cells) > 0

            index.append((sim_gid, ei_type_name))
            params.append(
                (baseline_total, effect_total, np.mean(spike_counts), plus_one_frm)
            )

    all_cmf = tctx.util.parallel.independent_tasks(_estimate_frm_norm_cmf, params)
    all_cmf = {index[i]: r for i, r in pbar(all_cmf.items())}

    all_cmf = pd.concat(all_cmf, axis=1)
    all_cmf.columns.names = ['sim_gid', 'ei_type']
    all_cmf.index.name = 'frm_norm'

    return all_cmf


def take_threshold_from_cmf(cmf: pd.Series, threshold_prob=DEFAULT_FRM_THRESHOLD_PROB):
    # noinspection PyTypeChecker
    return (
        (cmf > threshold_prob).idxmax().left,
        (cmf <= (1. - threshold_prob)).idxmin().right
    )


def classify_by_frm_null(frm_norm: pd.Series, cmf: pd.Series, threshold_prob=DEFAULT_FRM_THRESHOLD_PROB):
    thresholds = take_threshold_from_cmf(cmf, threshold_prob=threshold_prob)

    frm_cat = pd.Series(np.digitize(frm_norm, thresholds), index=frm_norm.index)

    frm_cat = frm_cat.map({0: 'anti', 1: 'bkg', 2: 'foll'})

    return frm_cat


def classify_cells_by_frm_null_dist_multisim(sim_gids, all_cmf, all_detailed_cells,
                                             threshold_prob=DEFAULT_FRM_THRESHOLD_PROB):
    """
    Use the CMF (cumulative mass function) representing the null distribution of the firing rate modulation
    to classify every cell as anti-follower ('anti'), background ('bkg'), and follower ('foll'), depending on
    two thresholds taken at the two extremes of the CMF

    This function will modify all_detailed_cells by adding a new col 'frm_cat'
    """

    for i, sim_gid in enumerate(pbar(sim_gids, desc='frm_cat')):
        for ei_type_code, ei_type_name in enumerate(CAT.category_types['ei_type'].categories):
            cells = all_detailed_cells[sim_gid]
            assert np.issubdtype(cells['ei_type'].dtype, np.number)
            cells = cells[cells.ei_type == ei_type_code]

            frm_cat = classify_by_frm_null(cells['frm_norm'], all_cmf[sim_gid, ei_type_name], threshold_prob)

            all_detailed_cells[sim_gid].loc[frm_cat.index, 'frm_cat'] = frm_cat
