"""
Extract summary stats about multiple simulations at once
"""

from tqdm.auto import tqdm as pbar
import numpy as np
import pandas as pd
from tctx.networks.turtle_data import DEFAULT_ACT_BINS
import logging


def extract_spike_counts_per_ei_type(cells, spikes, wins: pd.DataFrame):
    """
    count number of spikes of e/i type in each of the windows

    wins are expected to be monotonically increasing, contiguous and exclusive
    """

    is_exc = cells['ei_type'].reindex(spikes['gid']).values == 'e'
    ts = spikes['time'].values
    t_edges = np.concatenate([wins['start'].values, wins['stop'].values[-1:]])

    counts = pd.DataFrame.from_dict({
        'e': np.histogram(ts[is_exc], bins=t_edges)[0],
        'i': np.histogram(ts[~is_exc], bins=t_edges)[0],
    }, orient='columns').set_index(wins.index)

    counts['total'] = counts['e'] + counts['i']

    counts.rename_axis(columns='ei_type', index='win_idx', inplace=True)

    return counts


def extract_spike_count_per_sim(sim_gids, all_cells, all_spikes, wins) -> pd.DataFrame:
    """
    :returns:
        spike_count     e         i
        sim_gid
        0           19344     40644
        1           20341     43165
        2           13062     17942
    """
    spike_counts = {}

    from tctx.analysis import simbatch as sb

    for sim_gid in pbar(sim_gids, desc='spikes per sim'):
        spikes = all_spikes[sim_gid]

        # we do it like this because spikes may not have e/i info
        cells = all_cells[sim_gid]
        sb.CAT.add_cats(cells)

        counts = extract_spike_counts_per_ei_type(cells, spikes, wins)
        counts = counts.groupby(wins['cat']).sum()
        counts.loc['induction'] = counts.loc['baseline'] + counts.loc['effect']
        spike_counts[sim_gid] = counts.stack()

    spike_counts = pd.DataFrame.from_dict(spike_counts, orient='index')
    spike_counts.rename_axis(index='sim_gid', columns=['cat', 'ei_type'], inplace=True)

    return spike_counts


def extract_cell_counts_per_sim(all_cells: dict, sim_gids=None):
    """
    :param all_cells: dict-like of <sim_gid, cells: pd.DataFrame>
    :return:

        cell_count   e      i   total
        sim_gid
        0        89286  10714  100000
        1        89286  10714  100000
        2        89286  10714  100000
    """
    if sim_gids is None:
        sim_gids = list(all_cells.keys())

    sim_gids = pbar(sim_gids, total=len(sim_gids), desc='count cells')

    cell_counts = {}

    from tctx.analysis import simbatch as sb

    for sim_gid in sim_gids:
        cells = all_cells[sim_gid]
        sb.CAT.add_cats(cells)
        cell_counts[sim_gid] = cells['ei_type'].astype(str).value_counts()

    cell_counts = pd.DataFrame.from_dict(cell_counts, orient='index')

    assert cell_counts.columns.is_unique, cell_counts.columns

    cell_counts = cell_counts.reindex(['e', 'i'], axis=1, fill_value=0).fillna(0).astype(np.int)
    cell_counts.rename_axis(index='sim_gid', columns='cell_count', inplace=True)

    cell_counts['total'] = cell_counts.sum(axis=1)

    return cell_counts


def get_protocol_wins(reg: pd.DataFrame, trial_times='forced_times'):
    """
    Extract pre/induction/post windows that define the protocol of each sim
    so we can extract statistics separately
    """

    if isinstance(trial_times, str):
        trial_times = reg[trial_times].drop_duplicates()
        assert len(trial_times) == 1, f'Expected 1 protocol. Got: {len(trial_times)}'
        trial_times = trial_times.iloc[0]

    prepost = reg[[f'tstart_pre', f'tstop_pre', f'tstart_post', f'tstop_post']].drop_duplicates()
    assert len(prepost) == 1
    prepost = prepost.iloc[0]

    wins = [{'start': prepost[f'tstart_pre'], 'stop': prepost[f'tstop_pre'], 'cat': 'pre'}]

    baseline_duration = 100
    effect_duration = 300

    for t in trial_times:
        wins.append({
            'start': t - baseline_duration,
            'stop': t,
            'cat': 'baseline',
        })
        wins.append({
            'start': t,
            'stop': t + effect_duration,
            'cat': 'effect',
        })

    wins.append({'start': prepost[f'tstart_post'], 'stop': prepost[f'tstop_post'], 'cat': 'post'})

    wins = pd.DataFrame.from_records(wins)

    edges = wins[['start', 'stop']].sort_values(['start', 'stop']).values.flatten()
    assert np.all(np.diff(edges) >= 0), 'wins should be exclusive'

    edges = wins[['start', 'stop']].sort_values(['start', 'stop'])
    assert (edges['start'].values[1:] - edges['stop'][:-1]).max() <= 0, 'wins should be tight'

    return wins


def tag_sims_by_level(stats: pd.DataFrame, hz_col='cell_hz_baseline_total') -> pd.DataFrame:
    """add a column to indicate a sim is compatible with different levels of activity"""
    tags = {}

    for act, vrange in DEFAULT_ACT_BINS.items():
        col = f'{act}_compatible'
        tags[col] = stats[hz_col].between(*vrange)

    return pd.DataFrame.from_dict(tags)


def extract_simstats(reg: pd.DataFrame, all_cells: dict, all_spikes: dict, wins: pd.DataFrame = None):
    """
    call like:

        stats = simstats.extract_simstats(
            batch.reg,
            batch.stores['cells_raw'],
            batch.stores['spikes_raw'],
        )

    :param reg:
    :param all_cells:
    :param all_spikes:
    :param wins:

    :returns:
        a df like:

            spike_count  spike_count_e  spike_count_i  ...  cell_hz_i  cell_hz_total
            sim_gid                                    ...
            0                    23142          22247  ...   0.075670       0.010807
            1                     3173           2289  ...   0.007786       0.001300
            2                        0              0  ...   0.000000       0.000000
            3                        0              0  ...   0.000000       0.000000

        see STATS_COLS
    """
    if wins is None:
        wins = get_protocol_wins(reg)

    sim_gids = reg.index
    sim_gids = sim_gids.intersection(list(all_cells.keys()))
    sim_gids = sim_gids.intersection(list(all_spikes.keys()))

    cell_counts = extract_cell_counts_per_sim(all_cells, sim_gids=sim_gids)

    ms_to_s = .001

    wins_duration = wins['stop'] - wins['start']

    cat_durations_ms = wins_duration.groupby(wins['cat']).sum()
    cat_durations_ms.loc['induction'] = cat_durations_ms.loc['baseline'] + cat_durations_ms.loc['effect']

    cat_durations_s = cat_durations_ms * ms_to_s

    # This is a DF with a multi-index for columns
    # Shape is <sims, <win_cat, ei_type>>
    spike_counts_per_sim = extract_spike_count_per_sim(sim_gids, all_cells, all_spikes, wins)

    # Normalize by time (depends on window category)
    cats_cols = spike_counts_per_sim.columns.get_level_values('cat')
    sim_pop_hz = spike_counts_per_sim / cat_durations_s.reindex(cats_cols).values

    # Normalize by population size (depends on ei_type and on simulation)
    ei_types_cols = sim_pop_hz.columns.get_level_values('ei_type')
    sim_cell_hz = sim_pop_hz / cell_counts[list(ei_types_cols)].values

    # flatten the hierarchical index (cat, ei_type)
    spike_counts_per_sim.columns = spike_counts_per_sim.columns.map(lambda x: f'{x[0]}_{x[1]}')
    sim_pop_hz.columns = sim_pop_hz.columns.map(lambda x: f'{x[0]}_{x[1]}')
    sim_cell_hz.columns = sim_cell_hz.columns.map(lambda x: f'{x[0]}_{x[1]}')

    all_stats = [
        cell_counts.add_prefix('cell_count_'),
        spike_counts_per_sim.add_prefix(f'spike_count_'),
        sim_pop_hz.add_prefix(f'pop_hz_'),
        sim_cell_hz.add_prefix(f'cell_hz_'),
    ]

    sim_stats = pd.concat(all_stats, axis=1, sort=True, )
    sim_stats.rename_axis(index='sim_gid', inplace=True)
    sim_stats.name = 'sim_stats'

    tags = tag_sims_by_level(sim_stats)
    for col, values in tags.items():
        sim_cell_hz[col] = values

    if not np.all(sim_stats['spike_count_induction_total'] ==
                  (sim_stats['spike_count_baseline_total'] + sim_stats['spike_count_effect_total'])):
        logging.error(f'Induction window count does NOT match baseline+effect')

    return sim_stats
