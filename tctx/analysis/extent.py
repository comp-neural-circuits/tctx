"""
Extract, for each sequence, metrics about its:
 - How far it goes (distance to furthest follower)
 - How long does it go for (delay to latest average follower)
"""

from tctx.analysis import simbatch as sb
from tctx.util import spike_trains as spt
import pandas as pd
import numpy as np


DESC = 'extent'


def get_foll_spikes(is_foll_cell, spikes, max_delay=400):
    spikes = spikes[spikes['delay_from_induced'].between(0, max_delay)]

    is_foll_spike = is_foll_cell.reindex(spikes.gid.values)

    foll_spikes = spikes[is_foll_spike.values]

    return foll_spikes


def get_is_foll_cell(cells):
    foll_cat_values = cells['frm_cat']

    if np.issubdtype(foll_cat_values.dtype, np.number):
        is_foll_cell = foll_cat_values == sb.CAT.get_cat_code('foll_cat', 'foll')
    else:
        is_foll_cell = (foll_cat_values == 'foll')

    return is_foll_cell & ~cells['is_targeted']


def get_foll_activations(foll_spikes):
    """
    get the selection of spikes that are follower activations (first spike per trial)
    """
    activations = foll_spikes.groupby(['gid', 'win_idx'])['delay_from_induced'].idxmin()
    return foll_spikes.loc[activations.values]


def extract_extent(batch: sb.SimBatch, sim_gid):
    """
    Obtain multiple metrics about follower activations for a single simluation,
    for example, distance to last follower, mean follower temporal jitter, etc..
    :param batch:
    :param sim_gid:
    :return:
    """

    cells: pd.DataFrame = batch.stores['cells'][sim_gid]
    sb.CAT.remove_cats_cells(cells)

    spikes: pd.DataFrame = batch.stores['spikes'][sim_gid]
    sb.CAT.remove_cats_spikes(spikes)

    is_foll_cell = get_is_foll_cell(cells)
    foll_cells = cells[is_foll_cell]

    foll_spikes = get_foll_spikes(is_foll_cell, spikes)

    activations = get_foll_activations(foll_spikes)

    seq_extent = {}

    for ei_type_code, acts in activations.groupby('ei_type'):

        ei_type = sb.CAT.get_cat_name('ei_type', ei_type_code)

        mean_act = acts.groupby('gid')['delay_from_induced'].mean()
        jitt_act = acts.groupby('gid')['delay_from_induced'].std()

        origin = cells.loc[batch.reg.loc[sim_gid, 'targeted_gid'], ['x', 'y']].values.astype(np.float)
        distances = spt.get_wrapped_distance_points(
            xy=foll_cells[['x', 'y']].values.astype(np.float),
            origin=origin,
            side=batch.reg.loc[sim_gid, 'side_um'],
        )
        distances = pd.Series(distances, index=foll_cells.index)

        if len(mean_act) > 0:
            foll_sel = foll_cells[foll_cells['ei_type'] == ei_type_code]
            spks_sel = foll_spikes[foll_spikes['ei_type'] == ei_type_code]

            last_foll = mean_act.idxmax()

            seq_extent[f'{ei_type}_last_foll_activation_time'] = mean_act.loc[last_foll]
            seq_extent[f'{ei_type}_last_foll_activation_jitter'] = jitt_act.loc[last_foll]
            seq_extent[f'{ei_type}_last_foll_distance'] = distances.loc[last_foll]
            seq_extent[f'{ei_type}_mean_foll_activation_jitter'] = jitt_act.mean()

            seq_extent[f'{ei_type}_furthest_follower_distance'] = distances.loc[foll_sel.index].max()
            seq_extent[f'{ei_type}_mean_foll_jitter'] = spks_sel.groupby('gid')['delay_from_induced'].std().mean()

        else:
            seq_extent[f'{ei_type}_last_foll_activation_time'] = np.nan
            seq_extent[f'{ei_type}_last_foll_activation_jitter'] = np.nan
            seq_extent[f'{ei_type}_last_foll_distance'] = np.nan
            seq_extent[f'{ei_type}_mean_foll_activation_jitter'] = np.nan

            seq_extent[f'{ei_type}_furthest_follower_distance'] = np.nan
            seq_extent[f'{ei_type}_mean_foll_jitter'] = np.nan

    seq_extent = pd.Series(seq_extent)

    return seq_extent
