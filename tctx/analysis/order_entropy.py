from tctx.analysis import simbatch as sb
from tqdm.auto import tqdm as pbar
from tctx.analysis import amat as am

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def extract_batch(
        batch_original, sim_gids,
        min_participation=None,
        rank_method='first', ei_type='all', repeats=100,
        strict_trial_count=False,
        adjust_missing=True,
):

    all_foll_gids = batch_original.reg['e_foll_gids'] + batch_original.reg['i_foll_gids']

    res = {}

    for sim_gid in pbar(sim_gids, desc='sims'):

        trial_count = batch_original.reg.loc[sim_gid, 'trial_count']
        foll_gids = all_foll_gids.loc[sim_gid]

        all_spikes = batch_original.stores['spikes'][sim_gid]
        sb.CAT.add_cats_spikes(all_spikes)

        spks = am.SpikeBundle(all_spikes)

        spks = spks.sel(cat='effect').sel_isin(gid=foll_gids).sel_first_per_trial_and_cell()
        spks = spks.sel(ei_type=ei_type)

        if min_participation is not None and len(spks) > 0:
            amat = spks.get_activation_matrix(n_trials=trial_count)
            valid_trials = amat.sel_trials_min_participation(min_participation)

            # we dont' store trial index per spike, but window index
            # there are 2 windows per trial: baseline & effect
            # we're interested in effect spikes
            spks = spks.sel_isin(win_idx=valid_trials * 2 + 1)

            trial_count = len(valid_trials)

        if len(spks) > 0:

            if not strict_trial_count or len(spks.get_gids()) < trial_count:

                true_entropy = spks.calc_order_entropy(
                    trial_count,
                    rank_method=rank_method,
                    adjust_missing=adjust_missing,
                )

                shuffled_entropies = spks.calc_order_entropy_shuffled(
                    trial_count,
                    rank_method=rank_method,
                    adjust_missing=adjust_missing,
                    repeats=repeats,
                    show_pbar=False,
                )

                res[sim_gid] = pd.concat([
                    true_entropy.to_frame().T.set_index(np.array([-1])),
                    shuffled_entropies
                ], axis=0)

            else:
                print(f'skipping {sim_gid} because of fewer trials {trial_count} than followers {len(spks.get_gids())}')

        else:
            print(f'skipping {sim_gid} because of no valid foll spikes')

    return res


def load_batch(batch, col) -> tuple:

    all_entropies = dict(pbar(
        batch.stores[f'order_entropy_{col}'].items(),
        total=len(batch.stores[f'order_entropy_{col}'].keys())
    ))

    all_true_entropies = pd.DataFrame.from_dict({
        sim_gid: entropies.loc[-1]
        for sim_gid, entropies in all_entropies.items()
    }, orient='index')

    all_true_entropies.rename_axis(index='sim_gid', columns='trial_rank', inplace=True)
    all_true_entropies.fillna(1, inplace=True)

    all_mean_shuffles = pd.DataFrame.from_dict({
        sim_gid: entropies.loc[0:].mean()
        for sim_gid, entropies in all_entropies.items()
    }, orient='index')

    all_mean_shuffles.rename_axis(index='sim_gid', columns='trial_rank', inplace=True)
    all_mean_shuffles.fillna(1, inplace=True)

    return all_true_entropies, all_mean_shuffles


def plot_labeled_mat(mat: pd.DataFrame, ax=None, cmap='seismic', aspect='equal', norm=None, rotation=45):
    if ax is None:
        _, ax = plt.subplots(constrained_layout=True, figsize=(2, 2))

    im = ax.imshow(
        mat,
        cmap=cmap,
        norm=norm,
        aspect=aspect,
        origin='lower',
    )

    for ax_labels, axis in [(mat.index, ax.yaxis), (mat.columns, ax.xaxis)]:
        if not np.issubdtype(ax_labels.dtype, np.number):
            ax_labels = list(map(lambda x: x.replace('_', ' '), ax_labels))
            axis.set_ticks(np.arange(len(ax_labels)))
            axis.set_ticklabels(ax_labels)

    ax.tick_params(rotation=rotation, left=False, bottom=False, labelsize=6)

    ax.set_ylim(-.5, mat.shape[0]-.5)
    ax.set_xlim(-.5, mat.shape[1]-.5)

    return im
