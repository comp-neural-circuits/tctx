"""
Various post-processing steps of batches of simulations.
All steps work incrementally, detecting which sims have already been processed and which haven't.
All steps load and save the batch multiple times.
"""

import logging

from tqdm.auto import tqdm as pbar
import numpy as np
import pandas as pd
import gc

from tctx.util import spike_trains as spt, conn_check
from tctx.analysis import simbatch as sb
from tctx.analysis import simstats
from tctx.analysis import sequences as sqs
from tctx.analysis import extent as ext
from tctx.analysis import traversed_connectivity as tc
from tctx.analysis import order_entropy as oe
from tctx.analysis import branches


def clean_slate(batch_full):
    """Return a brand new clean copy of the batch"""
    batch_full = batch_full.copy_clean_reg().add_cols(batch_full.reg[['full_path', 'sim_idx']])
    batch_full.register_raw()
    return batch_full


def register_raw(working_filename):
    """register raw results as store access"""

    batch_full = sb.SimBatch.load(working_filename)

    missing_raw = batch_full.subsection(batch_full.get_missing('cells_raw_path'))

    if len(missing_raw) > 0:
        missing_raw.register_raw()

        batch_full = batch_full.add_cols(missing_raw.reg[[
            'cells_raw_idx', 'cells_raw_path',
            'spikes_raw_idx', 'spikes_raw_path',
            'conns_raw_idx', 'conns_raw_path',
        ]])

        batch_full.save_registry(working_filename)


def extract_simstats(working_filename):
    """add columns to the registry indicating firing rates and other simulation stats"""
    batch_full = sb.SimBatch.load(working_filename)

    for sbatch in batch_full.iter_missing_chunks('cell_hz_induction_total'):

        ssts = simstats.extract_simstats(
            sbatch.reg,
            batch_full.stores['cells_raw'],
            batch_full.stores['spikes_raw'],
        )

        batch_full = batch_full.add_cols(ssts)

        batch_full.save_registry(working_filename)


def extract_followers(working_filename, res_folder, mask_col=None, exec_mode=None):

    batch_full = sb.SimBatch.load(working_filename)

    new_cols = [
        'e_foll_count', 'e_foll_gids',
        'i_foll_count', 'i_foll_gids',
        'cells_idx', 'cells_path',
        'ewins_idx', 'ewins_path',
        'spikes_idx', 'spikes_path',
    ]

    if mask_col is None:
        chunks = batch_full.iter_missing_chunks('spikes_path')
    else:
        chunks = batch_full.subsection(batch_full.reg[mask_col]).iter_missing_chunks('spikes_path')

    for sbatch in chunks:
        updated_batch = sqs.compute_sequence_details_batch(
            sbatch,
            batch_folder=str(sb.abs_path(res_folder)),
            exec_mode=exec_mode,
        )

        batch_full = batch_full.add_cols(updated_batch.reg[new_cols])

        batch_full.save_registry(working_filename)


def _extract_sim_foll_cells(batch, sim_gid):
    foll_gids = (
            batch.reg.loc[sim_gid, 'e_foll_gids'] +
            batch.reg.loc[sim_gid, 'i_foll_gids']
    )

    cells = batch.stores['cells'][sim_gid]
    folls = cells.loc[list(foll_gids) + [batch.reg.loc[sim_gid, 'targeted_gid']]]

    centered = spt.center_cells(
        folls['x'], folls['y'],
        folls.loc[folls['is_targeted'], ['x', 'y']].iloc[0],
        batch.reg.loc[sim_gid, 'side_um']
    ).add_suffix('_centered')

    folls = pd.concat([folls.drop(['x', 'y'], axis=1), centered], axis=1)

    spikes = batch.stores['spikes'][sim_gid]
    sb.CAT.add_cats_spikes(spikes)

    spikes = spikes[spikes['gid'].isin(folls.index)]
    delays = spikes[spikes['cat'] == 'effect'].sort_values(['gid', 'delay_in_window'])

    first_spikes = delays.groupby(['gid', 'win_idx'])['delay_in_window'].min()

    timing = pd.DataFrame({
        'mean_delay': delays.groupby('gid')['delay_in_window'].mean(),
        'median_delay': delays.groupby('gid')['delay_in_window'].median(),
        'mean_first_delay': first_spikes.groupby('gid').mean(),
        'median_first_delay': first_spikes.groupby('gid').median(),
    })

    folls = pd.concat([folls, timing], axis=1)

    return folls


def extract_all_foll_cells(working_filename, output_path):
    batch_full = sb.SimBatch.load(working_filename)

    for sbatch in batch_full.sel(instance_label='original').iter_missing_chunks('foll_cells_path', chunk_size=16):

        saved = {}

        for sim_gid in sbatch.reg.index:
            folls = _extract_sim_foll_cells(sbatch, sim_gid)
            sb.CAT.remove_cats_cells(folls)

            saved[sim_gid] = (str(output_path), f's{sim_gid:06d}')

            folls.to_hdf(*saved[sim_gid])

        name = 'foll_cells'
        saved = pd.DataFrame.from_dict(saved, orient='index', columns=[f'{name}_path', f'{name}_idx'])
        batch_full = batch_full.add_cols(saved)
        batch_full.save_registry(working_filename)


def extract_net_foll_conns(working_filename, output_path, instance_label='original'):
    """
    This can take a lot of time (~5 hours for 300 nets).
    """
    batch_full = sb.SimBatch.load(working_filename)

    if 'net_foll_conns_path' not in batch_full.reg.columns:
        batch_full.reg['net_foll_conns_path'] = np.nan

    if 'net_foll_conns_idx' not in batch_full.reg.columns:
        batch_full.reg['net_foll_conns_idx'] = np.nan

    grouped = batch_full.sel(instance_label=instance_label).reg.groupby([
        'instance_path', 'instance_idx', 'targeted_gid']).groups.items()

    to_extract = []

    for ((instance_path, instance_idx, targeted_gid), sim_gids) in grouped:
        if batch_full.reg.loc[sim_gids, 'net_foll_conns_path'].isna().any():
            if batch_full.reg.loc[sim_gids, 'net_foll_conns_path'].notna().any():
                logging.warning('Overriding existing net folls for sims')
                batch_full.reg.loc[sim_gids, 'net_foll_conns_idx'] = np.nan
                batch_full.reg.loc[sim_gids, 'net_foll_conns_path'] = np.nan

            to_extract.append((targeted_gid, sim_gids))

    print(f'{len(to_extract)} nets missing')

    def get_new_key():
        existing = [int(k[1:]) for k in batch_full.reg['net_foll_conns_idx'].dropna().unique()]

        if len(existing) > 0:
            index = max(existing) + 1
        else:
            index = 0

        return f'c{index:06g}'

    for targeted_gid, sim_gids in pbar(to_extract):

        important_cell_gids = np.unique(np.append(
            targeted_gid,
            np.concatenate(batch_full.reg.loc[sim_gids, ['e_foll_gids', 'i_foll_gids']].values.ravel())
        ))

        all_conns = batch_full.stores['conns_raw'][sim_gids[0]]

        mask = (
            all_conns['source'].isin(important_cell_gids)
            & all_conns['target'].isin(important_cell_gids)
        )

        net_conns = all_conns[mask].copy()
        sb.CAT.remove_cats_conns(net_conns)

        output_key = get_new_key()

        net_conns.to_hdf(output_path, output_key, format='fixed')

        partial = {}
        for sim_gid in sim_gids:
            partial[sim_gid] = str(output_path), str(output_key)

        partial = pd.DataFrame.from_dict(partial, orient='index', columns=['net_foll_conns_path', 'net_foll_conns_idx'])
        batch_full = batch_full.add_cols(partial)

        batch_full.save_registry(working_filename)
        batch_full.stores['conns_raw'].empty_cache()

        gc.collect()


def extract_str_foll_conns(working_filename, output_path, strength_thresh=15, mask_col=None):
    batch_full = sb.SimBatch.load(working_filename)

    name = 'str_foll_conn'

    for col in f'{name}_path', f'{name}_idx':
        if col not in batch_full.reg.columns:
            batch_full.reg[col] = np.nan

    if mask_col is None:
        sel_gids = batch_full.reg.index
    else:
        sel_gids = batch_full.subsection(batch_full.reg[mask_col]).reg.index

    missing = batch_full.subsection(sel_gids).get_missing(f'{name}_path')

    instance_key_cols = ['instance_path', 'instance_idx']

    for instance_key, sims in pbar(batch_full.reg.loc[missing].groupby(instance_key_cols), desc='net'):
        for sim_gid in pbar(sims.index):
            all_conns = batch_full.stores['conns_raw'][sims.index[0]]

            targeted_gid = sims.loc[sim_gid, 'targeted_gid']
            important_cell_gids = np.unique(
                (targeted_gid,)
                + sims.loc[sim_gid, 'e_foll_gids']
                + sims.loc[sim_gid, 'i_foll_gids']
            )

            mask = all_conns['source'].isin(important_cell_gids) & all_conns['target'].isin(important_cell_gids)
            mask &= (all_conns['weight'] >= strength_thresh)

            str_foll_conns = all_conns[mask].copy()
            sb.CAT.remove_cats_conns(str_foll_conns)

            output_key = f's{sim_gid:06d}'

            str_foll_conns.to_hdf(output_path, output_key, format='fixed')

            batch_full.reg.loc[sim_gid, f'{name}_path'] = output_path
            batch_full.reg.loc[sim_gid, f'{name}_idx'] = output_key
            batch_full.save_registry(working_filename)

        batch_full.stores['conns_raw'].empty_cache()
        gc.collect()


def extract_extents(working_filename):
    batch_full = sb.SimBatch.load(working_filename)

    for ei_type in 'ei':

        batch_sel = batch_full.subsection(batch_full.reg[f'{ei_type}_foll_count'] > 0).sel(instance_label='original')

        for sbatch in batch_sel.iter_missing_chunks(f'{ei_type}_furthest_follower_distance', chunk_size=16):

            for sim_gid in sbatch.reg.index:
                extent = ext.extract_extent(batch_full, sim_gid)

                for k, v in extent.items():
                    batch_full.reg.loc[sim_gid, k] = v

            batch_full.save_registry(working_filename)


def extract_all_jumps(working_filename, output_path, mask_col='uniform_sampling_sub'):

    batch_full = sb.SimBatch.load(working_filename)

    name = 'jumps'
    cols = [f'{name}_path', f'{name}_idx']

    if mask_col is None:
        sel_gids = batch_full.reg.index
    else:
        sel_gids = batch_full.subsection(batch_full.reg[mask_col]).reg.index

    missing = batch_full.subsection(sel_gids).get_missing(cols[0])

    grouped = batch_full.reg.loc[missing].groupby(['instance_path', 'instance_idx', 'targeted_gid']).groups

    for _, sim_gids in pbar(grouped.items(), desc='instance'):

        batch_full.stores['conns_raw'].empty_cache()
        gc.collect()

        saved = {}

        conns = batch_full.stores['conns_raw'][sim_gids[0]]
        sb.CAT.add_cats_conns(conns)

        for sim_gid in pbar(sim_gids, desc='sims'):
            spikes = batch_full.stores['spikes'][sim_gid]
            sb.CAT.add_cats_spikes(spikes)

            jumps = tc.extract_spike_jumps(spikes, conns)

            sb.CAT.remove_cats_conns(jumps)

            saved[sim_gid] = (str(output_path), f's{sim_gid:06d}')

            jumps.to_hdf(*saved[sim_gid], format='fixed')

        saved = pd.DataFrame.from_dict(saved, orient='index', columns=cols)
        batch_full = batch_full.add_cols(saved)
        batch_full.save_registry(working_filename)


def extract_motifs(working_filename, output_path, mask_col='uniform_sampling'):

    output_path = str(sb.abs_path(output_path))
    working_filename = str(sb.abs_path(working_filename))

    batch_full = sb.SimBatch.load(working_filename)

    if mask_col is None:
        batch_sel = batch_full
    else:
        batch_sel = batch_full.subsection(batch_full.reg[mask_col])

    batch_sel = batch_sel.subsection(batch_sel.stores['jumps'].keys())

    for sbatch in batch_sel.iter_missing_chunks('motif_fan_path', chunk_size=16):

        entries = {}

        for sim_gid in pbar(sbatch.reg.index):

            entries[sim_gid] = {}

            jumps = batch_full.stores['jumps'][sim_gid]

            mots = conn_check.extract_motifs_quadruplets_in_df(jumps)

            for motif_name, vs in mots.items():

                key = f's{sim_gid:06d}_{motif_name}'

                entries[sim_gid][f'motif_{motif_name}_path'] = output_path
                entries[sim_gid][f'motif_{motif_name}_idx'] = key

                vs.to_hdf(output_path, key, format='fixed')

        entries = pd.DataFrame.from_dict(entries, orient='index').rename_axis(index='sim_gid')

        batch_full = batch_full.add_cols(entries)

        batch_full.save_registry(working_filename)


def extract_order_entropy(
        working_filename, output_path,
        rank_method='first',
        mask_col=None,
        adjust_missing=True,
        min_participation=.25,
        strict_trial_count=True,
        ei_type='all',
        chunk_size=16,
):

    name = f'order_entropy_{ei_type}'

    if not adjust_missing:
        name += '_noadj'

    if rank_method != 'first':
        name += f'_{rank_method}_rank'

    if min_participation is not None:
        name += f'_{min_participation:.2f}_part'

    if strict_trial_count:
        name += '_strict'

    output_path = str(sb.abs_path(output_path))

    batch_full = sb.SimBatch.load(working_filename)

    if mask_col is None:
        batch_sel = batch_full
    else:
        batch_sel = batch_full.subsection(batch_full.reg[mask_col])

    chunks = batch_sel.iter_missing_chunks(f'{name}_path', chunk_size=chunk_size)
    for sbatch in chunks:
        dfs = oe.extract_batch(
            batch_full, sbatch.reg.index,
            min_participation=min_participation,
            rank_method=rank_method,
            ei_type=ei_type,
            adjust_missing=adjust_missing,
            strict_trial_count=strict_trial_count,
        )

        saved = {}

        for sim_gid, df in dfs.items():
            pair = (output_path, f'{ei_type}_s{sim_gid:06d}')
            saved[sim_gid] = pair
            df.to_hdf(*pair, format='fixed')

        saved = pd.DataFrame.from_dict(
            saved,
            orient='index',
            columns=[f'{name}_path', f'{name}_idx'])

        batch_full = batch_full.add_cols(saved)
        batch_full.save_registry(working_filename)


def extract_kmodes_labels(
        working_filename, output_path,
        ei_type='all',
        mask_col=None,
        chunk_size=88*3,
        target_cluster_size=6,
        exec_mode=None,
        **kmodes_kwargs,
):
    name = f'kmodes_{ei_type}_{target_cluster_size}'

    output_path = str(sb.abs_path(output_path))

    batch_full = sb.SimBatch.load(working_filename)

    if mask_col is None:
        batch_sel = batch_full
    else:
        batch_sel = batch_full.subsection(batch_full.reg[mask_col])

    # sort by foll count because those sims with biggest numbers take very long to process
    foll_counts = batch_sel.reg['e_foll_gids'].map(len) + batch_sel.reg['i_foll_gids'].map(len)
    foll_counts.sort_values(inplace=True)
    print(foll_counts)
    batch_sel = sb.SimBatch(batch_sel.reg.reindex(foll_counts.index).copy())

    chunks = batch_sel.iter_missing_chunks(f'{name}_path', chunk_size=chunk_size)
    for sbatch in chunks:
        print(foll_counts.loc[sbatch.reg.index])

        dfs = branches.extract_batch(
            batch_full, sbatch.reg.index,
            target_cluster_size=target_cluster_size,
            ei_type=ei_type,
            exec_mode=exec_mode,
            **kmodes_kwargs,
        )

        saved = {}

        for sim_gid, df in dfs.items():
            pair = (output_path, f'{ei_type}_{target_cluster_size}_s{sim_gid:06d}')
            saved[sim_gid] = pair
            df.to_hdf(*pair, format='fixed')

        saved = pd.DataFrame.from_dict(
            saved,
            orient='index',
            columns=[f'{name}_path', f'{name}_idx'])

        batch_full = batch_full.add_cols(saved)
        batch_full.save_registry(working_filename)


def _process_missing_store_chunk(
        working_filename, output_path, store_name,
        process_chunk,
        mask_col='uniform_sampling',
        req_cols=None, chunk_size=16,
        key_fmt='s{sim_gid:06d}',
):
    """
    Process simulations in groups saving, for each simulation, a data frame.
    """
    output_path = str(sb.abs_path(output_path))

    batch_full = sb.SimBatch.load(working_filename)

    path_col, idx_col = sb.get_col_names(store_name)

    if path_col not in batch_full.reg.columns:
        batch_full.reg[path_col] = np.nan

    if idx_col not in batch_full.reg.columns:
        batch_full.reg[idx_col] = np.nan

    if mask_col is None:
        batch_sel = batch_full
    else:
        batch_sel = batch_full.subsection(batch_full.reg[mask_col])

    if req_cols is not None:
        req_cols = pd.Index(req_cols)

        missing_cols = req_cols.difference(batch_sel.reg.columns)
        if len(missing_cols) > 0:
            logging.warning(f'Required columns do not exist: {list(missing_cols)}')
            batch_sel = batch_sel.subsection([])

        else:
            batch_sel = batch_sel.subsection(batch_sel.reg[req_cols].notna().all(axis=1))

    chunks = batch_sel.iter_missing_chunks(path_col, chunk_size=chunk_size)

    for sbatch in chunks:
        dfs = process_chunk(batch_full, sbatch.reg.index)

        entries = {}

        for sim_gid, df in dfs.items():
            output_key = key_fmt.format(sim_gid=sim_gid)

            entries[sim_gid] = {
                path_col: output_path,
                idx_col: output_key,
            }
            df.to_hdf(output_path, output_key, format='fixed')

        entries = pd.DataFrame.from_dict(entries, orient='index')
        batch_full = batch_full.add_cols(entries)
        batch_full.save_registry(working_filename)


def process_missing_store_single(
        working_filename, output_path, store_name,
        process_one,
        mask_col='uniform_sampling', req_cols=None, chunk_size=16,
):
    """
    Process simulations in series saving, for each one, a data frame.
    """
    def _process_chunk_singles(batch_full, sim_gids):
        return {
            sim_gid: process_one(batch_full, sim_gid)
            for sim_gid in sim_gids
        }

    _process_missing_store_chunk(
        working_filename, output_path, store_name,
        process_chunk=_process_chunk_singles,
        mask_col=mask_col,
        req_cols=req_cols,
        chunk_size=chunk_size,
    )


def extract_foll_amats(
        working_filename, output_path,
        gid_cols=('e_foll_gids', 'i_foll_gids'),
        mask_col=None,
):
    """
    Extract the activation matrix of follower cells in each simulation
    """
    def _extract_foll_amat(batch: sb.SimBatch, sim_gid: int) -> pd.DataFrame:
        # stored as tuples, so sum concatenates
        foll_gids = batch.reg.loc[sim_gid, list(gid_cols)].sum()

        spikes = batch.stores['spikes_raw'][sim_gid]
        trial_times = batch.reg.loc[sim_gid, 'forced_times']

        return branches.extract_cell_amat(foll_gids, spikes, trial_times)

    process_missing_store_single(
        working_filename,
        output_path,
        store_name='amat',
        process_one=_extract_foll_amat,
        mask_col=mask_col,
        req_cols=['spikes_raw_path'] + list(gid_cols),
        chunk_size=16,
    )


def extract_ref_foll_amats(
        working_filename,
        output_path,
        batch_ref: sb.SimBatch,
        gid_cols=('e_foll_gids', 'i_foll_gids'),
        mask_col=None,
):
    """
    Extract the activation matrix of follower cells in each simulation.
    Followers are determined in a reference simulation.
    """
    def _extract_foll_amat(batch: sb.SimBatch, sim_gid: int) -> pd.DataFrame:
        # stored as tuples, so sum concatenates
        ref_sim = batch.reg.loc[sim_gid, 'ref_sim']
        foll_gids = batch_ref.reg.loc[ref_sim, list(gid_cols)].sum()

        spikes = batch.stores['spikes_raw'][sim_gid]
        trial_times = batch.reg.loc[sim_gid, 'forced_times']

        return branches.extract_cell_amat(foll_gids, spikes, trial_times)

    process_missing_store_single(
        working_filename,
        output_path,
        store_name='amat_ref',
        process_one=_extract_foll_amat,
        mask_col=mask_col,
        req_cols=['ref_sim', 'spikes_raw_path'],
        chunk_size=16,
    )
