"""
Shared code to run a sequence-like experiment where cells are driven by random background activity
and one or more cells are forced to spike at fixed times.
"""
import logging
import pandas as pd
import numpy as np
from tqdm.auto import tqdm as pbar

from tctx.util.profiling import log_time
from tctx.util import sim, track, sim_nest
from tctx.networks.base import NetworkInstance

from tctx.analysis import simbatch as sb

import socket


def load_instance(instance_path, instance_idx):
    """return a tctx network instance"""
    with log_time('load instance', pre=True):
        logging.info(f'instance "{instance_idx}" in\n{instance_path}')
        instance = sim.load_instance(instance_path, instance_idx)

        sb.CAT.add_cats_cells(instance.cells)
        sb.CAT.add_cats_conns(instance.connections)

        instance = NetworkInstance(
            instance.params, instance.cells, instance.connections, instance.extra)

    return instance


def run_experiment_single(instance: NetworkInstance, sim_params: pd.Series):
    """
    run a sequences experiment with:

        - Targeted cells are FORCED to spike by stopping the sim and setting their voltage.
          (as opposed to sending them a single spike with strong weight).

        - The single EXACT SAME network. Any parameters changing should only be about the drive.
    """
    sim_nest.reset_kernel()
    instance.implement()

    sim_nest.build_input(dict(sim_params.to_dict()), instance.cells)

    simlen = sim.SimLength(**sim_params[['tstart', 'tend', 'dt', 'dt_inv']])

    simsetup = sim_nest.build_simsetup(
        simlen, instance,
        spike_detect='all',
        voltage_measure=sim_params.get('voltage_measure', None),
        voltage_measure_freq=sim_params.get('voltage_measure_freq', 1),
    )

    force_sets = sim.build_forced_sets(sim_params)

    simres = sim_nest.run_multistop(simsetup, instance, force_sets)

    return simres


def verify_params_compatible(instance: NetworkInstance, sim_params: pd.Series):
    """check params between instance and simulation seem compatible"""
    instance_params = pd.Series(instance.params)

    # noinspection PyUnresolvedReferences
    shared = sim_params.index.intersection(instance_params.index)

    if not shared.empty:
        # noinspection PyTypeChecker
        different: pd.Series = sim_params.loc[shared] != instance_params.loc[shared]

        # noinspection PyUnresolvedReferences
        different = different.index[different.values]

        if np.any(different):

            different = pd.concat([
                sim_params.loc[different],
                instance_params.loc[different]],
                axis=1, keys=['instance', 'sim'])

            logging.warning(
                f'Different params between instance and simulation:\n{different}')


def select_by_hostname(batch):
    """Optionally, take a subsection of a batch based on the current hostname,
    so we can run on parallel on multiple nodes from a single input file"""

    if 'hostname' in batch.reg.columns:
        this_hostname = socket.gethostname()
        subbatch = batch.sel(hostname=this_hostname)
        logging.info(f'selecting {len(subbatch.reg)}/{len(batch.reg)} sims for hostname {this_hostname}')
        batch = subbatch
    else:
        logging.debug('no hostname found')

    return batch


def select_by_status(batch):
    """Optionally, take a subsection of a batch based that are marked as "pending" and ignore those as "success",
    so we can easily re-launch simulations that were cancelled"""

    if 'status' in batch.reg.columns:
        subbatch = batch.subsection(batch.reg['status'].fillna('pending') == 'pending')
        logging.info(f'selecting {len(subbatch.reg)}/{len(batch.reg)} sims with status "pending"')
        batch = subbatch
    else:
        logging.debug('no status found')

    return batch


def run_experiment_multiple(batch, results_path, batch_path, batch_idx='sweeps'):
    """
    Run multiple experiments sequentially

    :param batch_path: single file full path to save the resulting batch
    :param batch_idx: key of the saved batch
    :param results_path: single file full path to save the resulting spikes
    :param batch: a brand new batch with the table of simulation parameters (rows=one sim, cols=params).
    It must specify the instance to load.
    :return:
    """
    batch = select_by_hostname(batch)
    batch = select_by_status(batch)
    batch.reg['status'] = 'pending'

    logging.info(f'saving registry "{batch_idx}" in\n{batch_path}')
    batch.save_registry(batch_path, batch_idx)

    # loading an instance takes time but we will likely use the same one multiple times
    # If sweeps had been sorted by instance, so we can load the instance only when it changes
    # This assumes only sim_params that change form instance to instance are input related
    # sweeps = sweeps.sort_values(['instance_path', 'instance_idx'])
    instance = None
    cached_instance_key = (None, None)

    try:
        for i, sim_idx in enumerate(pbar(batch.reg.index, desc='sweep')):
            with log_time(f'sweep {i+1}/{len(batch.reg)}', pre=True, level=logging.INFO):
                sim_params = batch.reg.loc[sim_idx]

                instance_key = tuple(sim_params[['instance_path', 'instance_idx']].values)
                if instance_key != cached_instance_key:
                    instance = load_instance(*instance_key)
                    cached_instance_key = instance_key

                verify_params_compatible(instance, sim_params)
                simres = run_experiment_single(instance, sim_params)

                logging.info(f'saving results {results_path} {sim_idx}')
                simres.spikes.to_hdf(results_path, sim.get_spikes_key(sim_idx))

                if simres.voltages is not None:
                    simres.voltages.to_hdf(results_path, sim.get_voltages_key(sim_idx))

                batch.reg.loc[sim_idx, 'full_path'] = results_path
                batch.reg.loc[sim_idx, 'sim_idx'] = sim_idx
                batch.reg.loc[sim_idx, 'status'] = 'success'

                logging.info(f'updating registry {batch_path} {batch_idx}')
                batch.save_registry(batch_path, key=batch_idx)

    except KeyboardInterrupt:
        logging.info('KEYBOARD INTERRUPT: EARLY EXIT')
        logging.info(f'updating registry {batch_path} {batch_idx}')
        batch.save_registry(batch_path, key=batch_idx)


################################################################################################
# script mode

def main_script():

    parser = track.get_argparser()

    parser.add_argument('path', help='path to HDF5 containing input batch', type=str)
    parser.add_argument('--input-idx', default='sweeps', required=False, help='index of input batch in HDF5', type=str)

    config = parser.parse_args()

    input_sweeps_path = config.path
    input_sweeps_idx = config.input_idx

    with track.new_experiment(config=config, prefix_stack=False, timestamp=True, categories='seqexp') as et:
        with log_time(f'full multirun', level=logging.INFO):
            print(f'Loading input reg:\n{sb.abs_path(input_sweeps_path)}')
            batch = sb.SimBatch.load(input_sweeps_path, input_sweeps_idx)

            run_experiment_multiple(
                batch.copy_clean_reg(drop_cluster_params=False),
                et.get_data_path('sim_results.h5'),
                et.get_data_path('sweeps.h5')
            )


if __name__ == '__main__':
    main_script()
