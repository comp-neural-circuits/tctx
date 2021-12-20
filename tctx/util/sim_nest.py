"""code to interface with nest simulator"""

import numpy as np
import pandas as pd

import logging

from tctx.util import sim
from tctx.util.profiling import log_time
from tctx.util.defaults import DEFAULTS
from tqdm.auto import tqdm as pbar

########################################################################################################
# On import nest will initialise, print a lot of output and read the system args.
# In order to make this a bit more friendly to creating command-line executables
# for each experiment, delay nest init.
import os
os.environ['DELAY_PYNEST_INIT'] = '1'
import nest


def init(verbosity='WARNING'):
    try:
        nest.init(argv=['test.py', '--quiet', '--verbosity=' + verbosity])
    except nest.NESTError as e:
        print(f'Nest error on init: {e}')
    nest.set_verbosity('M_WARNING')


init()
########################################################################################################


def reset_kernel(dt=.1, local_num_threads=None):
    nest.ResetKernel()
    nest.SetKernelStatus({
        'grng_seed': DEFAULTS['grng_seed'],
        'resolution': dt,
        'print_time': False,
        'overwrite_files': True,
        'local_num_threads': DEFAULTS['max_workers'] if local_num_threads is None else local_num_threads
    })
    nest.set_verbosity('M_WARNING')


class SimSetup(object):

    def __init__(
            self, length,
            spike_detect=None,
            voltage_measure=None,
            voltage_measure_freq=1.,
    ):
        self.sim_length = length

        self.sp_detector = SpikeDetector(spike_detect or [])
        self.voltage_meter = Meter(voltage_measure or [], 'V_m', voltage_measure_freq)


def build_simsetup(sim_len, instance, spike_detect='all', voltage_measure=None, voltage_measure_freq=1.):
    def _node_list(param):
        if param is None or param is False:
            return []

        else:
            if isinstance(param, str) and param == 'all':
                chosen = instance.cells.nest_id.values

            elif isinstance(param, int):
                chosen = instance.cells.nest_id.sample(param)

            else:
                if isinstance(param, tuple):
                    param = list(param)

                chosen = instance.cells.loc[param, 'nest_id']
                logging.info(f'converted explicit voltage track gid list to nest_ids for {len(chosen):,g} cells')

            chosen = list(chosen)

            return chosen

    voltage_measure = _node_list(voltage_measure)
    simsetup = SimSetup(sim_len, _node_list(spike_detect), voltage_measure, voltage_measure_freq=voltage_measure_freq)

    return simsetup


def get_nest_id_to_cell_id(instance):
    return pd.Series(
        index=instance.cells['nest_id'].values,
        data=instance.cells.index.values,
    )


def run_multistop(simsetup, instance, force_sets):
    """
    Run a simulation in chunks where in each pause some cell properties are forced to a value

    This can be used to force a cell to spike externally without having to rely on external synapses or currents
    (which can unreliably produce 0 or multiple spikes)

    :param force_sets: a pd.DataFrame that looks like:

             time   gid varname  value
        0   500.0  2284     V_m   10.0
        1   900.0  2284     V_m   10.0
        2  1300.0  2284     V_m   10.0
        3  1700.0  2284     V_m   10.0
        4  2100.0  2284     V_m   10.0

    """
    nest_id_to_cell_id = get_nest_id_to_cell_id(instance)

    grouped_force_sets = force_sets.groupby('time')
    stop_points = np.array(list(grouped_force_sets.groups.keys()))

    stop_points = np.sort(np.unique([0] + list(stop_points) + [simsetup.sim_length.tend]))

    with pd.option_context('display.max_rows', 10, 'display.max_columns', 500, 'display.width', 500):
        logging.info('Running sim in %d chunks to force values:\n%s', len(stop_points) + 1, force_sets)

    with log_time('run_multistop'):
        nest.ResetNetwork()

        try:
            logging.debug('Run\n%s', sim.pretty_print_params(instance.params))

            for i, (t0, t1) in enumerate(zip(pbar(stop_points[:-1], desc='multistop'), stop_points[1:])):

                if t0 in grouped_force_sets.groups:
                    for (varname, value), to_apply in grouped_force_sets.get_group(t0).groupby(['varname', 'value']):

                        nest_ids = [int(nid) for nid in instance.cells.loc[to_apply['gid'].values, 'nest_id'].values]

                        logging.debug('force setting %s to %s for %d cells', varname, value, len(nest_ids))
                        nest.SetStatus(nest_ids, {varname: value})

                section_length = t1 - t0
                logging.debug('\nrun section %d/%d [%.1f - %.1f] (%.1f ms)',
                              i + 1, len(stop_points) - 1, t0, t1, section_length)

                nest.Simulate(section_length)

            spikes = simsetup.sp_detector.retrieve()
            voltages = simsetup.voltage_meter.retrieve()

            voltages.columns = nest_id_to_cell_id.reindex(voltages.columns).values
            spikes.gid = nest_id_to_cell_id.reindex(spikes.gid).values

            results = sim.SimResults(spikes, voltages)

        except nest.pynestkernel.NESTError as e:
            logging.exception('NESTError: %s', e)

            results = sim.SimResults.bad()

    return results


class SpikeDetector(object):
    def __init__(self, nodes):
        self.sp = nest.Create('spike_detector')
        self.nodes = nodes
        if nodes:
            nest.Connect(nodes, self.sp, {'rule': 'all_to_all'}, {})

    def retrieve(self):
        spikes = nest.GetStatus(self.sp)[0]['events']
        df = pd.DataFrame({'gid': spikes['senders'], 'time': spikes['times']})
        return df


class Meter(object):
    def __init__(self, nodes, recordable='V_m', frequency=1.):

        if nodes is not None and nodes:
            logging.info('Recording %s for %d nodes', recordable, len(nodes))

        self.frequency = frequency
        self.recordable = recordable
        self.multimeter = None
        self.nodes = nodes

        if self.nodes:
            self.multimeter = nest.Create('multimeter', n=len(nodes), params={'interval': 1. / frequency})
            nest.SetStatus(self.multimeter, {'withtime': True, 'record_from': [recordable]})
            nest.Connect(self.multimeter, nodes, {'rule': 'one_to_one'})

    def retrieve(self):
        # TODO nest doesn't return voltage for t = 0
        voltages = pd.DataFrame()

        if self.multimeter:
            traces = nest.GetStatus(self.multimeter)
            assert all([np.all(trace['events']['times'] == traces[0]['events']['times']) for trace in traces])

            trace_per_sender = {}
            for i, trace in enumerate(traces):
                senders = np.unique(trace['events']['senders'])
                assert len(senders) == 1
                trace_per_sender[senders[0]] = trace['events'][self.recordable]

            voltages = pd.DataFrame(index=traces[0]['events']['times'], data=trace_per_sender)
            voltages.rename_axis(columns='gid', index='time', inplace=True)

        return voltages


def random_init_normal(nodes, varname, mean, spread):
    node_info = nest.GetStatus(list(nodes))
    local_nodes = [ni['global_id'] for ni in node_info if ni['local']]

    values = np.random.normal(mean, spread, size=len(local_nodes))
    for i, gid in enumerate(local_nodes):
        nest.SetStatus([gid], {varname: values[i]})


def build_input(params: dict, cells):
    with log_time('build input'):
        build_global_input_rate(params, cells)
        build_global_input_current(params, cells)
        build_targeted_input(params, cells)
        build_pulsepacket_input(params, cells)


def build_global_input_rate(params: dict, cells):
    targets = {'': list(cells.nest_id.values)}
    targets.update({name + '_': list(g.values) for name, g in cells.groupby('ei_type')['nest_id']})

    for prefix, target in targets.items():
        param_name = prefix + 'input_rate'
        if param_name in params:
            rate = params.pop(param_name)
            weight = params.get(prefix + 'input_weight', 1.)
            logging.info('making poisson_generator %f Hz with strength %f nS', rate, weight)

            # A poisson_generator generates a unique spike train *for each* of it’s targets
            pg = nest.Create('poisson_generator', params={'rate': rate})
            nest.Connect(
                pg, target,
                syn_spec={'weight': weight},
            )


def build_global_input_current(params: dict, cells):
    """
    This looks for params named

        input_whitenoise_mean:
            a whitenoise current for all cells

        e_input_whitenoise_mean:
            a whitenoise current for e cells (analogous for i)

        input_whitenoise_mean_x:
            a whitenoise current for cells defined in param "input_whitenoise_targets_x" (iterable)
            there can be multiple of these.

    :param params:
    :param cells:
    :return:
    """

    nest_ids = cells.nest_id.values
    nest_id_by_ei_type = cells.groupby('ei_type')['nest_id']

    to_build = {}

    # make a copy because we may modify the dict (pop) during iteration (exception!)
    param_names = list(params.keys())

    for param_name in param_names:
        if param_name == 'input_whitenoise_mean':
            to_build['input_whitenoise_{param}'] = nest_ids

        elif param_name == 'e_input_whitenoise_mean':
            to_build['e_input_whitenoise_{param}'] = nest_id_by_ei_type.get_group('e').values

        elif param_name == 'i_input_whitenoise_mean':
            to_build['i_input_whitenoise_{param}'] = nest_id_by_ei_type.get_group('i').values

        elif param_name.startswith('input_whitenoise_mean_'):
            group = param_name[len('input_whitenoise_mean_'):]
            fmt = 'input_whitenoise_{param}_' + group
            targets = params.pop(fmt.format(param='targets'))
            logging.info(f'Special group "{group}" of {len(targets)} cells to receive additional input')
            to_build[fmt] = list(cells.loc[list(targets), 'nest_id'])

    for fmt, targets in to_build.items():
        noise_params = {
            'mean': float(params.pop(fmt.format(param='mean'))),
            'std': float(params.pop(fmt.format(param='std'))),
            'dt': float(params.pop(fmt.format(param='dt'), .1)),
        }

        logging.info(f'making noise_generator for {len(targets)} cells:\n{noise_params}')
        ng = nest.Create('noise_generator', params=noise_params)
        nest.Connect(ng, list(targets))


def _ensure_time_resolution(times: list, decimals=1) -> list:
    """
    because NEST will crash if we ar running at .1 ms
    but we ask for spikes sent at 100.25 ms, which
    can happen when generating sweeps of parameters
    """

    times = np.asarray(times)
    rounded = np.round(times, decimals=decimals)

    if np.any(times != rounded):
        logging.warning(f'Rounded times from {times} to {rounded}')

    return list(rounded)


def build_targeted_input(params: dict, cells):
    """send a regular externally generated spike to a selected group of cells"""
    times = params.pop('input_targeted_times', None)

    if times is not None:
        times = _ensure_time_resolution(times)

        targets = params.pop('input_targeted_targets')
        if np.issubdtype(type(targets), np.number):
            targets = (targets,)

        weight = params.pop('input_targeted_weight', 100.)
        logging.debug('building targeted input: sending %d spikes to %s', len(times), targets)

        cgs = nest.Create('spike_generator', 1)
        nest.SetStatus(cgs, [{
            'spike_times': [float(t) for t in times],
            'spike_weights': [float(weight)] * len(times)
        }])

        nest.Connect(cgs, list(cells.reindex(targets).nest_id))


def build_pulsepacket_input(params: dict, cells):
    """
    send, to a whole population, a bunch of spikes that are centered around a given time.
    All targets receive INDEPENDENT spike trains (one pulsegenerator object per neuron.

    see:
        http://www.nest-simulator.org/helpindex/cc/pulsepacket_generator.html
        http://www.nest-simulator.org/py_sample/pulsepacket/
    """

    name = 'pulsepacket'
    prefixes = [
        f'input_{name}_',
        f'e_input_{name}_',
        f'i_input_{name}_',
    ]

    for prefix in prefixes:
        if prefix + 'activity' in params:

            device_params = {'activity': None, 'pulse_times': None, 'sdev': None}
            for k in device_params.keys():
                device_params[k] = params.pop(prefix + k)

            targets = select_targets_by_prefix(cells, prefix)

            sample_targets = params.pop(prefix + 'sample_targets', None)
            if sample_targets is not None:
                if isinstance(sample_targets, float) and sample_targets.is_integer():
                    sample_targets = int(sample_targets)

                if not isinstance(sample_targets, int):
                    logging.warning(f'{prefix}sample_targets={sample_targets} should be valid int')

                targets = targets.sample(int(sample_targets))

            if 'pulse_times' in device_params:
                value = device_params['pulse_times']

                if isinstance(value, (tuple, list, np.ndarray)):
                    device_params['pulse_times'] = [float(t) for t in value]

                elif np.issubdtype(type(value), np.number):
                    device_params['pulse_times'] = [float(value)]

                else:
                    logging.warning(f'pulse_times={value} should be valid float or list of floats.'
                                    f'Got {type(value)}. Attempting cast.')
                    device_params['pulse_times'] = [float(value)]

            if 'activity' in device_params:
                device_params['activity'] = int(device_params['activity'])

            if 'sdev' in device_params:
                device_params['sdev'] = float(device_params['sdev'])

            logging.info('create pulse packet generator\n%s\n%s', device_params, targets)

            devices = nest.Create('pulsepacket_generator', len(targets), device_params)

            weight = params.pop(prefix + 'weight')
            nest.Connect(
                devices, list(targets.nest_id.values),
                conn_spec={'rule': 'one_to_one'},
                syn_spec={'weight': weight})


def select_targets_by_prefix(cells, prefix):
    """given all cells and a parameter name that may have a prefix, select the cells it refers to:
    if the param name starts with 'e_' or 'i_' it will be a subpopulation, otherwise return all
    """
    for subpopulation in ['e', 'i']:
        if prefix.startswith(subpopulation + '_'):
            return cells[cells.ei_type == subpopulation]

    return cells
