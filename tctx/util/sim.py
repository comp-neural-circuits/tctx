"""utils to handle and analyse simulation results"""

import logging
import pandas as pd

import numpy as np
import json

import h5py

from tctx.util import networks


def _load_hdf5_df(*args, **kwargs) -> pd.DataFrame:
    """just so I can suppress pycharm's linting specifically for loading DataFrames from HDF5"""
    # noinspection PyTypeChecker
    return pd.read_hdf(*args, **kwargs)


class SimLength(object):
    """parameters defining the length of a simulation"""

    def __init__(self, tstart, tend, dt=.1, dt_inv=None):
        self.tstart = tstart
        self.tend = tend
        self.dt = dt
        self.dt_inv = dt_inv or (1. / dt)

    def to_dict(self):
        return {
            'tstart': self.tstart,
            'tend': self.tend,
            'dt': self.dt,
            'dt_inv': self.dt_inv,
        }


class SimResults(object):
    def __init__(self, spikes, voltages=None):
        self.spikes = spikes
        self.voltages = voltages

    @staticmethod
    def bad():
        return SimResults(None, None)


def pretty_print_params(all_params):
    params_copy = {
        k: v if not isinstance(v, np.generic) else np.asscalar(v)
        for k, v in all_params.items()
    }
    return json.dumps(params_copy, sort_keys=True, indent=4)


def save_instance(filename, instance, label, save_conns=True):
    if isinstance(label, str) and label.startswith('_'):
        logging.warning('dupplicated "_" in instance label')
        label = label.lstrip('_')

    cells_key = get_cells_key(label)
    instance.cells.to_hdf(filename, cells_key, format="fixed")

    save_instance_params(filename, label, instance.params)

    if save_conns:
        instance.connections.to_hdf(filename, get_conns_key(label), format="fixed")


def save_instance_params(filename, label, params):
    if isinstance(label, str) and label.startswith('_'):
        logging.warning('dupplicated "_" in instance label')
        label = label.lstrip('_')

    def safe_convert(value):
        if isinstance(value, np.int64):
            return int(value)

        if isinstance(value, np.float64):
            return float(value)

        return value

    params_str = json.dumps({k: safe_convert(v) for k, v in params.items()})

    cells_key = get_cells_key(label)

    with h5py.File(filename, 'a') as f:
        f[cells_key].attrs['params'] = params_str


def _identify_df_key(instance_path, candidates):
    """
    :candidates: list of keys, in preference order, for backwards compatibility
    """

    with h5py.File(instance_path, 'r') as f:
        file_keys = list(f.keys())

    for key in candidates:

        if key in file_keys:
            return key

    else:
        desc = ', '.join(candidates)
        raise KeyError(f'None of expected keys found ({desc} in {instance_path})')


def _get_instance_df_key(instance_idx, name):
    return (
        f'{name}_{instance_idx:06g}'
        if np.issubdtype(type(instance_idx), np.number)
        else f'{name}_{instance_idx}')


def get_cells_key(instance_idx):
    return _get_instance_df_key(instance_idx, 'cells')


def identify_cells_key(instance_path, instance_idx):
    return _identify_df_key(instance_path, [
        get_cells_key(instance_idx),
        f'cells_{instance_idx}',
        f'cells__{instance_idx}',
    ])


def get_conns_key(instance_idx):
    return _get_instance_df_key(instance_idx, 'connections')


def identify_conns_key(instance_path, instance_idx):
    return _identify_df_key(instance_path, [
        get_conns_key(instance_idx),
        f'connections_{instance_idx}',
    ])


def load_instance_params(instance_path, instance_idx):
    cells_key = identify_cells_key(instance_path, instance_idx)

    with h5py.File(instance_path, 'r') as f:
        if 'params' in f[cells_key].attrs:
            params_str = f[cells_key].attrs['params']
            return json.loads(params_str)

    return None


def load_connections(instance_path, instance_idx):
    conns: pd.DataFrame = _load_hdf5_df(instance_path, identify_conns_key(instance_path, instance_idx))
    return conns


def load_cells(instance_path, instance_idx):
    cells: pd.DataFrame = _load_hdf5_df(instance_path, identify_cells_key(instance_path, instance_idx))
    if cells.index.name is None:
        cells.index.name = 'gid'

    return cells


def load_instance(instance_path, instance_idx, load_conns=True):
    return networks.NetworkInstance(
        params=load_instance_params(instance_path, instance_idx),
        cells=load_cells(instance_path, instance_idx),
        connections=load_connections(instance_path, instance_idx) if load_conns else None,
        extra=None
    )


def get_spikes_key(sim_idx):
    return f'spikes_{sim_idx:06g}'


def get_voltages_key(sim_idx):
    return f'voltages_{sim_idx:06g}'


def build_forced_sets(sim_params: pd.Series,
                      targeted_col='targeted_gid',
                      times_col='forced_times',
                      forced_value=+10.):
    """
    look for columns named:
        targeted_col + suffix
        times_col + suffix

    and set those as forced voltage sents to induce spikes.

    Suffixes can be missing or totally different so that we can produce
    simulations with multiple cells being pinged at once.

    If the value of the gid is np.nan, it is ignored.

    :param sim_params: a pd.Series like:
    :param targeted_col:
    :param times_col:
    :param forced_value: a pd.Series like:

        targeted_gid_a         1111
        forced_times_a    [1, 2, 3]
        targeted_gid_b         2222
        forced_times_b       [1, 3]
        dtype: object

    :returns: a pd.Dataframe that can be used with sim_nest.run_multistop. Looks like:

            gid  time varname  value
        0  1111     1     V_m   10.0
        1  2222     1     V_m   10.0
        2  1111     2     V_m   10.0
        3  1111     3     V_m   10.0
        4  2222     3     V_m   10.0

    """
    forced_sets = []

    for full_name in sim_params.index[sim_params.index.str.startswith(targeted_col)]:
        gid = sim_params[full_name]

        if not np.isnan(gid):
            suffix = full_name[len(targeted_col):]

            times_col_full = f'{times_col}{suffix}'
            assert times_col_full in sim_params, f'Got {full_name} so expected {times_col_full}'
            times = sim_params[times_col_full]

            forced_sets.append(pd.DataFrame({
                'gid': int(gid),
                'time': times,
                'varname': 'V_m',
                'value': forced_value,  # (mV)
            }))

    if len(forced_sets) > 0:
        forced_sets = pd.concat(forced_sets, axis=0).sort_values(['time', 'gid']).reset_index(drop=True)
    else:
        forced_sets = pd.DataFrame(columns=['gid', 'time', 'varname', 'value'])

    return forced_sets
