"""
Code to manage results of many simulations together.

"""
import pandas as pd

from tctx.networks.turtle_data import DEFAULT_ACT_BINS
from tctx.util import sim

import os.path
import logging
from pathlib import Path
import json
from tqdm.auto import tqdm as pbar
import datetime
import h5py

import numpy as np
import re


BASE_FOLDER = Path('/gpfs/gjor/personal/riquelmej/dev/tctx/data/interim')


# TODO find a way to get rid of these
LIST_LIKE_COLS = [
    r'.*forced_times.*',
    r'.*input_targeted_times.*',
    r'.*input_pulsepacket_pulse_times.*',
    r'.*voltage_measure.*',
    r'foll_gids',
]


def get_col_names(name):
    """
    We store results for each simulation in indexed files.
    Every simulation will indicate its results with a path and an idx property.
    Returns the pair of column names for the given type of results.
    Eg: "spikes" -> ("spikes_path", "spikes_idx")
    """
    return f'{name}_path', f'{name}_idx'


def _get_multi_store_cols(store_names):
    """return a list of columns that represent the given store names"""
    import itertools
    return list(itertools.chain(*[get_col_names(name) for name in store_names]))


def _hdf5_table_exists(path, key) -> bool:
    """
    Check if table was saved and it's not empty
    """
    with h5py.File(path, 'r') as f:
        if key in f.keys():
            # Empty DataFrames store an axis0 and axis1 of length 1, but no data blocks.
            # This is very tied into pytables implementation, but it saves us having to load the dataframe.
            return len(f[key].keys()) > 2

        else:
            return False


class CatMangler:
    """
    because categories are nice to work with interactively but are a pain to save to HDF5
    by default we save and load data as ints
    this object makes it easy to convert between the two
    """

    def __init__(self):
        self.category_types = {
            'layer': pd.CategoricalDtype(categories=['L1', 'L2', 'L3'], ordered=False),
            'con_type': pd.CategoricalDtype(categories=['e2e', 'e2i', 'i2e', 'i2i'], ordered=False),
            'syn_type': pd.CategoricalDtype(categories=['e2x', 'i2x'], ordered=False),
            'ei_type': pd.CategoricalDtype(categories=['e', 'i'], ordered=False),
            'spike_cat': pd.CategoricalDtype(categories=['baseline', 'effect'], ordered=False),
            'foll_cat': pd.CategoricalDtype(categories=['bkg', 'foll', 'anti'], ordered=False),
            'jump_foll_cat': pd.CategoricalDtype(
                categories=['b2b', 'b2f', 'f2b', 'f2f', 'a2a', 'b2a', 'f2a', 'a2b', 'a2f'], ordered=False),
            'w_cat': pd.CategoricalDtype(categories=['weak', 'mid', 'strong'], ordered=False),
            'jump_dir': pd.CategoricalDtype(categories=['incoming', 'outgoing'], ordered=False),
            'jump_dt': pd.CategoricalDtype(categories=[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], ordered=False)
        }

        # renaming cols
        self.mappings = {
            'cells': {'ei_type': 'ei_type', 'frm_cat': 'foll_cat'},
            'connections': {'con_type': 'con_type', 'syn_type': 'syn_type'},
            'spikes': {'ei_type': 'ei_type', 'cat': 'spike_cat'},
            'jumps': {
                'con_type': 'con_type',
                'target_cat': 'spike_cat',
                'target_ei_type': 'ei_type',
                'source_cat': 'spike_cat',
                'source_ei_type': 'ei_type',
            },
            'default': {n: n for n in self.category_types.keys()},
        }

        # dropping cols
        self.cleanup = {
            'cells': ['layer', 'z'],
            'connections': ['syn_type'],
            'spikes': ['layer', 'z'],
        }

        self.bins = {
            'jump_dt': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 100],
            'w_cat': [0., 17.5, 52.5, 70.],
        }

    def get_cat_code(self, type_name, value_name):
        """
        use like: sb.CAT.get_cat_code('foll_cat', 'foll')
        """
        return self.category_types[type_name].categories.get_loc(value_name)

    def get_cat_name(self, type_name, value_code):
        """
        use like: sb.CAT.get_cat_code('foll_cat', 'foll')
        """
        return (
            self.category_types[type_name].categories[value_code]
            if np.issubdtype(type(value_code), np.number)
            else value_code)

    def _lookup_mapping(self, cat_mapping):
        if isinstance(cat_mapping, str):
            return self.mappings.get(cat_mapping, {})
        else:
            assert isinstance(cat_mapping, dict)
            return cat_mapping

    def _lookup_cleanup(self, cleanup):
        if isinstance(cleanup, str):
            return self.cleanup.get(cleanup, [])
        else:
            assert isinstance(cleanup, list)
            return cleanup

    def remove_cats(self, df, name):
        """this edits the DF inplace! """
        cat_mapping: dict = self._lookup_mapping(name)
        drop: list = self._lookup_cleanup(name)
        self._remove_cats(df, drop, cat_mapping)

    def _remove_cats(self, df, drop, cat_mapping):
        """this edits the DF inplace! """
        for c in drop:
            if c in df.columns:
                df.drop(c, axis=1, inplace=True)

        for col_name, cat_name in cat_mapping.items():
            if col_name in df.columns:
                if df.dtypes[col_name] == np.dtype('O'):
                    # string -> cat
                    df[col_name] = pd.Categorical(df[col_name], dtype=self.category_types[cat_name])

                if isinstance(df.dtypes[col_name], pd.CategoricalDtype):
                    # cat -> string
                    assert df.dtypes[col_name] == self.category_types[cat_name]
                    df[col_name] = df[col_name].cat.codes

    def add_cats(self, df, cat_mapping='default'):
        """this edits the DF inplace! """
        cat_mapping: dict = self._lookup_mapping(cat_mapping)

        for col_name, cat_name in cat_mapping.items():
            try:
                if col_name in df.columns:
                    dtype = df.dtypes[col_name]
                    if np.issubdtype(dtype, np.integer):
                        df[col_name] = pd.Categorical.from_codes(df[col_name], dtype=self.category_types[cat_name])
                    else:
                        # noinspection PyCallingNonCallable
                        df[col_name] = self.category_types[cat_name](df[col_name])

            except TypeError:
                pass

    def remove_cats_cells(self, df):
        self.remove_cats(df, 'cells')

    def remove_cats_conns(self, df):
        self.remove_cats(df, 'connections')

    def remove_cats_spikes(self, df):
        self.remove_cats(df, 'spikes')

    def add_cats_cells(self, df):
        self.add_cats(df=df, cat_mapping='cells')

    def add_cats_conns(self, df):
        self.add_cats(df=df, cat_mapping='connections')

    def add_cats_spikes(self, df):
        self.add_cats(df=df, cat_mapping='spikes')

    def add_cats_jumps(self, df):
        self.add_cats(df=df, cat_mapping='jumps')


CAT = CatMangler()


def abs_path(path: str) -> Path:
    if not os.path.isabs(path):
        path = BASE_FOLDER / path
    else:
        path = Path(path)

    return path.resolve()


def today_stamp():
    """return today's date as a string to stamp simulations"""
    return datetime.datetime.today().strftime('%Y.%m.%d')


def now_stamp():
    """return right nowe as a string to stamp simulations"""
    return datetime.datetime.now().strftime('%Y.%m.%d.%H.%M.%S.%f')


class SplitStore:
    """
    Processed data of multiple simulations stored in different paths.

    We typically process sims in incremental batches, which means we
    end up with the results spread over multiple output files.
    The registry of a SimBatch will contain a {name}_path column
    indicating the HDF5 file  for each sim and a {name}_idx indicating
    the key in the file.

    We assume all data is stored as a DataFrame per simulation.

    This class looks dict-like with pairs <sim_gid, DataFrame>.
    """

    def __init__(self, reg, path_col, idx_col):
        self.reg = reg
        self.path_col = path_col
        self.idx_col = idx_col
        self.cache = {}

        self.opened_stores = {}

    def load(self, idx=None):
        # TODO make subsection return a clean slice to drop idx arg
        # The stores returned by subsection still point to the old reg
        reg = self.reg.sort_values([self.path_col, self.idx_col])
        if idx is not None:
            reg = reg.loc[idx]

        for sim_gid in pbar(reg.index, desc='sim'):
            _ = self[sim_gid]

    def _get_locs(self):
        """get the valid pairs of path-idx"""
        return self.reg[[self.path_col, self.idx_col]].dropna()

    def keys(self):
        """return the index of sim_gid available in this store according to the reg"""
        return self._get_locs().index

    def items(self, pbar=None):
        """to iterate over the contents of this store"""
        if pbar is None:
            pbar = lambda x, desc: x

        for k in pbar(self.keys(), desc='sim'):
            yield k, self[k]

    def __len__(self):
        """return the number of valid sims in this store"""
        return len(self._get_locs())

    def __contains__(self, sim_gid) -> bool:
        """check if this store contains the given sim"""
        return sim_gid in self._get_locs().index

    def __getitem__(self, key):
        locations = self._get_locs()
        # this will raise KeyError if we are missing the data
        path, idx = locations.loc[key]
        pair = (path, idx)

        # we implement cache at the level of path/idx rather than
        # at the level of sim_gid because some sims share data
        # (like the instance)

        if pair not in self.cache:

            if path not in self.opened_stores:
                self.opened_stores[path] = pd.HDFStore(path, mode='r')

            self.cache[pair] = self.opened_stores[path][idx]

        return self.cache[pair]

    def close(self):
        self.cache = {}
        names = list(self.opened_stores.keys())
        for path in names:
            self.opened_stores[path].close()
            self.opened_stores.pop(path)

        import gc
        gc.collect()

    def empty_cache(self):
        self.cache = {}

        import gc
        gc.collect()


class StoreCollection:
    """
    Handling all stores for a SimBatch.

    This class looks dict-like with pairs <name, SplitStore>.
    """

    def __init__(self, batch):
        self._reg: pd.DataFrame = batch.reg
        self._names: list = batch.identify_stores()
        self._stores = {}

    def open(self, desc):
        assert desc not in self._stores

        path_col, idx_col = get_col_names(desc)

        if path_col not in self._reg.columns or idx_col not in self._reg.columns:
            raise KeyError(f'No columns "{path_col}" and "{idx_col}" in reg')

        self._stores[desc] = SplitStore(self._reg, path_col, idx_col)
        return self._stores[desc]

    def get(self, desc) -> SplitStore:
        if desc not in self._stores:
            return self.open(desc)

        return self._stores[desc]

    def __len__(self):
        """return the number of valid stores in this collection"""
        return len(self._names)

    def __contains__(self, desc) -> bool:
        """check if this collection contains the given store"""
        return desc in self._names

    def keys(self):
        return self._names

    def __getitem__(self, desc) -> SplitStore:
        return self.get(desc)


class SimBatch:
    """A collection of simulation results, with processed data stored as hdf5"""

    def __init__(self, reg):
        # The registry is what defines a batch
        # It contains one row per sim, one col per param
        # Some cols will come in pairs and identify where other data can be found
        # These will always be columns <X_path, X_idx>. For instance <instance_path, instance_idx>
        # and identify processed data of this simulation into an HDF5 file
        self.reg = reg.copy()

        if not self.reg.index.is_unique:
            logging.error('reg index is not unique!')

        if not self.reg.columns.is_unique:
            logging.error('reg columns is not unique!')

        self.stores = StoreCollection(self)

        # for name in 'cells_raw_path', 'spikes_raw_path':
        #     if name not in reg.columns:
        #         logging.warning(f'No {name} col in registry. Should register these!')

    def copy(self):
        """make a copy of this object (deep-copies the registry, but doesn't modify stored data)"""
        return self.__class__(self.reg)

    @classmethod
    def load(cls, reg_path, reg_idx='sweeps', patch_lists=True):
        reg_path = abs_path(reg_path)

        # noinspection PyTypeChecker
        reg: pd.DataFrame = pd.read_hdf(reg_path, reg_idx)

        if patch_lists:
            for col in reg.columns:
                for pat in LIST_LIKE_COLS:
                    if re.match(pat, col) and isinstance(reg[col].iloc[0], str):
                        reg[col] = tuple(reg[col].map(json.loads))

        return cls(reg=reg)

    @classmethod
    def load_multiple(cls, reg_path: list, only_success=True, ignore_index=False, **load_kwargs):
        """
        load multiple registries as a single batch

        The indices of all registries should not overlap.
        """

        parts = [cls.load(path, **load_kwargs).reg for path in reg_path]
        merged_reg = pd.concat(parts, axis=0, sort=True, ignore_index=ignore_index)

        if only_success:
            merged_reg = merged_reg[merged_reg['status'] == 'success'].copy()
            assert not np.any(merged_reg[['full_path', 'sim_idx']].isna())
            assert not merged_reg[['full_path', 'sim_idx']].duplicated().any()

            # for failed sims, we might get some nans, which makes this column float
            # since we are filtering for successful sims, we should be able to remain int
            merged_reg['sim_idx'] = merged_reg['sim_idx'].astype(np.int)

        assert merged_reg.index.is_unique

        return cls(merged_reg)

    @classmethod
    def recover(
            cls,
            input_path: str,
            partial_paths: list,
            key_cols=None,
    ):
        """
        Given a set of sims we wanted to run (input_path) and several sets of sims that
        have finished (partial_paths), figure out which ones are missing from the original
        input and which ones are done.

        :param input_path:
        :param partial_paths:
        :param key_cols:

        :return: A SimBatch corresponding to the filled out input batch.

        """
        batch_input = cls.load(input_path)
        batch_results = cls.load_multiple(partial_paths, ignore_index=True)

        print('results:')
        batch_results.check_sim_results()

        if key_cols is None:
            key_cols = pd.Index([
                'input_targeted_targets',
                'input_targeted_times',
                'input_targeted_weight',
                'input_whitenoise_mean',
                'input_whitenoise_std',
                'forced_times',
                'targeted_gid',
                'instance_path',
                'instance_idx',
            ])

            key_cols = key_cols.intersection(batch_input.reg.columns)

        key_cols = list(key_cols)

        final = batch_input.fill_in_missing(batch_results, key_cols=key_cols)
        assert len(batch_input) == len(final)

        missing = final.are_missing_results()
        print(f'{np.count_nonzero(missing):,g}/{len(final):,g} sims missing')

        return final

    def are_missing_results(self) -> pd.Series:
        """return boolean series indicating missing results"""
        return self.reg['full_path'].isna()

    def __str__(self):
        """short description of contents"""

        txt = f'{len(self):,g} sim' + ('s' if len(self) != 1 else '')

        counts = {
            c[:-len('_idx')]: np.count_nonzero(self.reg[c].notna())
            for c in self.identify_store_columns()
            if c.endswith('_idx')
        }

        full = [name for name, count in counts.items() if count == len(self)]
        if full:
            txt += '\nall with: ' + ', '.join(full)

        partial = [
            f'{name} ({len(self) - count} missing)'
            for name, count in counts.items() if 0 < count < len(self)
        ]

        if partial:
            txt += '\nsome with: ' + ', '.join(partial)

        return txt

    def __repr__(self):
        return str(self)

    def __len__(self):
        """return number of simulations"""
        return len(self.reg)

    def add_cols(self, df: pd.DataFrame, quiet_missing=True):
        """Add new cols to the registry. Returns copy"""

        shared_sims = self.reg.index.intersection(df.index)
        missing_sims = self.reg.index.difference(df.index)
        unknown_sims = df.index.difference(self.reg.index)

        if len(unknown_sims) > 0:
            logging.warning(f'Data for {len(unknown_sims)}  unknown sims')

        override_cols = self.reg.columns.intersection(df.columns)

        if len(override_cols) > 0:
            exisiting_data = self.reg.loc[shared_sims][override_cols]
            if np.any(exisiting_data.notna().values):
                logging.warning(f'Overriding data for {len(shared_sims)} sims')

            expected_data = self.reg.loc[missing_sims][override_cols]
            if not quiet_missing and np.any(expected_data.isna().values):
                logging.warning(f'Missing data for {len(shared_sims)} sims')

        else:
            if not quiet_missing and len(missing_sims) > 0:
                logging.warning(f'Missing data for {len(missing_sims)} sims')

        df = df.reindex(shared_sims)
        reg = self.reg.copy()

        for c, vals in df.items():
            reg.loc[shared_sims, c] = vals

        return SimBatch(reg)

    def describe_reg(self, flush=None):
        """print a text summary of the registry"""

        def short_desc(k):
            d = f'{val_counts[k]} {k}'

            if val_counts[k] <= 10:
                enum = [f'{v}' for v in np.sort(self.reg[k].unique())]
                d += ': ' + ', '.join(enum)

            elif np.issubdtype(self.reg[k].dtype, np.number):
                d += f' between {self.reg[k].min():g} and {self.reg[k].max():g}'

            return d

        def get_sweep_lengths(subreg):
            """
            for each column, return number of tested values

            :return: pd.Series
                e_input_pulsepacket_sample_targets     5
                e_proj_offset                          3
                input_whitenoise_mean                  4
                instance_idx                          60
                dtype: int64
            """

            subreg = subreg.reset_index(drop=True)

            non_object = subreg.dtypes != np.dtype('O')
            is_string = subreg.applymap(type).eq(str).all()

            return subreg.T[non_object | is_string].T.nunique()

        interesting_columns = ~self.reg.columns.str.endswith('_idx') & ~self.reg.columns.str.endswith('_path')
        val_counts = get_sweep_lengths(self.reg.loc[:, interesting_columns])
        val_counts = val_counts[val_counts > 1]

        desc = '\n'.join(['  ' + short_desc(k) for k in val_counts.keys()])

        desc = f'Found {len(self.reg)} simulations that sweep over:\n{desc}'

        def describe_expected(cols, name):
            txt = ''

            for c in cols:
                how_many = None
                if c not in self.reg.columns:
                    how_many = 'All'
                else:
                    missing = np.count_nonzero(self.reg[c].isnull())
                    if missing > 0:
                        how_many = f'{missing: >5g}/{len(self.reg)}'

                if how_many is not None:
                    txt += f'\n{how_many} sims missing {name} {c}'

            return txt

        desc += '\n' + describe_expected(['instance_path', 'instance_idx'], 'required instance column').upper()
        desc += describe_expected(['full_path', 'sim_idx'], 'expected results column')

        derivative = self.reg.columns[
            self.reg.columns.str.endswith('_idx') & (~self.reg.columns.isin(['sim_idx', 'instance_idx']))]

        if len(derivative) > 0:
            missing_count = pd.Series({c: np.count_nonzero(self.reg[c].isnull()) for c in derivative})
            missing_count.sort_values(inplace=True)
            total = self.reg.shape[0]

            desc += '\n\n' + '\n'.join([
                (f'{mc: >5g}/{total} sims missing {c}' if mc > 0 else f'all {total} sims have {c}')
                for c, mc in missing_count.items()
            ])

        bad_cols = self.reg.columns[
            self.reg.columns.str.endswith('_x') |
            self.reg.columns.str.endswith('_y')
        ]
        if len(bad_cols):
            desc += '\n\n' + 'bad merges: ' + ', '.join(bad_cols)

        if 'status' in self.reg.columns:
            desc += '\n\nstatus:' + ', '.join([
                f'{count} {stat}' for stat, count in self.reg['status'].value_counts().items()])

        print(desc, flush=flush)

    def identify_stores(self, ignored=('instance',)):
        """return a list of store names estimated from this batch's columns
        A store name is identified by the presence of *both* a X_path and a X_idx column

        :return: a list like ['cells', 'ewins', 'frm_norm_null_cmf', 'jumps', ... ]
        """
        present_cols = {}

        suffixes = ['_path', '_idx']

        for suffix in suffixes:
            cols = self.reg.columns[self.reg.columns.str.endswith(suffix)].str.slice(None, -len(suffix))
            for c in cols:
                present_cols.setdefault(c, []).append(suffix)

        present_cols = [name for name, found in present_cols.items() if len(found) == 2]

        for c in ignored:
            if c in present_cols:
                present_cols.remove(c)

        return present_cols

    def identify_store_columns(self, ignored=('instance',)):
        """
        see identify_stores

        :return: a list like ['cells_path', 'cells_idx', 'ewins_path', 'ewins_idx', ... ]
        """
        store_names = self.identify_stores(ignored=ignored)
        store_cols = _get_multi_store_cols(store_names)

        assert np.all([c in self.reg.columns for c in store_cols])

        return store_cols

    def copy_clean_reg(self, drop_cluster_params=True, drop_protocol=False, quiet=True):
        """
        Sometimes we re-run old sims with slight modifications.
        This returns a copy of our registry with only columns that are strictly new sim parameters.
        All identifiable columns relative to the results of simulations will be dropped.

        :param drop_protocol: should protocol-related columns be removed?
        :param quiet: print output on which cols are dropped

        :return: a new SimBatch with a copy of this reg with less columns
        """

        drop_cols = []

        # store columns
        drop_cols.extend(list(self.identify_store_columns()))

        if drop_cluster_params:
            drop_cols.extend(['hostname'])

        # analysis result columns
        drop_cols.extend([
            'full_path', 'sim_idx', 'status', 'date_added', 'not_forced',

            'tag',
            'low_act', 'high_act',
            'low_act_compatible', 'high_act_compatible',

            'cell_count_e', 'cell_count_i', 'cell_count_total',

            'spike_count_e', 'spike_count_i', 'spike_count_total',
            'spike_count_pre_e', 'spike_count_pre_i', 'spike_count_pre_total',
            'spike_count_post_e', 'spike_count_post_i', 'spike_count_post_total',
            'spike_count_induction_e', 'spike_count_induction_i', 'spike_count_induction_total',
            'spike_count_baseline_e', 'spike_count_baseline_i', 'spike_count_baseline_total',
            'spike_count_effect_e', 'spike_count_effect_i', 'spike_count_effect_total',

            'cell_hz_e', 'cell_hz_i', 'cell_hz_total',
            'cell_hz_pre_e', 'cell_hz_pre_i', 'cell_hz_pre_total',
            'cell_hz_post_e', 'cell_hz_post_i', 'cell_hz_post_total',
            'cell_hz_induction_e', 'cell_hz_induction_i', 'cell_hz_induction_total',
            'cell_hz_baseline_e', 'cell_hz_baseline_i', 'cell_hz_baseline_total',
            'cell_hz_effect_e', 'cell_hz_effect_i', 'cell_hz_effect_total',

            'pop_hz_e', 'pop_hz_i', 'pop_hz_total',
            'pop_hz_pre_e', 'pop_hz_pre_i', 'pop_hz_pre_total',
            'pop_hz_post_e', 'pop_hz_post_i', 'pop_hz_post_total',
            'pop_hz_induction_e', 'pop_hz_induction_i', 'pop_hz_induction_total',
            'pop_hz_baseline_e', 'pop_hz_baseline_i', 'pop_hz_baseline_total',
            'pop_hz_effect_e', 'pop_hz_effect_i', 'pop_hz_effect_total',

            'bkg_count', 'bkg_count_e', 'bkg_count_i',
            'foll_count', 'foll_count_e', 'foll_count_i',
            'foll_gids_e', 'foll_gids_i', 'foll_gids',
            'e_foll_gids', 'i_foll_gids',
            'e_anti_count', 'e_bkg_count', 'e_foll_count',
            'i_anti_count', 'i_bkg_count', 'i_foll_count',

            'furthest_follower_distance_e',
            'last_foll_activation_jitter_e', 'last_foll_activation_time_e', 'last_foll_distance_e',
            'mean_foll_activation_jitter_e', 'mean_foll_jitter_e',

            'furthest_follower_distance_i',
            'last_foll_activation_jitter_i', 'last_foll_activation_time_i', 'last_foll_distance_i',
            'mean_foll_activation_jitter_i', 'mean_foll_jitter_i',

            'e_furthest_follower_distance',
            'e_last_foll_activation_jitter', 'e_last_foll_activation_time', 'e_last_foll_distance',
            'e_mean_foll_activation_jitter', 'e_mean_foll_jitter',

            'furthest_follower_distance',
            'last_foll_activation_jitter', 'last_foll_activation_time', 'last_foll_distance',
            'mean_foll_activation_jitter', 'mean_foll_jitter',

            'i_furthest_follower_distance',
            'i_last_foll_activation_jitter', 'i_last_foll_activation_time', 'i_last_foll_distance',
            'i_mean_foll_activation_jitter', 'i_mean_foll_jitter',
        ])

        if drop_protocol:
            drop_cols.extend([
                'tstart', 'tend',
                'tstart_pre', 'tstop_pre',
                'tstart_post', 'tstop_post',
                'tstart_induction', 'tstop_induction',
                'duration_baseline', 'duration_effect',
                'trial_count', 'trial_length_ms',
                'forced_times', 'input_targeted_times',
            ])

        cols_to_remove = self.reg.columns.intersection(drop_cols)

        if not quiet:
            print(f'Removing {len(cols_to_remove)} cols: {cols_to_remove}')

        new_reg = self.reg.drop(cols_to_remove, axis=1, errors='ignore')

        return SimBatch(new_reg)

    def check_sim_results(self):
        """verify results are readable and the batch seems correct"""
        errors = (
            self._check_sim_results_unique() +
            self._check_sim_results_readable()
        )

        if len(errors) == 0:
            print('all good')

        else:
            print('\n'.join(errors))

    def _check_sim_results_unique(self) -> list:
        """
        Check that all sim results are unique.
        """
        errors = []
        count = np.count_nonzero(self.reg[['full_path', 'sim_idx']].isna().any(axis=1))
        if count > 0:
            errors.append(f'{count} simulations with missing results')

        count = np.count_nonzero(self.reg[['full_path', 'sim_idx']].duplicated())
        if count > 0:
            errors.append(f'{count} sims duplicated in batch registry')

        return errors

    def _check_sim_results_readable(self) -> list:
        """
        Try to open every results file to check that they exist and that they are good HDF5.
        If the sim is suddenly killed, the HDF5 may become corrupted (truncated).
        This has happened when the we run out of storage space and hdf5 segfaults as a consequence.
        """

        failed = []

        unique = self.reg['full_path'].unique()

        if len(unique) > 50:
            unique = pbar(unique, desc='check results files')

        for path in unique:

            try:
                with h5py.File(path, 'r'):
                    pass

            except OSError as e:
                failed.append(f'Failed to open {path} {e}')

        return failed

    @staticmethod
    def new_reg_path(desc, folder=None, filename='registry') -> Path:
        """create a new path to store this batch"""
        if folder is None:
            folder = f'batch_{today_stamp()}_{desc}'

        path = abs_path(folder) / f'{filename}_{now_stamp()}.h5'

        assert not path.exists(), f'Path {str(path)} already exists'
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save_registry(self, path, key='sweeps', quiet=False, patch_lists=False, fmt='fixed'):
        copy: pd.DataFrame = self.reg.copy()

        if patch_lists:
            for col in copy.columns:
                for pat in LIST_LIKE_COLS:
                    if re.match(pat, col):  # don't check type because some may be NaNs and others lists
                        copy[col] = copy[col].map(json.dumps)

        full_path = str(abs_path(path))

        import warnings
        import tables
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
            warnings.simplefilter(action='ignore', category=tables.PerformanceWarning)

            copy.to_hdf(full_path, key, format=fmt)

        if not quiet:
            print(f'saved reg of {len(self):,g} sims to: {full_path}')
        else:
            return path

    def _register_results(self, name, paths, idcs, expected_diff=False):
        """register results for a simulation that have been saved in an indexed"""

        path_col, idx_col = get_col_names(name)

        self.reg.loc[paths.index, path_col] = paths
        self.reg.loc[idcs.index, idx_col] = idcs

        # noinspection PyUnresolvedReferences
        assert (self.reg[path_col].isna() == self.reg[idx_col].isna()).all(), 'inconsistent path/idcs'

        if expected_diff:
            repeated = np.count_nonzero(self.reg[[path_col, idx_col]].duplicated())
            if repeated > 0:
                logging.warning(f'Repeated "{name}" results for {repeated} simulations')

    def register_raw(self):
        """
        Find all common sim results for the simulations of this batch.
        These are results that are not the result of a post-processing but rather
        the raw simulation output, and the network instance.
        """
        self.register_spikes_raw()
        self.register_voltages_raw()
        self.register_conns_raw()
        self.register_cells_raw()

    def register_cells_raw(self, name='cells_raw'):
        """
        Add the path and index of the cells to each simulation so that they can
        be loaded through a SplitStore.

        We stored cells in 'instance' files with the connectivity following
        a convention cells_{instance_idx}.
        """
        self._register_results(
            name,
            self.reg['instance_path'],
            self.reg['instance_idx'].apply(lambda x: f'cells_{x}' if isinstance(x, str) else f'cells_{x:06g}'),
            expected_diff=False,
        )

    def register_voltages_raw(self, name='voltages_raw'):
        """
        Add the path and index of the voltages to each simulation so that they can
        be loaded through a SplitStore.
        """
        voltage_idcs = self.reg['sim_idx'].map(sim.get_voltages_key)
        voltage_paths = self.reg['full_path'].copy()

        voltage_idcs[voltage_paths.isna()] = np.nan

        sim_gids = voltage_paths.dropna().index

        sim_gids = pbar(sim_gids, total=len(sim_gids), desc=f'register {name}')

        for sim_gid in sim_gids:
            path = self.reg.loc[sim_gid, 'full_path']
            idx = voltage_idcs.loc[sim_gid]
            if not _hdf5_table_exists(path, idx):
                voltage_idcs.drop(sim_gid, inplace=True)
                voltage_paths.drop(sim_gid, inplace=True)

        if len(voltage_idcs) > 0:
            self._register_results(name, voltage_paths, voltage_idcs)

        path_col, idx_col = get_col_names(name)
        if 'voltage_measure' in self.reg.columns:
            missing = np.count_nonzero(self.reg['voltage_measure'].notna() & self.reg[idx_col].isna())

            if missing > 0:
                logging.error(f'Expected voltage measurements missing for {missing} simulations')

            unexpected = np.count_nonzero(self.reg['voltage_measure'].isna() & self.reg[idx_col].notna())
            if unexpected > 0:
                logging.warning(f'Unexpected voltage measurements for {unexpected} simulations')

    def register_conns_raw(self, name='conns_raw'):
        """
        Add the path and index of the connections to each simulation so that they can
        be loaded through a SplitStore.

        We stored connections in 'instance' files with the connectivity following
        a convention connections_{instance_idx}.
        """

        def get_conns_raw_idx(x):
            if np.issubdtype(type(x), np.number):
                return f'connections_{x:06d}'
            else:
                return f'connections_{x}'

        self._register_results(
            name,
            self.reg['instance_path'],
            self.reg['instance_idx'].apply(get_conns_raw_idx),

            expected_diff=False,
        )

    def register_spikes_raw(self, name='spikes_raw'):
        """
        The raw results of a simulation (spikes, sometimes voltages) are stored in HDF
        but they are registered with a non-standard path column (full_path) and an implicit
        idx as a format that depends on the sim_gid. That format has changed over the years.
        To try to standarise accessing this data, we can build new path/idx columns, but we need
        to go through each file to check which version of the data was used.
        This needs to be done only once and then we can save the batch as is.
        """
        spikes_raw_idx = {}
        raw_loc = self.reg[['full_path', 'sim_idx']].dropna()
        raw_loc = pbar(raw_loc.iterrows(), total=len(raw_loc), desc=f'register {name}')

        for sim_id, (filename, spikes_idx) in raw_loc:
            if isinstance(spikes_idx, str):
                spikes_idx = float(spikes_idx)

            try:
                # noinspection PyProtectedMember
                spikes_raw_idx[sim_id] = sim._identify_df_key(
                    filename, [sim.get_spikes_key(spikes_idx), f'spikes_{spikes_idx:g}'])

            except KeyError:
                logging.error(f'sim #{sim_id} (sim idx {spikes_idx}) has no spikes in {filename}')

        # be consistent on the path-idx pairs for simulations where spikes are missing
        spikes_raw_idx = pd.Series(spikes_raw_idx).reindex(self.reg.index)
        paths = self.reg['full_path'].copy()
        paths.loc[spikes_raw_idx.isna()] = np.nan

        self._register_results(name, paths, spikes_raw_idx)

    def register_and_save(self, base_path, name, data_dict):
        """
        register a new data dict for this batch and save the data

        Note this may take some time/space and that it will modify some of the data inplace (CAT patching)
        This will save the data, NOT the batch registry.

        call like:

            batch.register_and_save('batch_2020.04.21_only_mid', 'cells', all_detailed_cells)
        """
        base_path = abs_path(base_path)
        base_path.mkdir(exist_ok=True, parents=True)

        full_path = str(base_path / f'{name}.h5')  # patlib objects can't be serialized

        index = list(data_dict.keys())
        paths = pd.Series([full_path] * len(data_dict), index=index)
        idcs = pd.Series([f's{sim_gid:06g}_{name}' for sim_gid in data_dict.keys()], index=index)
        self._register_results(name, paths, idcs)

        # save data to the location indicated by the registry
        # Note this calls CAT.remove_cats which will modify the DFs inplace!
        path_col, idx_col = get_col_names(name)
        for path, sreg in self.reg.groupby(path_col):
            with pd.HDFStore(path, mode='a') as store:
                for sim_gid, key in pbar(sreg[idx_col].items(), total=sreg.shape[0], desc=f'saving {name}'):
                    data = data_dict[sim_gid]
                    CAT.remove_cats(data, name)
                    store.put(key, data, format='fixed')

    def merge(self, new_sb, keys=None):
        """
        Fix a new collection of simulations so that they are compatible with an existing registry
        See match_new_old_sims

        :returns: a merge of new (first arg) and old (second arg) with the index updated so that:
            - sims also in stored_all_sim_params have the same sim_gid
            - any new sims have consecutive, non-overlapping sim_gid
            This copy is safe to store without invalidating other caches.
        """
        return SimBatch(reg=SimBatch._merge_new_old_sims(self.reg, new_sb.reg, keys=keys))

    def merge_all(self, partials):
        merged = self

        for name, part in partials.items():
            part = part.subsection(~part.are_missing_results())
            print()
            print(name)
            merged = part.merge(merged)

        return merged

    @staticmethod
    def _merge_new_old_sims(
            old_reg,
            new_reg,
            keys=None):

        if keys is None:
            keys = ('instance_path', 'instance_idx', 'full_path', 'sim_idx')

        keys = list(keys)

        mapping = SimBatch._match_new_old_sims(old_reg, new_reg, keys).reset_index()

        remapped_new = pd.merge(
            new_reg,
            mapping[['sim_gid_new', 'sim_gid']],
            left_index=True,
            right_on='sim_gid_new',
            how='left',
        )

        remapped_new = remapped_new.set_index('sim_gid').drop('sim_gid_new', axis=1)

        assert remapped_new.shape == new_reg.shape
        assert np.all(remapped_new.columns == new_reg.columns)
        assert remapped_new.index.name == 'sim_gid'

        print(len(remapped_new.index.intersection(old_reg.index)), 'shared sims')
        print(len(remapped_new.index.difference(old_reg.index)), 'new sims missing in old reg')
        print(len(old_reg.index.difference(remapped_new.index)), 'old sims missing in new reg')

        missing_old = old_reg.index.difference(remapped_new.index)

        final_reg = pd.concat([remapped_new, old_reg.loc[missing_old]], axis=0, sort=True)
        final_reg.index.name = 'sim_gid'
        final_reg = final_reg.sort_index()

        assert len(old_reg.index.difference(final_reg.index)) == 0
        assert len(remapped_new.index.difference(final_reg.index)) == 0
        assert len(final_reg.index.difference(old_reg.index).difference(remapped_new.index)) == 0

        return final_reg

    @staticmethod
    def _match_new_old_sims(
            old_reg,
            new_reg,
            keys=('instance_path', 'instance_label', 'full_path', 'sim_idx')):
        """
        After collecting simulation results, compare them with the stored registry and ensure
        the sim_gids are still matching the correct simulations

        test:
            sr.match_new_old_sims(
                old=pd.DataFrame.from_records([
                    [0] + ['a'],
                    [1] + ['b'],
                    [2] + ['c'], # exclusive
                ], columns=('sim_gid', 'data')).set_index('sim_gid'),

                new=pd.DataFrame.from_records([
                    [0] + ['a'],  # exists, same id
                    [11] + ['b'],  # exists, different id
                    [13] + ['d'], # exclusive
                ], columns=('sim_gid', 'data')).set_index('sim_gid'),
                keys=['data'],
            )

        :returns: df with a new index where:
            - sims also in stored_all_sim_params have the same sim_gid
            - any new sims have consecutive, non-overlapping sim_gid

            The df has two columns:

                         sim_gid_new  sim_gid_old
                sim_gid
                0                0.0          0.0
                1               11.0          1.0
                2                NaN          2.0
                3               13.0          NaN

            there will be nans for missing new or old ids.
        """
        keys = list(keys)

        new_reg = new_reg.sort_index(axis=1).rename_axis(index='sim_gid')
        old_reg = old_reg.sort_index(axis=1).rename_axis(index='sim_gid')

        assert new_reg.index.name == 'sim_gid'
        assert old_reg.index.name == 'sim_gid'
        # assert np.all(new_reg.columns == old_reg.columns)

        mapping = pd.merge(
            new_reg[keys].reset_index(),
            old_reg[keys].reset_index(),
            left_on=keys,
            right_on=keys,
            suffixes=['_new', '_old'],
            how='outer',
        )[['sim_gid_new', 'sim_gid_old']].sort_values(['sim_gid_old', 'sim_gid_new'])

        mapping[mapping['sim_gid_new'] != mapping['sim_gid_old']].dropna().head()
        mapping['sim_gid'] = mapping['sim_gid_old'].copy()
        missing = mapping.sim_gid_old.isna()
        mapping.loc[missing, 'sim_gid'] = mapping.sim_gid_old.max() + np.arange(1, np.count_nonzero(missing) + 1)
        mapping['sim_gid'] = mapping['sim_gid'].astype(np.int)

        assert len(mapping[mapping['sim_gid'] != mapping['sim_gid_old']].dropna()) == 0
        assert mapping['sim_gid'].is_unique

        return mapping.set_index('sim_gid')

    def subsection(self, index):
        """take a slice of the current registry"""
        return SimBatch(
            self.reg.loc[index].sort_index().copy(),
        )

    def sel(self, **kwargs):
        """
        select a subset of this registry according to multiple == comparisons

        use like:
            batch.sel(instance_label='original')
        """

        mask = np.ones(len(self.reg), dtype=np.bool_)
        for col, v in kwargs.items():
            mask = mask & (self.reg[col] == v)

        return SimBatch(self.reg.loc[mask])

    def sel_between(self, **kwargs):
        """
        select a subset of this registry according to multiple "between" comparisons

        use like:
            batch.sel_between(cell_hz_total=figs_fi_curve.DEFAULT_ACT_BINS['low_act'])

        """

        mask = np.ones(len(self.reg), dtype=np.bool_)
        for col, (vmin, vmax) in kwargs.items():
            mask = mask & (vmin <= self.reg[col]) & (self.reg[col] < vmax)

        return SimBatch(self.reg.loc[mask])

    def sel_act_level(self, which, act_level_bins=None, col='cell_hz_induction_total'):
        """
        select a subset of this registry according to compatibility with a named activity level

        use like:
            batch.sel_act_level('low_act')
        """
        alias = {
            'low_act': 'low_act',
            'low': 'low_act',
            'ex_vivo': 'low_act',

            'high_act': 'high_act',
            'high': 'high_act',
            'in_vivo': 'high_act',
        }

        if act_level_bins is None:
            act_level_bins = DEFAULT_ACT_BINS

        return self.sel_between(**{col: act_level_bins[alias[which]]})

    def fill_in_missing(
            self, other,
            key_cols=('input_targeted_mean', 'input_targeted_std', 'targeted_gid', 'instance_path', 'instance_idx'),
            copy_cols=('full_path', 'sim_idx'),
    ):
        """
        Given another batch, identify the same simulation by some columns
        and fill in the value of other columns if they are nan.

        Note this will not override existing data.

        This is useful when we want to add new results to an existing,
        partially completed, input batch.

        :param other:
        :param key_cols:
        :param copy_cols:
        :return:
        """
        key_cols = list(key_cols)
        copy_cols = list(copy_cols)

        original = self.reg.copy()

        for col in copy_cols:
            if col not in original.columns:
                original[col] = np.nan

        results = pd.merge(
            original,
            other.reg[key_cols + copy_cols],
            how='left',
            on=key_cols,
        )

        for c in copy_cols:
            results[c] = results[c + '_x'].fillna(results[c + '_y'])
            results.drop(c + '_x', axis=1, inplace=True)
            results.drop(c + '_y', axis=1, inplace=True)

        assert results.index.is_unique and results.columns.is_unique

        return self.__class__(results)

    def get_missing(self, ref_col, quiet=False) -> pd.Index:
        """
        Select all sims that are missing a value for the given column
        :return: index of sim_gids
        """

        if ref_col in self.reg.columns:
            missing = self.reg.index[self.reg[ref_col].isna()]

        else:
            missing = self.reg.index

        if not quiet:
            print(f'{len(missing):,g}/{len(self.reg.index):,g} sims missing entry')

        return missing

    def iter_missing_chunks(self, ref_col, chunk_size=512):
        """
        Iterate over all sims that are missing a value for the given column
        Iteration is in sub-batches of simulations.

        :return: iterator over SimBatch objects containing up to chunk_size sims
        """

        missing = self.get_missing(ref_col)
        chunk_count = int(np.round(len(missing) / chunk_size))

        if len(missing) > 0:
            chunk_count = max(1, chunk_count)

            for missing_chunk in pbar(np.array_split(missing, chunk_count), desc='missing sims chunk'):
                yield self.subsection(missing_chunk)
