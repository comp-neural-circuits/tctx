"""
Code to generate new simulations.
"""

import logging

import numpy as np
import pandas as pd
from tctx.analysis import simbatch as sb
import datetime


#########################################################################################
# Generate new simulation protocols


class ProtocolSweep:
    """
    Represents a batch of simulations we are preparings to run.
    These are either previously cancelled simulations or brand new ones.
    New simulations are usually sweeps of sim parameters or re-runs of old sims
    with slightly different params.

    The class is a wrapper around a dataframe to methods to construct sweeps from
    combinations of parameters.

    This class is very similar to SimBatch and it's meant to contain all code
    on the preparation stage. Because of this, most methods modify the registry inplace.
    """

    def __init__(self, reg):
        self.reg = reg
        assert self.reg.index.is_unique
        assert self.reg.columns.is_unique

    def _repr_html_(self):
        """pretty printing for jupyter"""
        # noinspection PyProtectedMember
        return self.reg._repr_html_()

    @classmethod
    def from_missing(cls, batch_results: sb.SimBatch):
        """prepare to run sims that failed or got cancelled"""
        missing_results = batch_results.reg['full_path'].isna()

        missing_sims = cls(
            batch_results.copy_clean_reg().reg.loc[missing_results]
        )

        return missing_sims

    def __getitem__(self, item) -> pd.Series:
        """look up a column"""
        return self.reg[item]

    def __len__(self):
        return len(self.reg)

    def sel(self, index):
        """
        select a subsection of the new protocols (inplace)

        eg: protocols.sel(protocols['input_targeted_weight'] != 0)
        """
        self.reg = self.reg.loc[index].copy()
        self.reg.reset_index(inplace=True, drop=True)

    def split(self, count):
        sections = np.array_split(self.reg.index, count)
        parts = [self.__class__(self.reg.loc[s]) for s in sections]
        return parts

    def make_param_tuple(self, col):
        """
        Make sure a given column is of tuple type.
        This is sometimes necessary for nest.
        List values will be casted, single values will be wrapped.
        Column is assumed to have homogeneous type.
        """
        values = self.reg[col].dropna()

        if len(values) > 0:
            if isinstance(values.iloc[0], tuple):
                logging.warning(f'column {col} already contains tuples')

            elif isinstance(values.iloc[0], (list, np.ndarray)):
                self.reg[col] = values.apply(tuple)

            else:
                self.reg[col] = values.apply(lambda x: (x,))

    def convert_toffset_to_explicit_times(self, new_col, trial_times, col='toffset'):
        """
        Convert a column containing a relative temporal offset to actual values.
        This adds a new column without removing the old one.
        """
        self.reg[new_col] = self.reg[col].map(lambda x: tuple(x + np.asarray(trial_times)))

    def complete_with_ref_sims(self, existing_batch, ref_sims='ref_sim'):
        """
        Generate one sim for entry in new_protocols, where all non-specified parameters are taken from a reference
        simulation in the exisiting batch
        """
        if isinstance(ref_sims, str):
            ref_sims = self.reg[ref_sims]

        elif not isinstance(ref_sims, pd.Series):
            ref_sims = pd.Series(ref_sims, index=self.reg.index)

        assert len(self.reg) == len(ref_sims)

        if self.reg.isna().any().any():
            logging.warning(
                f'Unexpected nans in new protocols: ' +
                ', '.join([
                    f'{np.count_nonzero(v.isna())} in {c}'
                    for c, v in self.reg.items()
                    if v.isna().any()
                ]))

        clean_batch = existing_batch.copy_clean_reg().reg
        completed_protocol = clean_batch.reindex(ref_sims.values)
        completed_protocol.index = self.reg.index

        for col, values in self.reg.items():
            completed_protocol[col] = values

        assert not completed_protocol['instance_path'].isna().any()
        assert not completed_protocol['instance_idx'].isna().any()
        assert completed_protocol.index.is_unique
        assert completed_protocol.columns.is_unique

        self.reg = completed_protocol

    def drop_existing(self, existing, columns=None):
        """remove entries that already exist in an existing batch"""

        exists = self.exists(existing, columns=columns)

        entries = self.reg[~exists]

        print(f'Dropped {len(self.reg) - len(entries):,g}/{len(self.reg):,g} entries')

        self.reg = entries

    def exists(self, existing, columns=None) -> pd.Series:

        common = self.reg.columns.intersection(
            existing.reg.columns
        )

        if not isinstance(existing, pd.DataFrame):
            existing = existing.reg
        assert isinstance(existing, pd.DataFrame)

        # we concat twice the existing batch, so those get dropped for sure
        diff = pd.concat([
            self.reg.loc[:, common],
            existing.loc[:, common],
            existing.loc[:, common],
        ]).drop_duplicates(keep=False, subset=columns)

        exists = pd.Series(np.ones(len(self.reg), dtype=np.bool_), index=self.reg.index)

        exists.loc[diff.index] = False

        return exists

    def assign_hostname(self, hostnames):
        """Assign a hostname to each sim.
        If multiple hostnames are given, these are assigned alternating, so that the batch order is preserved.
        This is useful to prioritise sims over others in a global manner and maintain that order when
        running in parallel.
        """
        if 'hostname' in self.reg.columns:
            logging.warning('Overwriting existing hostnames')

        if isinstance(hostnames, str):
            hostnames = [hostnames]

        assigned = pd.Series(index=self.reg.index)

        for i, hn in enumerate(hostnames):
            sim_gids = self.reg.index[i::len(hostnames)]
            assigned.loc[sim_gids] = hn

        self.reg['hostname'] = assigned

    def assign_hostname_except(self, hostnames=None):
        all_hostnames = pd.Index([
            'lnx-cm-24010',
            'lnx-cm-24009',
            'lnx-cm-24008',
            'lnx-cm-24006.mpibr.local',
            'lnx-cm-24005',
            'lnx-cm-24004',
            'lnx-cm-24003',
            'lnx-cm-24002',
            'lnx-cm-24001',
        ])

        if hostnames is None:
            hostnames = []

        if isinstance(hostnames, str):
            hostnames = [hostnames]

        self.assign_hostname(
            all_hostnames.difference(hostnames)
        )

    def to_batch(self) -> sb.SimBatch:
        """
        Create a new batch to run from this set of protocols

        note that a ProtocolSweep is just a thin wrapper around a dataframe just like
        SimBatch, so here we are just going to check the protocols look right before
        preparing a run.
        """
        assert not self.reg['instance_path'].isna().any()
        assert not self.reg['instance_idx'].isna().any()
        assert self.reg.index.is_unique
        assert self.reg.columns.is_unique

        self.reg.drop('status', axis=1, inplace=True, errors='ignore')

        return sb.SimBatch(self.reg)

    def save(self, desc, folder=None):
        """save these new simulations to prepare a new run"""

        path = sb.SimBatch.new_reg_path(desc=desc, folder=folder, filename='registry_input')
        batch = self.to_batch()
        batch.save_registry(path, quiet=True)

        # Not elegant but useful: we need to manually launch these sims everytime, so print the command
        log_level = 'info'
        rel_path = path.relative_to(sb.abs_path(''))
        self.describe_size()
        print(f'To run simulations execute:')
        print(f'nohup python tctx/seqexp.py {rel_path} -l {log_level} --redirect &> nohup_$HOSTNAME.log &')

        return rel_path

    def shuffle(self):
        """Reorder the rows randomly. Does not reset the index."""
        self.reg = self.reg.sample(frac=1, replace=False)

    def describe_size(self):
        """print a summary of the size of this set of simulations"""
        if len(self.reg) == 0:
            print(f'No simulations to run')
            return

        sims_per_hour = 2

        hostnames = pd.Series('none', index=self.reg.index)
        if 'hostname' in self.reg.columns:
            hostnames = self.reg['hostname']

        n_machines = hostnames.nunique()

        highest_load = self.reg.groupby(hostnames).size().max()
        hours = highest_load / sims_per_hour
        endtime = datetime.datetime.now() + datetime.timedelta(hours=hours)

        desc = (
            f'{len(self.reg)} sims on {n_machines} machines, '
            f'around {hours:.1f} hours ({endtime:%Y-%m-%d %A %H:%M})'
        )

        if hours > 12:
            desc += f' ({hours/24:.2f} days)'

        print(desc)


def recover_and_run(
        input_path: str,
        partial_paths: list,
        hostnames_except=None,
        save_desc=None,
        key_cols=None,
) -> sb.SimBatch:
    """
    Find what sims are missing and save them as a new small input batch.
    Returns the collected results.
    """

    results = sb.SimBatch.recover(
        input_path=input_path,
        partial_paths=partial_paths,
        key_cols=key_cols,
    )

    missing = ProtocolSweep.from_missing(results)

    if len(missing) > 0:
        missing.assign_hostname_except(hostnames_except)

        if save_desc is not None:
            missing.save(save_desc)
        else:
            missing.describe_size()

    return results
