"""
    Some classes to pre-process big connectivity data and speed up the processing.
"""

import numpy as np
import pandas as pd
import numba
import collections


class PresortedGroupBy:
    """using a connections dataframe directly is very costly if you want to do lots of groupby operations
    specially for big-ish networks (>100K cells, >50M connections)

    instead, keep a sorted array of targets/sources so we can do a binary search, which is much faster
    """

    @staticmethod
    def target_to_source(all_connections):
        return PresortedGroupBy(all_connections, by='target')

    @staticmethod
    def source_to_target(all_connections):
        return PresortedGroupBy(all_connections, by='source')

    def __init__(self, all_connections, by='target'):
        """
        note this will be doing a sort (which can take some time) and make some data copies (which take memory)

        :param all_connections: a DataFrame with all connections
        :param by: name of the column to group by
        """
        self.sorted_connections = all_connections.sort_values(by)
        self.by = by
        self.by_raw = self.sorted_connections[by].values
        self._unique_values, self._unique_counts = np.unique(self.by_raw, return_counts=True)

    @property
    def total(self):
        return len(self._unique_values)

    def __len__(self):
        return self.total

    def get(self, name, value):
        """
        :param name: name of the column to get a value from.
            Any column in the original df is valid

        :param value: value whose group you want to get
            (eg. a particular cell index whose target/sources you want to get)

        :return:
        """
        vrange = self._find_range(value)
        return self.sorted_connections[name].iloc[vrange]

    def _find_range(self, cell_idx):
        return slice(
            np.searchsorted(self.by_raw, cell_idx, side='left'),
            np.searchsorted(self.by_raw, cell_idx, side='right')
        )


class ConnectionTester:
    """
    A cache-based way to quickly check if two cells are connected or not
    """
    @classmethod
    def from_conns(cls, all_gids, all_conns: pd.DataFrame):
        return cls(
            max_gid=np.max(all_gids),
            sources=all_conns['source'].values,
            targets=all_conns['target'].values,
        )

    def __init__(self, max_gid, sources, targets):
        """
        :param max_gid: max node gid that exist, required for hashing
        :param sources: list of sources
        :param targets: list of targets
        """
        self._hash_type = np.uint

        self._maxgid = max_gid
        self._base = np.power(10, np.ceil(np.log10(self._maxgid))).astype(self._hash_type)

        self._maxhash = self._hash_pairs(self._maxgid, self._maxgid)
        self._all_conn_hashes = self._hash_pairs(sources, targets)

        # we add an extra one because "searchsorted" may return a position beyond length of connections
        # (for a pair of very high gids that do NOT share a connection)
        # the fake hash that we generate here is impossible because no cells exist with these gids
        impossible_hash = (self._maxhash + 1).astype(self._hash_type)
        self._all_conn_hashes = np.concatenate([self._all_conn_hashes, [impossible_hash]])
        self._all_conn_hashes = np.sort(self._all_conn_hashes)

    def _hash_pairs(self, gids0, gids1):
        return (gids0 * self._base + gids1).astype(self._hash_type)

    def check(self, pre_gids, post_gids):
        hashed = self._hash_pairs(pre_gids, post_gids)
        return _searchsorted_parallel(self._all_conn_hashes, hashed)


@numba.njit(parallel=True)
def _searchsorted_parallel(_all_conn_hashes, hashed):
    found = np.empty(len(hashed), dtype=numba.boolean)

    for i in numba.prange(len(hashed)):

        idx = np.searchsorted(_all_conn_hashes, hashed[i], side='left')
        found[i] = hashed[i] == _all_conn_hashes[idx]

    return found


_Links = collections.namedtuple(
    'Links',  # numba-friendly version of a graph
    [
        'idcs',  # np.ndarray, int, unique index of the jump
        'sources',  # np.ndarray, int, unique identifier for the source of each jump
        'targets',  # np.ndarray, int, unique identifier for the target of each jump
    ])


@numba.njit
def _extract_motifs_quadruplets_in_jit(original: _Links):
    """
    This will classify incoming connections per spike taking into account 2 levels back: parents and grand parents.
    Note that this classification is only valid for the "2me" connections even though all are given per motif.
    For example: a "pop2dad" connection may in fact be part of a fan-in for dad, but we don't check that far back.
    :return:
    """
    fans = []
    diamonds = []
    singles = set()

    # sort so that targets are in contiguous space and we can use searchsorted
    sorting = np.argsort(original.targets)
    links = _Links(
        original.idcs[sorting],
        original.sources[sorting],
        original.targets[sorting])

    def slice_links(lnks, sl):
        return _Links(
            lnks.idcs[sl],
            lnks.sources[sl],
            lnks.targets[sl],
        )

    def find_sources(who):
        sl = slice(
            np.searchsorted(links.targets, who, side='left'),
            np.searchsorted(links.targets, who, side='right'))

        return slice_links(links, sl)

    i_me = 0

    total_count = len(links.targets)

    while i_me < len(links.idcs):

        me = links.targets[i_me]
        count = 0
        while (i_me + count) < total_count and links.targets[i_me + count] == me:
            count += 1

        all_dads2me = slice_links(links, slice(i_me, i_me + count))
        assert np.all(all_dads2me.targets == all_dads2me.targets[0])

        classified = set()

        for i_dad0 in range(len(all_dads2me.idcs)):
            all_pops2dad = find_sources(all_dads2me.sources[i_dad0])

            dad2me_idx = all_dads2me.idcs[i_dad0]

            for i_pop in range(len(all_pops2dad.idcs)):
                for i_dad1 in range(len(all_dads2me.idcs)):
                    if all_pops2dad.sources[i_pop] == all_dads2me.sources[i_dad1]:

                        pop2dad_idx = all_pops2dad.idcs[i_pop]
                        pop2me_idx = all_dads2me.idcs[i_dad1]

                        fans.append((pop2me_idx, dad2me_idx, pop2dad_idx))

                        classified.add(dad2me_idx)
                        classified.add(pop2me_idx)

            for i_mum in range(i_dad0 + 1, len(all_dads2me.idcs)):
                assert all_dads2me.idcs[i_dad0] != all_dads2me.idcs[i_mum]

                all_pops2mum = find_sources(all_dads2me.sources[i_mum])

                for i_paternal_pop in range(len(all_pops2dad.idcs)):
                    for i_maternal_pop in range(len(all_pops2mum.idcs)):

                        if all_pops2dad.sources[i_paternal_pop] == all_pops2mum.sources[i_maternal_pop]:

                            pop2dad_idx = all_pops2dad.idcs[i_paternal_pop]
                            pop2mum_idx = all_pops2mum.idcs[i_maternal_pop]
                            mum2me_idx = all_dads2me.idcs[i_mum]

                            diamonds.append((
                                pop2dad_idx, dad2me_idx, pop2mum_idx, mum2me_idx,
                            ))

                            classified.add(dad2me_idx)
                            classified.add(mum2me_idx)

        singles.update(set(all_dads2me.idcs) - classified)

        i_me = i_me + count

    return fans, diamonds, singles


def extract_motifs_quadruplets_in_df(jumps: pd.DataFrame):
    """
    Wrapper for extract_motifs_quadruplets that converts back and forth a graph in DataFrame format
    for a friendly notebook experience

    :param jumps:
    :return:
    """

    graph = _Links(
        jumps.index.values.astype(np.int),
        jumps.source_spike.values.astype(np.int),
        jumps.target_spike.values.astype(np.int),
    )

    fans, diamonds, singles = _extract_motifs_quadruplets_in_jit(graph)

    return {
        'fan': pd.DataFrame.from_records(fans, columns=['pop2me_idx', 'dad2me_idx', 'pop2dad_idx']),
        'diamond': pd.DataFrame.from_records(diamonds, columns=['pop2dad_idx', 'dad2me_idx', 'pop2mum_idx', 'mum2me_idx']),
        'single':  pd.DataFrame({'dad2me_idx': list(singles)}),
    }
