"""
Code to study groups of cells that activate together
"""
from tqdm.auto import tqdm as pbar

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from tctx.analysis import simbatch as sb

from tctx.util import spike_trains as spt, plot
from kmodes.kmodes import KModes


class Amat:
    """
    An activation matrix is a matrix indicating who (rows) gets active when (columns).
    It may be:
        bool representing active/non-active
        int representing number of times active
        float [0, 1] representing a percentage of presence (for clusters of cells)
    """
    def __init__(self, table):
        self.table = table

    def _is_float(self):
        return np.all([np.issubdtype(t, np.floating) for t in self.table.dtypes])

    # noinspection PyPep8Naming
    @property
    def T(self):
        """return the transpose of this table"""
        return self.__class__(self.table.T)

    def get_contrast_trials(self, good_labels=None, bad_labels=None, req_quantile=.9) -> pd.Index:
        """
        Get the trials (columns) sorted so that we maximize the activation level of
        good_labels and minimize the one from bad_labels.
        :param good_labels: elements (row) to maximize
        :param bad_labels: elements (row) to minimize
        :param req_quantile: minimum quantile of presence to consider

        :return: a subset of all trials, sorted from worst to best
        """

        if good_labels is not None:
            good_score = self.table.loc[good_labels].astype(np.float).mean(axis=0)
        else:
            good_score = pd.Series(0., index=self.table.columns)

        if bad_labels is not None:
            bad_score = self.table.loc[bad_labels].astype(np.float).mean(axis=0)
        else:
            bad_score = pd.Series(0., index=self.table.columns)

        mask = (
                (bad_score <= np.quantile(bad_score, 1 - req_quantile)) &
                (np.quantile(good_score, req_quantile) <= good_score)
        )

        return (good_score - bad_score)[mask].sort_values().index

    def get_clusters_presence(self, cell_clusters: pd.Series):
        """
        Extract the presence matrix for every cluster in every trial.

        The presence matrix indicates, for every cluster and trial, the ratio of the cells
        that belong to that cluster that spiked at least once in that trial.

        :returns:
            a new float Amat that has shape <clusters, trials> and contains entries within [0, 1].
        """
        assert not self._is_float()
        cluster_sizes = cell_clusters.groupby(cell_clusters).size()
        table = (self.table.astype(np.int).groupby(cell_clusters).sum().T / cluster_sizes).T
        return self.__class__(table)

    def get_presence(self) -> pd.Series:
        """
        :return: the percentage of elements present at each event
        """
        assert not self._is_float()
        return self.table.astype(np.int).sum() / self.table.shape[0]

    def get_clusters_active(self, cell_clusters: pd.Series, thresh=.4):
        """
        Extract the trials in which a cluster is considered to be active.
        :returns: a new binary Amat with clusters as rows.
        """
        presence = self.get_clusters_presence(cell_clusters)
        return self.__class__(presence.table >= thresh)

    def get_counts(self) -> pd.Series:
        """Count in how many trials each item (neuron or cluster) is active"""
        assert not self._is_float()
        return self.table.astype(np.int).sum(axis=1)

    def sel_trials_min_participation(self, thresh) -> np.ndarray:
        """select trial indices with a minimum participation of followers"""
        assert 0 <= thresh <= 1
        participation = self.get_presence()
        return self.table.columns[participation >= thresh]

    def sort_by_ward_distance(self, axis='both'):
        """
        Sort trying to minimize ward distance between pairs
        see scipy.cluster.hierarchy.optimal_leaf_ordering
        """
        if axis in ('cols', 'col', 'y'):
            return self._sort_cols_by_ward_distance()

        elif axis in ('rows', 'row', 'x'):
            return self.T._sort_cols_by_ward_distance().T

        else:
            assert axis == 'both'
            return self.sort_by_ward_distance('cols').sort_by_ward_distance('rows')

    # noinspection PyUnresolvedReferences
    def _sort_cols_by_ward_distance(self):
        """see sort_by_ward_distance"""

        import scipy.cluster.hierarchy

        linkage = scipy.cluster.hierarchy.ward(self.table.values.T)
        sorted_linkage = scipy.cluster.hierarchy.optimal_leaf_ordering(linkage, self.table.T.values)

        sorted_idcs = scipy.cluster.hierarchy.leaves_list(sorted_linkage)
        sorted_idcs = self.table.columns[sorted_idcs]

        sorted_table = self.table.loc[:, sorted_idcs]
        assert sorted_table.shape == self.table.shape
        assert (sorted_table.sort_index().index == self.table.sort_index().index).all()
        assert (sorted_table.sort_index(axis=1).columns == self.table.sort_index(axis=1).columns).all()

        return self.__class__(sorted_table)

    def sort_by_sum(self, ascending=False):
        """sort the matrix by number of activations in both trials and cells"""

        table = self.table

        table = table.loc[table.sum(axis=1).sort_values(kind='stable', ascending=ascending).index, :]
        table = table.loc[:, table.sum(axis=0).sort_values(kind='stable', ascending=ascending).index]

        return self.__class__(table)

    def generate_kmodes_labels(self, target_cluster_size=5, init='Huang', n_init=200) -> pd.Series:
        """
        cluster rows (cells) by activations
        :param target_cluster_size:
            Approx. how many cells per cluster.
            We'll divide the number of cells by this number and use that as K in k-modes clustering.
        """
        cell_count = len(self.table.index)

        if cell_count <= 1:
            return pd.Series(np.zeros(len(self.table.index), dtype=np.int), index=self.table.index)

        n_clusters = np.clip(cell_count // target_cluster_size, 2, cell_count)

        # print(f'clustering k={n_clusters}, init={n_init}', flush=True)
        # print(self.table, flush=True)
        # print(self.table.drop_duplicates(), flush=True)
        km = KModes(n_clusters=n_clusters, init=init, n_init=n_init, verbose=False)
        labels = km.fit_predict(self.table.values)
        labels = pd.Series(labels, index=self.table.index)

        return labels

    def plot(self, ax=None, show_cbar=False):
        if ax is None:
            _, ax = plt.subplots(constrained_layout=True, figsize=(2, 2))

        im = ax.imshow(
            self.table.astype(np.float),
        )
        if show_cbar:
            ax.figure.colorbar(im, ax=ax)


class SpikeBundle:

    def __init__(self, spikes: pd.DataFrame):
        self.spikes = spikes.copy()
        sb.CAT.add_cats_spikes(self.spikes)

    def __len__(self):
        return len(self.spikes)

    def get_gids(self) -> np.ndarray:
        return self.spikes['gid'].unique()

    def sel_first_per_trial_and_cell(self):
        return SpikeBundle(self.spikes.loc[
                               self.spikes.groupby(['win_idx', 'gid'])['time'].idxmin().values
                           ])

    def sel(self, **query):
        mask = np.ones(len(self.spikes), dtype=np.bool_)

        for k, v in query.items():
            mask = mask & (self.spikes[k] == v)

        return SpikeBundle(self.spikes.loc[mask])

    def sel_isin(self, **query):
        mask = np.ones(len(self.spikes), dtype=np.bool_)

        for k, v in query.items():
            mask = mask & (self.spikes[k].isin(v))

        return SpikeBundle(self.spikes.loc[mask])

    def sel_between(self, **query):
        mask = np.ones(len(self.spikes), dtype=np.bool_)

        for k, (low, high) in query.items():
            mask = mask & (low <= self.spikes[k]) & (self.spikes[k] <= high)

        return SpikeBundle(self.spikes.loc[mask])

    def shuffle_within_trial(self):
        """This only modifies 'delay_from_induced', not other columns (like 'time')"""
        shuffled = self.spikes.copy()
        np.random.shuffle(shuffled.loc[:, 'delay_from_induced'].values)
        return SpikeBundle(shuffled)

    def get_trial_ranks(self, rank_method) -> pd.Series:
        grouped = self.spikes.sort_values(['win_idx', 'gid']).groupby('win_idx')
        return grouped['delay_from_induced'].rank(method=rank_method).astype(np.int)

    def get_trial_gid_spike_counts(self, gids=None, n_trials=None) -> pd.DataFrame:
        """
        :return: an integer DF indicating,
        for every trial and cell, how many spikes were generated.
        looks like:

            trial_idx  0   1   98  99
            gid
            2677        0   0   1   0
            10862       0   0   1   1
            13993       0   0   1   1
            19799       0   0   0   0
            20578       0   0   0   0

        """
        if gids is not None:
            # making a selection here can make things much faster
            spikes = self.sel_isin(gid=gids).spikes
        else:
            spikes = self.spikes

        counts = spikes.groupby(['gid', 'win_idx']).size().unstack('win_idx', fill_value=0)

        # convert win_idx to trial_idx
        counts.columns = np.floor(counts.columns / 2).astype(np.int)
        assert counts.columns.is_unique

        counts = counts.rename_axis(columns='trial_idx')

        if n_trials is not None:
            if len(counts.columns) >= 1:
                assert counts.columns.min() >= 0, counts
                assert counts.columns.max() < n_trials, counts

            counts = counts.reindex(np.arange(n_trials), fill_value=0, axis=1)

        if gids is not None:
            counts = counts.reindex(gids, fill_value=0, axis=0)

        return counts

    def get_activation_matrix(self, gids=None, n_trials=None) -> Amat:
        """
        :return: a binary DF indicating,
        for every trial and cell, if any spikes were generated.
        looks like:

            trial_idx     0      1      98     99
            gid
            2677       False  False   True  False
            10862      False  False   True   True
            13993      False  False   True   True
            19799      False  False  False  False
            20578      False  False  False  False

        """
        # noinspection PyTypeChecker
        return Amat(self.get_trial_gid_spike_counts(gids=gids, n_trials=n_trials) > 0)

    def get_trial_rank_count_per_gid(self, foll_gids, rank_method) -> pd.DataFrame:
        """
        :returns: an integer DF indicating how many times each neuron spiked
        in which rank order.

        Looks like:

            trial_rank  1   2  3  4  5
            gid
            1998        0   7  5  6  0
            2188        0   4  7  1  1
            9558        0  22  7  2  0
            14242       0   0  3  4  3
            30095       0   1  1  3  2

        """

        trial_ranks = self.get_trial_ranks(rank_method)

        counts = pd.DataFrame({
            'trial_rank': trial_ranks,
            'gid': self.spikes['gid'],
        })

        counts = counts.groupby(['trial_rank', 'gid']).size().unstack('trial_rank', fill_value=0)

        counts = counts.reindex(np.arange(len(foll_gids)) + 1, axis=1, fill_value=0)

        return counts

    def calc_order_entropy(self, trial_count, rank_method='first', adjust_missing=True):
        foll_gids = self.get_gids()

        counts = self.get_trial_rank_count_per_gid(foll_gids, rank_method)
        assert rank_method != 'first' or np.all(counts.sum() <= trial_count), counts.sum()

        probs = _counts_to_probs(counts, foll_gids, trial_count, adjust_missing=adjust_missing)

        entropies_norm = _get_entropies_norm(probs)

        return entropies_norm

    def calc_order_entropy_shuffled(
            self, trial_count,
            rank_method='first', adjust_missing=True, repeats=100, show_pbar=True):

        shuffled_entropies = {}

        repeats = np.arange(repeats)
        if show_pbar:
            repeats = pbar(repeats, desc='shuffles')

        for i in repeats:
            shuffled = self.shuffle_within_trial()

            shuffled_entropies[i] = shuffled.calc_order_entropy(trial_count, rank_method, adjust_missing=adjust_missing)

        shuffled_entropies = pd.DataFrame.from_dict(shuffled_entropies, orient='index')

        shuffled_entropies.rename_axis(index='repeat', columns='trial_rank', inplace=True)

        return shuffled_entropies


def _counts_to_probs(counts, foll_gids, trial_count, adjust_missing):
    """
    Normalise the number of times a follower takes a particular rank over all trials.

    Because not all followers fire on all trials, high-rank values are typically empty.
    This gives them a very low entropy.
    In Hemberger 2019 they adjust this by "padding" with the uniform distribution:

        'If some followers did not fire during a trial (number of neuron in the kth
        position divided by the number of trials was less than 1), we added to P_i
        of all neurons an equal amount such that sum(P_i) = 1'
    """
    ratios = counts / trial_count

    # ratios should be below one, allowing for floating errors
    assert np.all(ratios.sum() <= 1 + 1e-10), ratios.sum()
    ratios = ratios.clip(upper=1)

    if adjust_missing:
        missing = (1 - ratios.sum()) / len(foll_gids)
        probs = ratios + missing

    else:
        # in this case we may get invalid probability distributions
        # for high ranks that no cell ever takes.
        probs = ratios / ratios.sum()

    assert np.allclose(probs.sum()[ratios.sum() > 0], 1), probs.sum().sort_values()

    return probs


def _calc_entropies(probabilities):
    return (- probabilities * np.log2(probabilities)).fillna(0).sum(axis=0)


def _gen_probs_uniform(cell_gids):
    n_cells = len(cell_gids)
    probs = pd.DataFrame(1. / n_cells, columns=np.arange(1, n_cells + 1), index=cell_gids)
    return probs


def _get_entropies_norm(probs):
    entropies = _calc_entropies(probs)

    probs_uniform = _gen_probs_uniform(probs.index)
    entropies_uniform = _calc_entropies(probs_uniform)

    entropies_norm = entropies / entropies_uniform
    return entropies_norm


def plot_example_trials_multisim(
        sim_example_trials: dict,
        sim_spks: dict,
        gid_colors: pd.Series,
        gid_sorting: spt.Sorting,
        s=6, alpha=1, twin=(0, 100),
        suptitle=None
):
    """
    one col per sim, one row per trial

    :param sim_example_trials:
        dict of sim_gid, [list of trials]
        all sims must have the same number of trials!
    :param sim_spks:
    :param gid_colors:
    :param gid_sorting:
    :param s:
    :param alpha:
    :param twin:
    :param suptitle:
    :return:
    """
    n_trials = [len(trials) for trials in sim_example_trials.values()]
    n_trials = np.unique(n_trials)
    assert len(n_trials) == 1
    n_trials = n_trials[0]

    f, axs = plt.subplots(
        ncols=len(sim_example_trials), nrows=n_trials,
        squeeze=False,
        sharex='all', sharey='all',
        constrained_layout=True, figsize=(len(sim_example_trials) * .5 + .5, n_trials * .75 + .75),
    )

    if suptitle is not None:
        f.suptitle(suptitle)

    grouped_spikes = {}
    for j, (sim_label, trials) in enumerate(sim_example_trials.items()):
        for i, (trial) in enumerate(trials):
            ax = axs[i, j]

            grouped_spikes[i, j] = sim_spks[sim_label].sel(trial_idx=trial).sel_between(delay_from_induced=twin).spikes

            if i == 0:
                ax.set_title(f'{sim_label}\n{trial}')
            else:
                ax.set_title(f'{trial}')

            ax.set_xlim(twin[0] - 5, twin[1] + 5)
            ax.spines['bottom'].set_bounds(*twin)
            ax.set_xticks(twin)
            ax.spines['left'].set_visible(False)
            ax.tick_params(left=False, labelleft=False, which='both')

    all_shown_gids = pd.concat(list(grouped_spikes.values()))['gid'].unique()

    gid_sorting = gid_sorting.collapse(all_shown_gids)

    for (i, j), trial_spks in grouped_spikes.items():
        ax = axs[i, j]

        ax.scatter(
            trial_spks['delay_from_induced'],
            gid_sorting.apply(trial_spks['gid']),
            marker=plot.custom_marker_spike(height=1),
            facecolor=trial_spks['gid'].map(gid_colors),
            s=s,
            alpha=alpha,
        )

        ax.set_yticks(np.arange(len(gid_sorting.series)), minor=True)

    return axs
