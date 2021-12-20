import pandas as pd
import numpy as np
import logging
from tqdm.auto import tqdm as pbar
from tctx.analysis import simbatch as sb

from matplotlib import pyplot as plt
from tctx.util import plot


def _get_colors():
    style_aliases = {
        'only_a': ['a', 'a_and_not_b_and_not_c', 'a_and_not_b', 'a_and_not_c'],

        'only_b': ['b', 'not_a_and_b_and_not_c', 'not_a_and_b', 'b_and_not_c'],
        'only_c': ['c', 'not_a_and_not_b_and_c', 'not_b_and_c', 'not_a_and_c'],

        'a_or_b': ['a_and_b', 'a_and_b_and_not_c'],
        'b_or_c': ['b_and_c', 'not_a_and_b_and_c'],
        'a_or_c': ['a_and_c', 'a_and_not_b_and_c'],

        'none': [
            'not_a', 'not_b', 'not_c',
            'not_a_and_not_b', 'not_a_and_not_c', 'not_b_and_not_c',
            'not_a_and_not_b_and_not_c',
        ],

        'a_or_b_or_c': [
            'any', 'a_and_b_and_c',
        ],
    }

    colors = {
        'is_a': '#9642B0',
        'is_b': '#DB504A',
        'none': 'xkcd:black',

        # new
        'only_a': '#9642B0',
        'only_b': '#DB504A',
        'only_c': '#00CC66',
        'a_or_b': '#1E96FC',
        'a_or_c': '#F1A208',
        'b_or_c': '#C284A4',
        'a_or_b_or_c': 'xkcd:grey',

        'singles_and_double': 'xkcd:grey',
        'not_singles_and_double': '#00CC66',
        'singles_and_not_double': '#1E96FC',
        'not_singles_and_not_double': 'xkcd:black',
    }

    labels = {
        'only_a': 'only\na',
        'only_b': 'only\nb',
        'only_c': 'only\na+b',
        'a_or_b': 'a or b',
        'b_or_c': 'b or\na+b',
        'a_or_c': 'a or\na+b',
        'none': 'none',
        'a_or_b_or_c': 'any',

        'singles_and_double': 'always',
        'not_singles_and_double': 'only\na+b',
        'singles_and_not_double': 'only\na or b',
        'not_singles_and_not_double': 'none',
    }

    for k, v in style_aliases.items():
        for alias in v:
            if alias not in colors:
                colors[alias] = colors[k]
            if alias not in labels:
                labels[alias] = labels[k]

    return labels, colors


VENN_LABEL_ALIAS, VENN_COLORS = _get_colors()

_VENN_BOOL_LABELS = {
    (False, False, False): 'none',
    (True, False, False): 'only_a',
    (False, True, False): 'only_b',
    (True, True, False): 'a_and_b',
    (False, False, True): 'only_c',
    (True, False, True): 'a_and_c',
    (False, True, True): 'b_and_c',
    (True, True, True): 'any',
}

_VENN3_BOOL_COMBS = [
    (False, False,  True),
    (False,  True, False),
    (False,  True,  True),
    (True, False, False),
    (True, False,  True),
    (True,  True, False),
    (True,  True,  True)
]


class VennFollSets:
    def __init__(self, a, b, foll_a, foll_b, foll_ab):
        self.a = a
        self.b = b
        self.sets = {
            'a': set(foll_a),
            'b': set(foll_b),
            'c': set(foll_ab),
        }

    def get_venn_labels(self):
        belong = self.get_belong()

        # noinspection PyTypeChecker
        labels = belong.apply(tuple, axis=1).map(_VENN_BOOL_LABELS)

        labels.index = belong.index

        labels.loc[self.a] = 'is_a'
        labels.loc[self.b] = 'is_b'

        return labels

    def get_belong(self) -> pd.DataFrame:
        all_foll_gids = list(set.union(*self.sets.values()))

        belong = pd.DataFrame({
            k: np.isin(all_foll_gids, list(gids))
            for k, gids in self.sets.items()
        }, index=all_foll_gids)

        belong.rename_axis(index='gid', columns='mode', inplace=True)

        return belong

    def get_mode_counts(self):
        """
        :return: example:

                    a      b      c
            False  False  True     82
                   True   False     0
                          True      0
            True   False  False     0
                          True      0
                   True   False    14
                          True     16
            Name: mode_count, dtype: int64
        """
        belong = self.get_belong()

        patterns, counts = np.unique(belong, axis=0, return_counts=True)

        counts = pd.Series(
            counts,
            index=pd.MultiIndex.from_arrays(patterns.T, names=belong.columns)
        )

        counts = counts.reindex(_VENN3_BOOL_COMBS, fill_value=0)
        counts.rename('mode_count', inplace=True)

        return counts


class VennCounts:
    def __init__(self, table: pd.DataFrame):
        # sort starting with 'none', then 'only_X', then pairs, etc.
        table = table.sort_index(level=table.columns.names[::-1], axis=1)

        self.table = table

    def get_sum_comparison(self, naive=('a', 'b'), actual=('c',)):
        return pd.DataFrame.from_dict({
            'naive': sum([
                self.marginal(cond).table[True] for cond in naive
            ]),
            'actual': sum([
                self.marginal(cond).table[True] for cond in actual
            ]),
        })

    def drop_dummy_levels(self):
        level_counts = self._get_n_options_per_level()
        # noinspection PyTypeChecker
        bad_levels = level_counts.index[level_counts <= 1]
        return self.__class__(
            self.table.droplevel(bad_levels, axis=1)
        )

    def drop_options(self, *which):
        table = self.table
        for cond in which:
            table = table.drop(cond, axis=1)

        return self.__class__(table)

    @property
    def totals(self):
        return self.table.sum(axis=1)

    def _get_n_options_per_level(self) -> pd.Series:
        return pd.Series({
            level: condition.nunique()
            for level, condition in self.table.columns.to_frame().items()
        })

    def cond_on(self, cond, being=True):
        mask = self.table.columns.get_level_values(cond).values.astype(np.bool_)
        if not being:
            mask = ~mask

        new_counts = self.table.loc[:, mask].copy()
        # new_counts.columns = new_counts.columns.droplevel(cond)
        new_counts = new_counts.T.groupby(new_counts.columns.names, axis=0).sum().T

        return self.__class__(new_counts)

    def copy(self):
        return self.__class__(self.table.copy())

    def integrate_out(self, *conds):
        out = self.copy()
        for cond in conds:
            pre = out

            being_true = out.cond_on(cond, being=True).drop_dummy_levels()
            being_false = out.cond_on(cond, being=False).drop_dummy_levels()
            out = being_true + being_false

            assert len(out.totals.index.difference(pre.totals.index)) == 0
            assert np.all(out.totals == pre.totals.reindex(out.table.index, fill_value=0))

        return out

    def marginal(self, cond):
        conds = list(self.conditions)
        conds.remove(cond)
        return self.integrate_out(*conds)

    @property
    def conditions(self):
        return self.table.columns.names

    def __add__(self, other):
        return self.__class__(self.table + other.table)

    def __len__(self):
        return len(self.table)

    def _repr_html_(self):
        """pretty printing for jupyter"""
        # noinspection PyProtectedMember
        return self.table._repr_html_()

    def get_label(self, condition):
        if self.table.columns.nlevels == 1:
            condition = (condition,)

        condition_names = self.table.columns.names
        assert len(condition_names) == len(condition)

        return '_and_'.join([
            ('not_' if not c else '') + n
            for n, c in zip(condition_names, condition)
        ])

    def items(self):
        for k, v in self.table.items():
            yield self.get_label(k), v

    def plot_boxes(self, ax=None, skip=('none',), q1=.25, q2=.5, q3=.75, symlog=False, ylim=None, ylabel='prob', noise_hist=True):
        if ax is None:
            _, ax = plt.subplots(constrained_layout=True, figsize=(4.5, 1.25))

        ticks = {}
        for i, (label, vs) in enumerate(self.items()):

            if not (VENN_LABEL_ALIAS[label] in skip or label in skip):

                main = VENN_COLORS[label]
                light = plot.lighten_color(main, .5)
                # dark = plot.lighten_color(main, 2)

                plot.plot_scatter_n_box(
                    ax,
                    i,
                    vs,
                    q1=q1, q2=q2, q3=q3,
                    rect_facecolor=light,
                    rect_edgecolor='none',
                    facecolor=main,
                    noise_hist=noise_hist,
                    clip_on=False,
                )

                ticks[i] = VENN_LABEL_ALIAS[label]

        ax.spines['bottom'].set_position(('outward', 2))

        ax.set_ylabel(ylabel)
        ax.set_xticks(list(ticks.keys()))
        ax.set_xticklabels(list(ticks.values()))
        ax.tick_params(axis='x', rotation=0, labelsize=6)

        if symlog:
            ax.set_yscale('symlog', linthreshy=1)

            yticks = [0, 1, 10, 100, 1000]

            ax.set_yticks(yticks)
            ax.set_yticks(np.concatenate([np.linspace(0, 1, 11)[1:-1] * t for t in yticks[2:]]), minor=True)
            ax.set_yticklabels(['0', '1', '10', '100', '1k'])

            ax.tick_params(length=3, which='both', axis='y')
            ax.tick_params(bottom=False)

            ax.set_ylim(-.1, 1e3)
            ax.spines['left'].set_bounds(0, 1e3)

        if ylim is not None:
            ax.set_ylim(ylim)

class MultipingBatch:
    def __init__(self, singles, doubles):
        self.singles = singles
        self.doubles = doubles

    @classmethod
    def from_batch(cls, batch_multi: sb.SimBatch):
        assert (batch_multi.reg.groupby(['targeted_gid', 'targeted_gid_k']).size().sort_values() == 1).all()

        batch_single = batch_multi.subsection(batch_multi.reg['targeted_gid_k'].isna())
        batch_single = batch_single.reg.rename_axis(index='sim_gid').reset_index().groupby('targeted_gid').first().sort_index()
        batch_single.dropna(subset=['e_foll_gids', 'i_foll_gids'], inplace=True)
        assert batch_single.index.is_unique

        batch_double_t0 = batch_multi.subsection(batch_multi.reg['targeted_gid_k'].notna())
        batch_double_t0 = batch_double_t0.reg.rename_axis(index='sim_gid').reset_index().groupby(['targeted_gid', 'targeted_gid_k']).first().sort_index()
        batch_double_t0.dropna(subset=['e_foll_gids', 'i_foll_gids'], inplace=True)
        assert batch_double_t0.index.is_unique

        # batch_double_t0 = batch_double_t0.loc[[targeted_gid]]

        return cls(
            batch_single,
            batch_double_t0
        )

    def get_foll_sets(self, a, b):
        """Return the sets of followers to only-a, only-b and a-and-b"""
        return VennFollSets(
            a=a, b=b,
            foll_a=self.singles.loc[a, ['i_foll_gids', 'e_foll_gids']].sum(),
            foll_b=self.singles.loc[b, ['i_foll_gids', 'e_foll_gids']].sum(),
            foll_ab=self.doubles.loc[(a, b), ['i_foll_gids', 'e_foll_gids']].sum(),
        )

    def iter_foll_sets(self):
        for gid_a, gid_b in pbar(self.doubles.index, desc='cell pairs'):
            if gid_a in self.singles.index and gid_b in self.singles.index:

                yield self.get_foll_sets(gid_a, gid_b)

    def extract_venn_counts(self, total_cells=100_000) -> VennCounts:
        """

        :return: example:

            a               False             True
            b               False True        False       True
            c               True  False True  False True  False True

            a       b
            294.0   65058.0    19     0     0     0     0    14    39
            504.0   85220.0    73     0     0     0     0     0     0
            ...               ...   ...   ...   ...   ...   ...   ...
            88805.0 16261.0    77     0     0     0     0     3     2
            92973.0 4640.0     35     0     0     0     0    19    36

            [1712 rows x 7 columns]

        """
        venn_counts = {
            (foll_sets.a, foll_sets.b): foll_sets.get_mode_counts()
            for foll_sets in self.iter_foll_sets()
        }

        mode_names = list(venn_counts.values())[0].index.names

        venn_counts = pd.DataFrame.from_dict(venn_counts, orient='index').fillna(0).astype(np.int)
        venn_counts.rename_axis(index=['a', 'b'], columns=mode_names, inplace=True)

        venn_counts = venn_counts.reindex(_VENN3_BOOL_COMBS, axis=1, fill_value=0)

        assert (False, False, False) not in venn_counts.columns
        venn_counts[False, False, False] = total_cells - venn_counts.sum(axis=1)

        return VennCounts(venn_counts)

    def get_sims(self, a, b):
        interesting_sims = {
            'a': self.singles.loc[a, 'sim_gid'],
            'b': self.singles.loc[b, 'sim_gid'],
            'c': self.doubles.loc[(a, b), 'sim_gid'],
        }
        return interesting_sims

