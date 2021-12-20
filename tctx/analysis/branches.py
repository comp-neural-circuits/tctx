"""Code to determine branches of strong connections from activity"""

import logging

import numpy as np
import pandas as pd
from tqdm.auto import tqdm as pbar

import matplotlib.ticker
import matplotlib.cm
import matplotlib.colors

from tctx.util import spike_trains as spt, plot, parallel, plot_graph

from tctx.analysis import simbatch as sb
from tctx.analysis import amat as am


########################################################################################################################
# Plotting


def _get_label_colors(count=100):
    label_colors = {
        np.nan: 'xkcd:grey',
        -3: 'xkcd:charcoal',
        -2: 'xkcd:grey',
        -1: 'xkcd:purple',
        0: plot.styles_df.loc['cluster_a', 'main'],  # green
        1: plot.styles_df.loc['cluster_b', 'main'],  # orange
        2: plot.styles_df.loc['cluster_c', 'main'],  # yellow
        3: plot.styles_df.loc['cluster_d', 'main'],  # brown-ish
        4: 'xkcd:royal blue',
        5: 'xkcd:red',
        6: 'xkcd:pink',
        7: 'xkcd:cyan',
        8: 'xkcd:olive',
        9: 'xkcd:coral',
        10: 'xkcd:black',
        11: 'xkcd:sage',
        12: 'xkcd:sienna',
        13: 'xkcd:sick green',
        14: 'xkcd:cloudy blue',
        15: 'xkcd:strong pink',
        16: 'xkcd:windows blue',
        17: 'xkcd:purpley grey',
        18: 'xkcd:old rose',
        19: 'xkcd:seafoam',
        20: 'xkcd:baby blue',
    }

    for i in range(int(np.nanmax(list(label_colors.keys()))), count):
        label_colors[i] = matplotlib.cm.get_cmap('jet')(np.random.rand())

    label_colors['a'] = label_colors[0]
    label_colors['b'] = label_colors[1]

    return label_colors


LABEL_COLORS = _get_label_colors()

LABEL_COLORS_DARK = {
    k: plot.lighten_color(v, 1.5)
    for k, v in LABEL_COLORS.items()
}


def _colored_matrix_set_item_labels(row_labels, col_labels, by='both') -> np.ndarray:
    """
    produce a single matrix with integer values representing the
    merged label of row & columns
    """
    if by == 'none':
        label_mat = np.zeros((len(row_labels), len(col_labels)))

    elif by == 'col':
        label_mat = np.tile(col_labels, (len(row_labels), 1))

    elif by == 'row':
        label_mat = np.tile(row_labels, (len(col_labels), 1)).T

    else:
        label_mesh = np.array(np.meshgrid(col_labels, row_labels))
        _, combined_label = np.unique(label_mesh.reshape(2, -1).T, axis=0, return_inverse=True)

        label_mat = combined_label.reshape((len(row_labels), len(col_labels)))

    return label_mat


def _colored_matrix_get_mapping(
        label_colors, background_color='#EFEFEF', outside_color='xkcd:charcoal',
        vmin=-10000, vmax=1000
):
    """
    Generates a matplotlib-friendly cmap and norm objects
    """

    label_colors = label_colors.sort_index()

    outside_color = np.array(matplotlib.colors.to_rgb(outside_color))
    colors = [outside_color] + list(label_colors.values) + [outside_color]

    label_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'label_cmap', colors, len(colors))

    label_cmap.set_bad(color=background_color)

    if len(label_colors) > 1:
        boundaries = label_colors.index.values
        boundaries = (boundaries[1:] + boundaries[:-1]) * .5
        boundaries = np.concatenate([[vmin, np.min(boundaries) - 1], boundaries, [np.max(boundaries) + 1, vmax]])
    else:
        assert len(label_colors) == 1
        v = label_colors.index.values[0]
        boundaries = [vmin, v - 1, v + 1, vmax]

    norm = matplotlib.colors.BoundaryNorm(boundaries, len(colors))

    return label_cmap, norm


def _colored_matrix_mark_trials(ax, amat_sorted, example_trial_idcs, facecolor=None, s=30):
    """Draw a little arrow below the given trials"""

    if facecolor is None:
        facecolor = 'k'

    ax.scatter(
        [amat_sorted.columns.get_loc(trial_idx) for trial_idx in example_trial_idcs],
        y=[0] * len(example_trial_idcs),
        marker='^',
        facecolor=facecolor,
        transform=ax.get_xaxis_transform(),
        s=s,
        clip_on=False,
    )


def plot_colored_matrix(
        ax, amat: pd.DataFrame,
        row_label=None, col_label=None,
        label_colors=None, color_by=None, background_color='#EFEFEF',
        mark_trials=None, mark_trials_colors=None,
):
    """labels must come sorted and sharing index with the index/column of the matrix"""

    if label_colors is None:
        label_colors = LABEL_COLORS

    if color_by is None:
        if row_label is not None and col_label is None:
            color_by = 'row'
        elif row_label is None and col_label is not None:
            color_by = 'col'
        elif row_label is None and col_label is None:
            color_by = 'none'
        else:
            color_by = 'both'

    if row_label is None:
        row_label = np.ones(amat.shape[0])

    if not isinstance(row_label, pd.Series):
        row_label = pd.Series(np.asarray(row_label), index=amat.index)

    row_label = row_label.reindex(amat.index)

    if col_label is None:
        col_label = np.ones(amat.shape[1])

    if not isinstance(col_label, pd.Series):
        col_label = pd.Series(np.asarray(col_label), index=amat.columns)

    col_label = col_label.reindex(amat.columns)

    labels = _colored_matrix_set_item_labels(row_label, col_label, color_by)

    unique_labels, labels_rebased = np.unique(labels, return_inverse=True)

    labels_rebased = labels_rebased.reshape(labels.shape)

    # we're going to use nan to represent False
    labels_rebased = labels_rebased.astype(np.float)
    labels_rebased[~amat.values] = np.nan

    mapping = dict(zip(unique_labels, np.arange(len(unique_labels))))
    label_colors_rebased = pd.Series({ni: label_colors[i] for i, ni in mapping.items()})

    cmap, norm = _colored_matrix_get_mapping(label_colors_rebased, background_color)
    ax.imshow(labels_rebased, cmap=cmap, norm=norm, origin='lower')

    plot.remove_spines(ax)
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

    if mark_trials is not None:
        _colored_matrix_mark_trials(ax, amat, mark_trials, facecolor=mark_trials_colors)


class RoutingGraphPlot:
    """A single figure of a graph showing the path activity took"""

    def __init__(self, graph: plot_graph.Graph, source_gid):
        self.graph = graph
        self.source_gid = source_gid

    def copy(self):
        return self.__class__(self.graph.copy(), self.source_gid)

    def get_scaled_node_size(self, node_count_range=(15, 50), node_size_range=(5, 30)):
        """linear interpolation of node size based on graph size"""
        cell_count = len(self.graph.nodes)
        size_norm = (cell_count - node_count_range[0]) / (node_count_range[1] - node_count_range[0])
        node_size = node_size_range[1] - (node_size_range[1] - node_size_range[0]) * size_norm

        # noinspection PyTypeChecker
        node_size = int(min(max(node_size_range[0], node_size), node_size_range[1]))

        return node_size

    @classmethod
    def prepare_graph(
            cls,
            interesting_cells: pd.DataFrame,
            sel_conns: pd.DataFrame,
            color_bkg='xkcd:light grey',
            orientation='vertical',
    ):
        source = interesting_cells[interesting_cells['is_targeted']]
        assert len(source) == 1, source
        source_gid = source.index

        nodes = interesting_cells.copy()
        sb.CAT.add_cats_cells(nodes)
        nodes['style'] = interesting_cells['ei_type'].astype(str)

        edges = sel_conns.copy()
        valid = edges['source'].isin(interesting_cells.index) & edges['target'].isin(interesting_cells.index)
        if not np.all(valid):
            logging.warning(f'Dropping {np.count_nonzero(~valid)}/{len(valid)} conns with missing nodes')
            edges = edges[valid]
        sb.CAT.add_cats_conns(edges)
        edges['style'] = edges['con_type'].map(lambda x: f'{x[:1]}')

        graph = plot_graph.Graph(nodes, edges)
        graph.styles.loc['e-bkg'] = plot.style_mix('grey', marker_space='e', main=color_bkg)
        graph.styles.loc['i-bkg'] = plot.style_mix('grey', marker_space='i', main=color_bkg)

        graph.layout_best_fit(around=source_gid[0], orientation=orientation)

        return cls(graph, source_gid)

    @staticmethod
    def _get_active(graph, active_nodes, active_edges) -> (pd.Series, pd.Series):
        if active_nodes is None:
            active_nodes = pd.Series(True, index=graph.nodes.index)

        if active_nodes.dtype != 'bool':
            active_nodes = pd.Series(True, index=active_nodes)

        active_nodes = active_nodes.reindex(graph.nodes.index, fill_value=False)

        if isinstance(active_edges, str):
            active_target = active_nodes.reindex(graph.edges['target'], fill_value=False).values
            active_source = active_nodes.reindex(graph.edges['source'], fill_value=False).values

            active_edges = active_edges.lower()
            if active_edges == 'target':
                active_edges = pd.Series(active_target, index=graph.edges.index)

            elif active_edges == 'source':
                active_edges = pd.Series(active_source, index=graph.edges.index)

            elif active_edges == 'both':
                active_edges = pd.Series(active_source & active_target, index=graph.edges.index)

            elif active_edges == 'any':
                active_edges = pd.Series(active_source | active_target, index=graph.edges.index)

            elif active_edges == 'all':
                active_edges = pd.Series(False, index=graph.edges.index)

            else:
                assert active_edges == 'none', f'unknown value "{active_edges}"'
                active_edges = pd.Series(False, index=graph.edges.index)

        else:
            if active_edges is None:
                active_edges = pd.Series(True, index=graph.edges.index)

            if active_edges.dtype != 'bool':
                active_edges = pd.Series(True, index=active_edges)

        assert isinstance(active_edges, pd.Series)
        active_edges = active_edges.reindex(graph.edges.index, fill_value=False)

        return active_nodes, active_edges

    def plot_strong(
            self, ax,
            weight_threshold=15., extra_conns=None, weak_alpha=.25,
            active_nodes=None, node_size=None, **kwargs,
    ):
        graph = self.graph.copy()

        if extra_conns is not None:
            extra_conns = extra_conns.loc[extra_conns.index.difference(graph.edges.index)]
            graph.add_edges(extra_conns, style='default')

        graph.edges.loc[graph.edges['weight'] >= weight_threshold, 'style'] = 'strong'
        graph.edges.loc[graph.edges['weight'] < weight_threshold, 'style'] = 'weak'
        graph.edges.sort_values('weight', ascending=False, inplace=True)

        active_nodes, active_edges = RoutingGraphPlot._get_active(graph, active_nodes=active_nodes, active_edges=None)
        self._style_bkg(graph, active_nodes, active_edges)

        RoutingGraphPlot._plot_solid(
            ax, graph,
            node_size=self.get_scaled_node_size() if node_size is None else node_size,
            nodes_kwargs=kwargs,
            edges_kwargs=dict(style=dict(
                strong=dict(zorder=101),
                weak=dict(zorder=100, alpha=weak_alpha),
            )))

    def plot_strong_cmap_edges(
            self, ax,
            c='weight',
            extra_conns=None,
            active_nodes=None,
            node_size=None,
            cmap=None,
            norm=None,
            **edges_kwargs,
    ):
        graph = self.graph.copy()

        if extra_conns is not None:
            extra_conns = extra_conns.loc[extra_conns.index.difference(graph.edges.index)]
            graph.add_edges(extra_conns, style='default')

        if isinstance(c, str):
            graph.edges.sort_values(c, ascending=False, inplace=True)
            c = graph.edges[c]

        node_size = self.get_scaled_node_size() if node_size is None else node_size

        active_nodes, active_edges = RoutingGraphPlot._get_active(graph, active_nodes=active_nodes, active_edges=None)
        self._style_bkg(graph, active_nodes, active_edges)

        nodes_kwargs = RoutingGraphPlot._default_node_kwargs(
            clip_on=False, graph=graph, node_size=30, nodes_kwargs=None,
            facecolor='main',
        )
        graph.plot_nodes_solid(ax, **nodes_kwargs)

        all_sm = graph.plot_edges_cmap(
            ax,
            c=c,
            head_length=(node_size / 30.) * 2,
            head_width=(node_size / 30.) * 1.,
            head_offset=2,
            zorder=500,
            clip_on=False,
            cmap=cmap,
            norm=norm,
            **edges_kwargs,
        )
        for sm in all_sm.values():
            ax.figure.colorbar(sm, ax=ax)

        RoutingGraphPlot._setup_ax(ax)

    def plot_clustered(
            self, ax, cell_labels,
            label_colors=None,
            active_nodes=None, active_edges='both', node_size=None, **kwargs,
    ):

        if label_colors is None:
            label_colors = LABEL_COLORS

        assert len(cell_labels) == len(self.graph.nodes), \
            f'Got {len(cell_labels)} labels, expected {len(self.graph.nodes)}'

        if not isinstance(cell_labels, pd.Series):
            cell_labels = pd.Series(np.asarray(cell_labels), index=self.graph.nodes.index)

        graph = self.graph.copy()
        graph.style_by_node_labels(cell_labels, label_colors, style_edges='source')

        active_nodes, active_edges = RoutingGraphPlot._get_active(graph, active_nodes, active_edges)
        self._style_bkg(graph, active_nodes, active_edges)

        RoutingGraphPlot._plot_solid(
            ax, graph,
            node_size=self.get_scaled_node_size() if node_size is None else node_size,
            nodes_kwargs=kwargs
        )

    @staticmethod
    def _style_bkg(graph, active_nodes, active_edges):
        ei_types = graph.nodes.loc[~active_nodes, 'ei_type']
        graph.nodes.loc[~active_nodes, 'style'] = ei_types.astype(str).map(lambda x: f'{x}-bkg')

        graph.nodes.loc[graph.nodes['is_targeted'], 'style'] = 'induced'

        con_types = graph.edges.loc[~active_edges, 'con_type']
        graph.edges.loc[~active_edges, 'style'] = con_types.astype(str).map(lambda x: f'{x[:1]}-bkg')

    @staticmethod
    def _plot_solid(ax, graph, node_size=30, clip_on=False, nodes_kwargs=None, edges_kwargs=None):
        nodes_kwargs = RoutingGraphPlot._default_node_kwargs(
            clip_on, graph, node_size, nodes_kwargs,
            facecolor='main',
        )
        graph.plot_nodes_solid(ax, **nodes_kwargs)

        RoutingGraphPlot._plot_edges_solid(ax, clip_on, edges_kwargs, graph, node_size)
        RoutingGraphPlot._setup_ax(ax)

    def plot_node_labels(self, ax, fontsize=6, active_nodes=None, zorder=5000):
        active_nodes, active_edges = RoutingGraphPlot._get_active(self.graph, active_nodes, active_edges=None)
        self.graph.plot_nodes_labels(ax, nodes=active_nodes, fontsize=fontsize, zorder=zorder)

    @staticmethod
    def _plot_edges_solid(ax, clip_on, edges_kwargs, graph, node_size):

        edges_kwargs = edges_kwargs if edges_kwargs is not None else {}

        for name in graph.styles.index:
            if name.endswith('-bkg'):
                edges_kwargs.setdefault('style', {})
                edges_kwargs['style'].setdefault(name, {})
                edges_kwargs['style'][name].setdefault('zorder', 250)

        graph.plot_edges_solid(
            ax,
            color='main',
            head_length=(node_size / 30.) * 2,
            head_width=(node_size / 30.) * 1.,
            head_offset=2,
            zorder=500,
            clip_on=clip_on,
            **edges_kwargs,
        )

    @staticmethod
    def _default_node_kwargs(clip_on, graph, node_size, nodes_kwargs, **extra):
        nodes_kwargs = nodes_kwargs if nodes_kwargs is not None else {}

        style = dict(
            induced=dict(zorder=1750, linewidth=.75, facecolor='light', edgecolor='darker'),
        )

        for name in graph.styles.index:
            if name.endswith('-bkg'):
                style.setdefault(name, {})
                style[name].setdefault('zorder', 1250)

        defaults = dict(
            marker='marker_space',
            s=node_size,
            linewidth=0,
            zorder=1500,
            style=style,
            clip_on=clip_on,
        )

        nodes_kwargs = {**defaults, **extra, **nodes_kwargs}

        return nodes_kwargs

    @staticmethod
    def _setup_ax(ax):
        ax.set_aspect('equal')
        for name in 'left', 'bottom':
            ax.spines[name].set_visible(False)
            ax.tick_params(**{f'{name}': False, f'label{name}': False})


########################################################################################################################
# Clustering

def _extract_batch_one(
        spikes: am.SpikeBundle,
        foll_gids: pd.Index,
        ei_type='all',
        trial_count=100,
        target_cluster_size=6,
        n_init=200
) -> pd.Series:
    """
    perform k-modes clustering on follower cells by their activity,
    """
    spikes = spikes.sel(cat='effect')
    spikes = spikes.sel_isin(gid=foll_gids)

    if ei_type is not None and ei_type != 'all':
        spikes = spikes.sel(ei_type=ei_type)
        # print(f'{len(spikes.spikes)} ei_type spikes', flush=True)

    amat = spikes.get_activation_matrix(n_trials=trial_count, gids=foll_gids)
    # print(f'amat:\n{amat.table}', flush=True)

    labels = amat.generate_kmodes_labels(target_cluster_size=target_cluster_size, n_init=n_init)

    # for consistency, make two biggest labels, the early numbers
    labels = remap_labels_by_size(labels)

    assert len(labels) == len(foll_gids)

    return labels


def extract_batch(
        batch_original, sim_gids,
        target_cluster_size=5,
        ei_type='all',
        exec_mode=None,
        n_init=200,
):
    """perform k-modes clustering on follower cells by their activity,
     independently for each simulation"""

    all_foll_gids = batch_original.reg['e_foll_gids'] + batch_original.reg['i_foll_gids']

    res = parallel.independent_tasks(
        _extract_batch_one,
        [
            (
                am.SpikeBundle(batch_original.stores['spikes'][sim_gid]),
                all_foll_gids.loc[sim_gid],
                ei_type,
                batch_original.reg.loc[sim_gid, 'trial_count'],
                target_cluster_size,
                n_init,
            )
            for sim_gid in pbar(sim_gids, desc='load spikes')
        ],
        mode=exec_mode,
    )

    # remember res may not come in the correct order
    all_labels = {
        sim_gid: res[i].to_frame()
        for i, sim_gid in enumerate(sim_gids)
    }

    # sanity check
    if ei_type == 'all':
        foll_counts = all_foll_gids.dropna().map(len)
        labeled_counts = pd.Series({sim_gid: len(labels) for sim_gid, labels in all_labels.items()})
        mismatched = labeled_counts.loc[sim_gids] != foll_counts.loc[sim_gids]
        if np.count_nonzero(mismatched) > 0:
            logging.warning(f'{np.count_nonzero(mismatched)} sims with mismatched'
                            f' number of followers and number of labeled followers')

    return all_labels


# noinspection PyTypeChecker
def remap_labels_by_size(labels: pd.Series) -> pd.Series:
    """
    The actual labels don't mean anything, and they may change
    every time we cluster items.
    This reassigns labels in inverse order of cluster size.
    It respects negative labels, which indicates special neurons
    (-1 is the trigger neuron)
    """

    special = labels[labels < 0]
    label_sizes = labels[labels >= 0].value_counts()

    sorting = pd.Series(
        label_sizes.sort_index().index,
        index=label_sizes.sort_values(ascending=False).index,
    )

    for v in special:
        sorting.loc[v] = v

    return labels.map(sorting)


def remap_labels_by_order(labels: pd.Series, new_order) -> pd.Series:
    """
    Make the given labels are assigned values 0, 1, etc..
    Labels not in "new_order" are ignored.
    Values are assigned starting at 0.
    """
    return remap_labels(labels, *[(k, i) for i, k in enumerate(new_order)])


def remap_labels(labels: pd.Series, *pairs) -> pd.Series:
    """
    The actual labels don't mean anything, and they may change
    every time we cluster items.

    For a series of pairs (current_label,  desired_label),
    this will swap labels to make sure that current_label becomes desired_label

    For example, the biggest cluster may have been labeled as 3, but we
    want to deal always with value "0" (for coloring). So you can:

    current_label = 3
    labels = remap_labels(labels, (current_label, 0))
    """

    label_mapping = pd.Series(labels.unique(), labels.unique()).sort_index()

    for target, desired in pairs:
        if target != desired:

            current_desired = label_mapping.index[label_mapping == desired]
            assert len(current_desired) == 1
            to_swap = current_desired[0]

            current_target = label_mapping.loc[target]

            label_mapping.loc[to_swap] = current_target
            label_mapping.loc[target] = desired

    assert label_mapping.is_unique

    return labels.map(label_mapping)


def invalidate_labels_except(labels, which):
    """
    Set all labels except the given ones (and special -1) to
    invalid label -2.
    """

    labels = labels.copy()

    which = list(which)

    labels[~labels.isin([-1] + which)] = -2

    return labels


def extract_cluster_stats(
        conns_store: sb.SplitStore,
        cells_store: sb.SplitStore,
        labels_store: sb.SplitStore,
        weight_thresh,
        ei_type='e',
        sim_gids=None,
) -> pd.DataFrame:
    """
    Extract a table with all of the extracted clusters and associated statistics.
    Looks like:

               sim_gid  label  ...  prob_conn_between  total_cells
        0            2      0  ...                0.0           86
        1            2      1  ...                0.0           86
        2            2      2  ...                0.0           86
        3            2      3  ...                0.0           86
        4            2      4  ...                0.0           86
        ...        ...    ...  ...                ...          ...
        20499    18047     34  ...                0.0          239
        20500    18047     35  ...                0.0          239
        20501    18047     36  ...                0.0          239
        20502    18047     37  ...                0.0          239
        20503    18047     38  ...                0.0          239

    """

    all_stats = []

    if sim_gids is None:
        sim_gids = conns_store.keys().intersection(labels_store.keys())

    try:

        for sim_gid in pbar(sim_gids, desc='sims'):

            all_labels = labels_store[sim_gid]
            if isinstance(all_labels, pd.DataFrame):
                assert len(all_labels.columns) == 1
                all_labels = all_labels.iloc[:, 0]

            cells = cells_store[sim_gid]
            sb.CAT.add_cats_cells(cells)

            foll_conns = conns_store[sim_gid]
            sb.CAT.add_cats_conns(foll_conns)

            # drop targeted cell if it got saved
            all_labels = all_labels[all_labels >= 0]

            for label0, gids0 in all_labels.groupby(all_labels).groups.items():
                others = all_labels.index.difference(gids0)

                pop_labels = all_labels.loc[cells.loc[all_labels.index, 'ei_type'] == ei_type]

                foll_conns = foll_conns[foll_conns['weight'] >= weight_thresh]

                if len(foll_conns) == 0 and len(pop_labels) > 1:
                    logging.warning(f'No foll conns selected for sim {sim_gid} (with {len(pop_labels)} folls)')

                source_in_l0 = foll_conns['source'].isin(gids0)
                target_in_l0 = foll_conns['target'].isin(gids0)
                conn_within = source_in_l0 & target_in_l0

                source_in_others = foll_conns['source'].isin(others)
                target_in_others = foll_conns['target'].isin(others)
                conn_between = (source_in_l0 & target_in_others) | (source_in_others & target_in_l0)

                stats = {
                    'sim_gid': sim_gid,
                    'label': label0,
                }

                for ei_type in 'ei':
                    stats[f'cluster_size_{ei_type}'] = np.count_nonzero(cells.loc[gids0, 'ei_type'] == ei_type)
                    stats[f'others_size_{ei_type}'] = np.count_nonzero(cells.loc[others, 'ei_type'] == ei_type)

                for con_type in ['e2e', 'e2i', 'i2e', 'i2i']:
                    mask = conn_between & (foll_conns['con_type'] == con_type)
                    stats[f'between_count_{con_type}'] = np.count_nonzero(mask)

                    mask = conn_within & (foll_conns['con_type'] == con_type)
                    stats[f'within_count_{con_type}'] = np.count_nonzero(mask)

                all_stats.append(stats)

    except KeyboardInterrupt:
        print('early exit')

    all_stats = pd.DataFrame(all_stats)

    # note that connections are directional and we don't allow autapses
    for s in 'ei':
        for t in 'ei':
            con_type = f'{s}2{t}'

            cluster_size_s = all_stats[f'cluster_size_{s}']
            others_size_s = all_stats[f'others_size_{s}']

            cluster_size_t = all_stats[f'cluster_size_{t}']
            others_size_t = all_stats[f'others_size_{t}']

            within_count = all_stats[f'within_count_{con_type}']
            between_count = all_stats[f'between_count_{con_type}']

            if s == t:
                potential_within = (cluster_size_s * (cluster_size_t - 1))
            else:
                potential_within = (cluster_size_s * cluster_size_t * 2)

            all_stats[f'within_prob_{con_type}'] = within_count / potential_within
            assert all_stats[f'within_prob_{con_type}'].dropna().between(0, 1).all()

            potential_within = cluster_size_s * others_size_t + others_size_s * cluster_size_t
            all_stats[f'between_prob_{con_type}'] = between_count / potential_within
            assert all_stats[f'between_prob_{con_type}'].dropna().between(0, 1).all()

    all_stats['total_cells'] = all_stats[[
        'cluster_size_e', 'others_size_e',
        'cluster_size_i', 'others_size_i']].sum(axis=1)

    assert (all_stats.groupby('sim_gid')['total_cells'].nunique() == 1).all()

    return all_stats


def get_labels(batch, sim_gid, store_name='kmodes_all_6'):
    labels = batch.stores[store_name][sim_gid]

    if isinstance(labels, pd.DataFrame):
        assert len(labels.columns) == 1
        labels = labels.iloc[:, 0]

    targeted_gid = batch.reg.loc[sim_gid, 'targeted_gid']
    labels.loc[targeted_gid] = -1

    return labels


def get_spikes(batch, sim_gid, ei_type):
    foll_gids = batch.reg.loc[sim_gid, f'{ei_type}_foll_gids'] # + batch.reg.loc[sim_gid, 'i_foll_gids']

    spikes = am.SpikeBundle(batch.stores['spikes'][sim_gid])
    spikes = spikes.sel(cat='effect')
    spikes = spikes.sel_isin(gid=foll_gids)

    return spikes


def get_gate_per_clust(batch, stats):
    """For each cluster, collect the ID of the cell of the cluster with shortest median delay"""

    gates = {}

    try:
        for sim_gid, clust_idcs in pbar(stats.groupby('sim_gid').groups.items()):
            labels = get_labels(batch, sim_gid)

            spikes = get_spikes(batch, sim_gid, 'e')

            gate_per_label = spikes.spikes.groupby('gid')['delay_from_induced'].median().groupby(labels).idxmin()

            for clust_idx in clust_idcs:

                gates[clust_idx] = gate_per_label.get(stats.loc[clust_idx, 'label'], np.nan)

    except KeyboardInterrupt:
        pass

    return pd.Series(gates)


##################################################################################################################
# process the presence of labels
# TODO replace these with methods in Amat class

def extract_cell_amat(cell_gids, spikes, trial_times) -> pd.DataFrame:
    """
    Extract the activation matrix of each cell in each trial.
    An entry in the matrix will be True if that cell spiked at least once in that trial.

    :returns: a pd.DataFrame with binary entries and shape <gids, trials>
    """
    foll_spikes = spikes[spikes['gid'].isin(cell_gids)]

    binned_foll_spikes = spt.assign_spikes_to_windows(foll_spikes, spt.make_windows(trial_times, (0, 300)))

    spike_count_mat = binned_foll_spikes.groupby(['gid', 'win_idx']).size().unstack('win_idx', fill_value=0)
    spike_count_mat = spike_count_mat.reindex(np.arange(len(trial_times)), axis=1, fill_value=0)
    spike_count_mat = spike_count_mat.reindex(cell_gids, axis=0, fill_value=0)

    cell_amat = spike_count_mat > 0

    return cell_amat
