"""
Utils to plot graphs with arrows
"""

import matplotlib.transforms
import matplotlib.patches
import matplotlib.colors
import matplotlib.cm

import numpy as np
import pandas as pd

import logging

from tctx.util import plot


def _clip_arrows(arrows, tail_offset, head_offset):
    """
    shorten head & tail so the arrows don't overlap with markers

    :param arrows: a pd.DataFrame with columns: 'source_x', 'source_y', 'target_x', 'target_y'
    :param tail_offset: how much shorter to make the tail (so it doesn't overlap with the markers)
    :param head_offset: how much shorter to make the head (so it doesn't overlap with the markers)
    :return: 2 numpy arrays of shape Nx2
    """
    source_pos = arrows[['source_x', 'source_y']].values
    target_pos = arrows[['target_x', 'target_y']].values
    direction = target_pos - source_pos
    length = np.sqrt(np.sum(np.square(direction), axis=1))
    direction = direction / length[:, np.newaxis]

    source_pos = source_pos + direction * tail_offset
    target_pos = target_pos + direction * (-1 * head_offset)

    return source_pos, target_pos


def plot_arrows_cmap(
        ax, arrows, c, cmap=None, norm=None,
        tail_offset=0, head_offset=0, head_length=4, head_width=1.25, **kwargs):
    """

    Draw multiple arrows using a colormap.

    :param ax: matplotlib.axes.Axes
    :param arrows: a pd.DataFrame with columns: 'source_x', 'source_y', 'target_x', 'target_y'
    :param c: a pd.Series with the same index as arrows or a string that identifies a column in it.
    :param tail_offset: how much shorter to make the tail (so it doesn't overlap with the markers)
    :param head_offset: how much shorter to make the head (so it doesn't overlap with the markers)
    :param kwargs: args for matplotlib.patches.FancyArrowPatch
    :return: matplotlib.cm.Mappable that can be used for a colorbar

    :param cmap:
    :param norm:
    :param head_length:
    :param head_width:
    :return:
    """

    if cmap is None:
        cmap = 'default'

    if isinstance(cmap, str):
        cmap = plot.lookup_cmap(cmap)

    if isinstance(c, str):
        c = arrows[c]

    if norm is None:
        norm = matplotlib.colors.Normalize(vmin=c.min(), vmax=c.max())

    arrowstyle = matplotlib.patches.ArrowStyle.CurveFilledB(head_length=head_length, head_width=head_width)

    kwargs.setdefault('linewidth', .75)

    source_pos, target_pos = _clip_arrows(arrows, tail_offset, head_offset)

    for i, idx in enumerate(arrows.index):
        color = cmap(norm(c[idx]))
        _plot_single_arrow(ax, source_pos[i], target_pos[i], arrowstyle, color, **kwargs)

    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(c.values)

    return sm


def _plot_single_arrow(ax, source_pos, target_pos, arrowstyle, color, **kwargs):
    patch_kwargs = kwargs.copy()
    patch_kwargs.setdefault('edgecolor', color)
    patch_kwargs.setdefault('facecolor', color)
    patch = matplotlib.patches.FancyArrowPatch(
        posA=source_pos,
        posB=target_pos,
        arrowstyle=arrowstyle,
        **patch_kwargs,
    )
    ax.add_artist(patch)


def plot_arrows_solid(
        ax, arrows, color=None,
        tail_offset=0, head_offset=0, head_length=4, head_width=1.25, **kwargs):
    """
    Draw multiple arrows using a solid color.

    :param ax: matplotlib.axes.Axes
    :param arrows: a pd.DataFrame with columns: 'source_x', 'source_y', 'target_x', 'target_y'
    :param tail_offset: how much shorter to make the tail (so it doesn't overlap with the markers)
    :param head_offset: how much shorter to make the head (so it doesn't overlap with the markers)
    :param kwargs: args for matplotlib.patches.FancyArrowPatch

    :param color:
    :param head_length:
    :param head_width:
    :param kwargs:
    :return:
    """

    arrowstyle = matplotlib.patches.ArrowStyle.CurveFilledB(head_length=head_length, head_width=head_width)

    kwargs.setdefault('linewidth', .75)

    source_pos, target_pos = _clip_arrows(arrows, tail_offset, head_offset)

    for i, idx in enumerate(arrows.index):
        _plot_single_arrow(ax, source_pos[i], target_pos[i], arrowstyle, color, **kwargs)


class Graph:
    """
    A class to plot graphs with per-node and per-edge styles
    """
    def __init__(self, nodes, edges, styles=None, transform=None, kwargs_nodes=None, kwargs_edges=None):
        """
        :param nodes: a pd.DataFrame with columns ['x', 'y'] representing the 2d position and
            column 'style' that can be indexed into the styles DF

        :param edges: a pd.DataFrame with columns ['source', 'target'] that can be indexed into the nodes DF and
            column 'style' that can be indexed into the styles DF

        :param styles: pd.DataFrame with columns for different cmaps ('cmap_from_white', etc),
        color levels ('light', 'dark', etc). By default: plot.styles_df

        :param kwargs_nodes: default kwargs to nodes plotting

        :param kwargs_edges: default kwargs to edges plotting

        :param transform: the transform to apply to the graph. Useful when drawing an inset.
        """
        assert np.all(edges['source'] != edges['target']), 'self edges'
        assert np.all([np.issubdtype(nodes[c].dtype, np.number) for c in ['x', 'y']])

        if styles is None:
            styles = plot.styles_df.copy()

        self.styles = styles
        self.nodes = nodes
        self.edges = edges
        self.transform = transform

        self.default_kwargs_nodes = dict(
            cmap='cmap',
            marker='marker_time',
            linewidth=.5,
            facecolor='light',
            edgecolor='darker',
        )
        self.default_kwargs_nodes.update(kwargs_nodes or {})

        self.default_kwargs_edges = dict(
            cmap='cmap',
            facecolor='main',
            edgecolor='main',
        )
        self.default_kwargs_edges.update(kwargs_edges or {})

        edge_len = self.get_edge_lengths()
        too_short = np.count_nonzero(np.isclose(edge_len, 0))
        if too_short:
            logging.warning(f'{too_short}/{len(edge_len)} edges of zero length')

        # pandas complains when editing categories which is inconvenient
        if self.nodes['style'].dtype.name == 'category':
            self.nodes['style'] = self.nodes['style'].astype(str)

        if self.edges['style'].dtype.name == 'category':
            self.edges['style'] = self.edges['style'].astype(str)

    def copy(self):
        return Graph(
            nodes=self.nodes.copy(),
            edges=self.edges.copy(),
            styles=self.styles.copy(),
            transform=None if self.transform is None else self.transform.copy(),
            kwargs_nodes=self.default_kwargs_nodes.copy(),
            kwargs_edges=self.default_kwargs_edges.copy(),
        )

    def get_edge_lengths(self):
        xy0 = self.nodes.loc[self.edges['source'], ['x', 'y']].values
        xy1 = self.nodes.loc[self.edges['target'], ['x', 'y']].values
        edge_len = np.sqrt(np.sum(np.square(xy0 - xy1), axis=1))
        return pd.Series(edge_len, index=self.edges.index)

    def _get_arrows(self, selection=None):
        if selection is None:
            selection = self.edges

        if isinstance(selection, (np.ndarray, pd.Index)):
            selection = self.edges.loc[selection]

        arrows = [selection]

        for end in ['source', 'target']:
            pos = self.nodes[['x', 'y']].reindex(selection[end])
            pos.index = selection.index
            pos.columns = [end + '_' + c for c in pos.columns]
            arrows.append(pos)

        arrows = pd.concat(arrows, axis=1)

        return arrows

    def _lookup_style_kwargs(self, style, kwargs):
        kwargs = kwargs.copy()

        if 'style' in kwargs:
            specific = kwargs.pop('style')
            if style in specific:
                kwargs.update(specific[style])

        styled_kwargs = kwargs.copy()

        for k, v in kwargs.items():
            if isinstance(v, str) and v in self.styles.columns:
                styled_kwargs[k] = self.styles.loc[style, v]

        if self.transform is not None:
            styled_kwargs['transform'] = self.transform

        return styled_kwargs

    def plot_nodes_solid(self, ax, selection=None, **kwargs):
        """
        Plot all of the nodes with a flat color

        :param ax:
        :param selection: an array, index or boolean series that
        can be used on self.nodes.loc to draw a subset of the known nodes
        :param kwargs: scatter params
        :return:
        """

        final_kwargs = self.default_kwargs_nodes.copy()
        final_kwargs.update(kwargs)

        nodes_to_draw = self.nodes

        if selection is not None:
            assert isinstance(selection, (np.ndarray, pd.Index, pd.Series))
            nodes_to_draw = self.nodes.loc[selection]

        for style, nodes in nodes_to_draw.groupby('style'):

            style_kwargs = self._lookup_style_kwargs(style, final_kwargs)

            if 'cmap' in style_kwargs:
                style_kwargs.pop('cmap')

            ax.scatter(
                nodes.x,
                nodes.y,
                **style_kwargs,
            )

    def plot_nodes_cmap(self, ax, c=None, selection=None, **kwargs):
        """
        Plot all of the nodes with a color map

        :param ax:

        :param c: series or array matching length of self.nodes,
        if none indicated, we expect a column 'c' in self.nodes

        :param selection: an array, index or boolean series that
        can be used on self.nodes.loc to draw a subset of the known nodes

        :param kwargs: scatter params
        :return: a dict of style to mappable for use in colorbars
        """

        final_kwargs = self.default_kwargs_nodes.copy()
        final_kwargs.update(kwargs)

        nodes_to_draw = self.nodes

        if selection is not None:
            assert isinstance(selection, (np.ndarray, pd.Index, pd.Series))
            nodes_to_draw = self.nodes.loc[selection]

        if c is None:
            c = 'c'

        if isinstance(c, str):
            c = self.nodes[c]

        if isinstance(c, np.ndarray):
            c = pd.Series(c, index=self.nodes.index)

        all_sm = {}
        for style, nodes in nodes_to_draw.groupby('style'):

            style_kwargs = self._lookup_style_kwargs(style, final_kwargs)

            if 'facecolor' in style_kwargs:
                style_kwargs.pop('facecolor')

            all_sm[style] = ax.scatter(
                nodes.x,
                nodes.y,
                c=c.loc[nodes.index],
                **style_kwargs,
            )

        return all_sm

    def plot_nodes_labels(self, ax, nodes=None, va='center', ha='center', fmt='{index}', fontsize=6, **kwargs):
        """
        plot a descriptive text for each node.
        By default, the index is show, modify fmt to use something else
        """
        # TODO allow the style column in the fmt to color by dark of the "label" column.

        if nodes is None:
            nodes = self.nodes
        else:
            nodes = self.nodes.loc[nodes]

        for idx, row in nodes.iterrows():
            ax.text(row['x'], row['y'], fmt.format(index=idx, **row), va=va, ha=ha, fontsize=fontsize, **kwargs)

    def plot_edges_cmap(self, ax, c=None, **kwargs):
        """
        Plot all of the nodes with a color map

        :param ax:

        :param c: series or array matching length of self.edges,
        if none indicated, we expect a column 'c' in self.edges

        :param kwargs: params to plot_arrows_cmap

        :return: a dict of style to mappable for use in colorbars
        """

        final_kwargs = self.default_kwargs_edges.copy()
        final_kwargs.update(kwargs)

        if c is None:
            c = self.edges['c']

        all_sm = {}

        for style, arrows in self._get_arrows().groupby('style'):
            style_kwargs = self._lookup_style_kwargs(style, final_kwargs)

            if 'facecolor' in style_kwargs:
                style_kwargs.pop('facecolor')

            if 'edgecolor' in style_kwargs:
                style_kwargs.pop('edgecolor')

            all_sm[style] = plot_arrows_cmap(
                ax, arrows, c,
                **style_kwargs
            )

        return all_sm

    def plot_edges_solid(self, ax, selection=None, **kwargs):
        """
        Plot all of the edges with a flat color

        :param ax:
        :param selection:
        :param kwargs:
        :return:
        """

        final_kwargs = self.default_kwargs_edges.copy()
        final_kwargs.update(kwargs)

        for style, arrows in self._get_arrows(selection=selection).groupby('style'):
            style_kwargs = self._lookup_style_kwargs(style, final_kwargs)

            if 'cmap' in style_kwargs:
                style_kwargs.pop('cmap')

            plot_arrows_solid(
                ax, arrows,
                **style_kwargs
            )

    @classmethod
    def from_conns(cls, conns, cells, node_style='ei_type', edge_style='con_type'):
        """plot the connections in XY space"""

        all_gids = np.unique(conns[['source_gid', 'target_gid']].values.flatten())

        nodes = cells.loc[all_gids, ['x', 'y']].copy()
        nodes['style'] = cells.loc[nodes.index, node_style]

        edges = conns[['source_gid', 'target_gid']].copy()
        edges.columns = ['source', 'target']
        edges['style'] = conns.loc[edges.index, edge_style]

        return cls(nodes, edges)

    @classmethod
    def from_conn_jumps(
            cls, selected_jumps, detailed_spikes, node_keys, edge_style,
            **kwargs):
        """plot spike jumps"""
        assert 'x' in node_keys and 'y' in node_keys and 'style' in node_keys

        nodes = {}
        for k, v in node_keys.items():
            if isinstance(v, str):
                v = detailed_spikes[v]

            else:
                assert isinstance(v, (tuple, list, pd.Series, np.ndarray))

            nodes[k] = v

        nodes = pd.DataFrame(nodes)

        edges = selected_jumps[['source_spike', 'target_spike']].copy()
        edges.columns = ['source', 'target']

        edges['style'] = selected_jumps.loc[edges.index, edge_style]

        return cls(nodes, edges, **kwargs)

    def get_floating_nodes(self) -> pd.Index:
        """
        :return: the index of nodes with no connections in or out
        """
        return self.nodes.index[
            ~self.nodes.index.isin(self.edges['source']) &
            ~self.nodes.index.isin(self.edges['target'])
        ]

    def get_linked_nodes(self) -> pd.Index:
        """
        :return: the index of nodes with at least a connection in or out
        """
        return self.nodes.index[~self.nodes.index.isin(self.get_floating_nodes())]

    def drop_nodes(self, drop_gids: pd.Index):
        """
        remove the given nodes from the graph. This will also remove edges to/from those nodes
        :param drop_gids: either a list of node ids or a boolean mask (True == remove)
        :return:
        """

        if drop_gids.dtype == 'bool':
            if isinstance(drop_gids, pd.Series):
                drop_gids = drop_gids.reindex(self.nodes.index, fill_value=False)

            assert len(drop_gids) == len(self.nodes)
            drop_gids = self.nodes.index[drop_gids]

        drop_gids = pd.Index(np.asarray(drop_gids))
        remaining_gids = self.nodes.index.difference(drop_gids)
        self.nodes = self.nodes.loc[remaining_gids].copy()

        bad_edges = (
                self.edges['source'].isin(drop_gids) |
                self.edges['target'].isin(drop_gids)
        )
        self.edges = self.edges.loc[~bad_edges].copy()

    def drop_edges(self, drop_gids: pd.Index):
        """
        remove the given edges from the graph
        example:
            graph.drop_edges(graph.edges['weight'] < .75 * 70)

        :param drop_gids: either a list of edge ids or a boolean mask (True == remove)
        :return:
        """

        if drop_gids.dtype == 'bool':
            if isinstance(drop_gids, pd.Series):
                drop_gids = drop_gids.reindex(self.edges.index, fill_value=False)

            assert len(drop_gids) == len(self.edges)
            drop_gids = self.edges.index[drop_gids]

        drop_gids = pd.Index(np.asarray(drop_gids))
        remaining_gids = self.edges.index.difference(drop_gids)
        self.edges = self.edges.loc[remaining_gids].copy()

    def add_edges(self, new_edges: pd.DataFrame, **overwrite_cols):
        """
        Add edges to this graph.
        Inplace.

        :param overwrite_cols: pairs of <column, value> to assign to new_edges before adding them.
        For example, to set a style.
        """
        new_edges = new_edges.copy()

        for c, v in overwrite_cols.items():
            new_edges[c] = v

        missing_cols = self.edges.columns.difference(new_edges.columns)

        if len(missing_cols) > 0:
            logging.error(f'Missing columns: {list(missing_cols)}. Got: {list(new_edges.columns)}')
            return

        repeated = self.edges.index.intersection(new_edges.index)
        if len(repeated):
            logging.warning(f'Repeated edges will be ignored: {repeated}')
            new_edges = new_edges.drop(repeated)

        valid = (
                new_edges['source'].isin(self.nodes.index) &
                new_edges['target'].isin(self.nodes.index)
        )
        if np.any(~valid):
            logging.warning(f'{np.count_nonzero(~valid):,g} edges without source or target will be ignored')
            new_edges = new_edges[valid]

        all_edges = pd.concat([self.edges, new_edges], axis=0, sort=False)
        assert all_edges.index.is_unique
        self.edges = all_edges

    def add_nodes(self, new_nodes: pd.DataFrame, **overwrite_cols):
        """
        Add edges to this graph.
        Inplace.

        :param overwrite_cols: pairs of <column, value> to assign to new_nodes before adding them.
        For example, to set a style.
        """
        new_nodes = new_nodes.copy()

        for c, v in overwrite_cols.items():
            new_nodes[c] = v

        missing_cols = self.nodes.columns.difference(new_nodes.columns)

        if len(missing_cols) > 0:
            logging.warning(f'Missing columns: {list(missing_cols)}. Got: {list(new_nodes.columns)}')

        repeated = self.nodes.index.intersection(new_nodes.index)
        if len(repeated):
            logging.warning(f'Repeated nodes will be ignored: {repeated}')
            new_nodes = new_nodes.drop(repeated)

        all_nodes = pd.concat([self.nodes, new_nodes], axis=0, sort=False)
        assert all_nodes.index.is_unique
        self.nodes = all_nodes

    def add_graph(self, other):
        """
        Add another graph to this one.
        Inplace.
        """
        self.add_nodes(other.nodes)
        self.add_edges(other.edges)

    def drop_edges_orphan(self):
        """remove edges without a known source or target"""
        mask_edges = (
                self.edges['source'].isin(self.nodes.index) &
                self.edges['target'].isin(self.nodes.index)
        )
        self.edges = self.edges[mask_edges].copy()

    def layout_spring(self, edges_idx=None, iterations=100, source_gid=None, **kwargs):
        """
        modify inplace the XY positions of the graph using a spring force algorithm
        if source_gid is provided, it will be fixed at coordinate (0, 0)
        initial position are taken from the current XY.
        """

        fixed = kwargs.pop('fixed', None)

        if source_gid is not None:
            if fixed is None:
                fixed = {}
            fixed[source_gid] = (0, 0)

        from networkx import spring_layout
        pos = spring_layout(
            self._get_as_networkx_digraph(edges_idx),
            pos={i: (x, y) for i, x, y in self.nodes[['x', 'y']].itertuples()},
            fixed=fixed,
            iterations=iterations,
            **kwargs,
        )
        self._set_node_xy(pd.DataFrame.from_dict(pos, orient='index', columns=['x', 'y']))

    def layout_graphviz(self, edges_idx=None, **kwargs):
        """
        modify inplace the XY positions of the graph using a one of the graphviz algorithms
        see https://stackoverflow.com/questions/21978487/improving-python-networkx-graph-layout
        """
        from networkx.drawing.nx_agraph import graphviz_layout

        pos = graphviz_layout(
            self._get_as_networkx_digraph(edges_idx),
            **kwargs)

        self._set_node_xy(pd.DataFrame.from_dict(pos, orient='index', columns=['x', 'y']))

    def layout_raster_graphviz(self, all_spikes):
        """
        modify inplace the Y positions of the graph (preserving X)
        using the 'dot' algorithm (hierarchical)
        """
        oldx = self.nodes['x'].copy()
        self.layout_graphviz(prog='dot')
        self.layout_transpose()
        self.layout_reflect('x')

        # restore x as time
        self.nodes['x'] = oldx

        # force y to be different and unique per gid
        self.nodes['y'] = self.nodes['y'].astype(np.float)
        gids = all_spikes.loc[self.nodes.index, 'gid'].values
        yloc = self.nodes['y'].groupby(gids).median().rank(method='first').reindex(gids)
        yloc.index = self.nodes.index
        self.nodes['y'] = yloc

        assert np.all([np.issubdtype(self.nodes[c].dtype, np.number) for c in ['x', 'y']])

    def layout_best_fit(self, around, orientation='vertical'):
        """
        Place nodes using graphviz.
        For 'floating' (disconnected) nodes, force them at the bottom of the plot.
        Rotate plot to best use the orientation

        :return:
        """
        floating_gids = self.get_floating_nodes()

        self.layout_graphviz()

        center = self._layout_around(around)

        # make sure floating gids don't interfere when we are rotationg our graph
        # their position will get set afterwards
        self.nodes.loc[floating_gids, 'x'] = center[0]
        self.nodes.loc[floating_gids, 'y'] = center[1]
        self.layout_rotate_to_match(around=center, orientation=orientation)

        linked_gids = self.get_linked_nodes()

        bbox = (
            np.minimum(self.nodes.loc[linked_gids, ['x', 'y']].min(), -10),
            np.maximum(self.nodes.loc[linked_gids, ['x', 'y']].max(), +10),
        )

        x = np.linspace(bbox[0]['x'], bbox[1]['x'], len(floating_gids) + 2)[1:-1]
        self.nodes.loc[floating_gids, 'x'] = x

        y = bbox[0]['y'] - (bbox[1]['y'] - bbox[0]['y']) * .2
        self.nodes.loc[floating_gids, 'y'] = y

    def _layout_around(self, around):
        """
        translate the "around" param of other functions

        :param around:
            None: the center of mass of the graph
            tuple, list or array: the exact 2d coordinates
            anything else: the ID of the node we want to center around
        :return: array of 2 elements containing xy position
        """
        xy = self.nodes[['x', 'y']].values

        if around is None:
            around = np.mean(xy, axis=0)

        elif isinstance(around, (list, tuple, np.ndarray)):
            around = np.array(around)

        else:
            around = self.nodes.loc[around, ['x', 'y']].values.astype(np.float)

        assert np.issubdtype(around.dtype, np.number)
        return around

    def sort_edges(self, by, ascending=True):
        """sort the edges by the given series. inplace"""
        if isinstance(by, str):
            by = self.edges[by]

        if not isinstance(by, pd.Series):
            by = pd.Series(np.asarray(by), by.index)

        assert isinstance(by, pd.Series)

        by = by.reindex(self.edges.index)
        by = by.sort_values(ascending=ascending)
        self.edges = self.edges.loc[by.index]

    def layout_get_dists(self):
        """
        get the distances for every node with respect to (0, 0)
        :return:
        """
        xy = self.nodes[['x', 'y']].values.T

        dists = np.sqrt(np.sum(np.square(xy), axis=0))

        return pd.Series(dists, index=self.nodes.index)

    def layout_get_angles(self):
        """
        get the angles for every node with respect to (0, 0)
        :return:
        """

        dists = self.layout_get_dists()

        xy = self.nodes[['x', 'y']].values.T
        vectors = xy / dists.values

        angles = np.degrees(np.arctan2(vectors[1], vectors[0]))

        return pd.Series(angles, index=self.nodes.index)

    def layout_center(self, around=None):
        """
        force the graph to be centered around the given point

        inplace

        :param around: see _layout_around
        :return:
        """
        around = self._layout_around(around)

        self.nodes[['x', 'y']] = self.nodes[['x', 'y']].values - around

    def layout_transpose(self):
        """swap x and y coordinates. inplace"""
        origx = self.nodes['x'].copy()
        self.nodes['x'] = self.nodes['y'].copy()
        self.nodes['y'] = origx

    def layout_reflect(self, axis='x'):
        """reflect either x or y dimension around its center. inplace"""

        axis = {
            'x': 'x', 0: 'x',
            'y': 'y', 1: 'y',
        }[axis]

        center = np.mean(self.nodes[axis])
        self.nodes[axis] = (self.nodes[axis] - center) * -1 + center

    def layout_rotate(self, degrees, around=(0, 0)):
        """
        rotate a graph clockwise by the given amount

        inplace

        :param degrees:
        :param around: see _layout_around
        :return:
        """

        around = self._layout_around(around)

        theta = np.radians(degrees)
        c, s = np.cos(theta), np.sin(theta)

        rotation_mat = np.array([
            [c, -s],
            [s,  c],
        ])

        xy = self.nodes[['x', 'y']].values

        xy_rotated = ((xy - around) @ rotation_mat) + around

        self.nodes[['x', 'y']] = xy_rotated

    def layout_rotate_to_match(self, around=(0, 0), orientation='vertical'):
        """
        rotate this graph to best fit in a particular orientation

        inplace

        :param around: see _layout_around
        :param orientation:
        float: the direction, in degrees, to orient the graph.
        str: 'vertical' or 'horizontal'

        :return:
        """

        if isinstance(orientation, str):
            orientation = {'vertical': -90, 'horizontal': 0.}[orientation]

        self.layout_center(around=around)

        dists = self.layout_get_dists()
        angles = self.layout_get_angles()
        angle_furthest = angles[dists.idxmax()]

        self.layout_rotate(angle_furthest + orientation)

    def _get_as_networkx_digraph(self, edges_idx=None):
        """get a networkx directed graph representation of this object"""
        if edges_idx is None:
            edges_idx = slice(None)

        import networkx as nx
        nx_graph = nx.DiGraph()

        extra_cols = self.edges.loc[:, self.edges.columns.str.startswith('layout_')].copy()
        extra_cols.columns = extra_cols.columns.str.slice(len('layout_'), None)

        edges_data = [
            (s, t, extra_cols.loc[i].to_dict())
            for i, s, t in self.edges.loc[edges_idx, ['source', 'target']].itertuples()
        ]

        nx_graph.add_edges_from(edges_data)

        return nx_graph

    def _set_node_xy(self, xy):
        count_extra = np.count_nonzero(~xy.index.isin(self.nodes.index))
        if count_extra > 0:
            print(f'Got {count_extra} unexpected nodes from given edges_idx.')

        self.nodes.loc[xy.index, 'x'] = xy['x']
        self.nodes.loc[xy.index, 'y'] = xy['y']

    def style_by_node_labels(self, labels, label_colors, style_edges='target'):
        """
        given some unique labels for the nodes, create and assign styles based on different colors
        :param labels:
        :param label_colors: a dictionary from label to color
        :param style_edges: use the style of the target/source for the edges or None for no re-styling.
        :return:
        """
        if not isinstance(labels, pd.Series):
            assert len(labels) == len(self.nodes)
            labels = pd.Series(np.asarray(labels), index=self.nodes.index)

        labels = labels.reindex(self.nodes.index)

        for label, group_gids in labels.groupby(labels).groups.items():
            existing_style = self.nodes.loc[group_gids, 'style']

            label_color = label_colors[label]

            for old_style_name, gids in existing_style.groupby(existing_style).groups.items():

                new_style_name = f'{old_style_name}-{label}'

                if old_style_name in self.styles.index:
                    base_style = old_style_name
                else:
                    base_style = 'default'

                self.styles.loc[new_style_name] = plot.style_mix(
                    base_style, self.styles,
                    lighter=plot.resaturate_color(plot.lighten_color(label_color), .25),
                    light=plot.resaturate_color(plot.lighten_color(label_color), .5),
                    main=label_color,
                    dark=plot.lighten_color(label_color, 1.125),
                    darker=plot.lighten_color(label_color, 1.25),
                )

                self.nodes.loc[gids, 'style'] = new_style_name

                if style_edges is not None:
                    edges_sel = self.edges[style_edges].isin(gids)
                    self.edges.loc[edges_sel, 'style'] = new_style_name

    def style_by_edge_labels(self, labels, label_colors: pd.Series):
        """
        given some unique labels for each edge, create and assign styles based on different colors
        :param labels:
        :param label_colors: a dictionary from label to color
        :return:
        """
        labels = labels.reindex(self.edges.index)

        for label, group_gids in labels.groupby(labels).groups.items():
            existing_style = self.edges.loc[group_gids, 'style']

            label_color = label_colors[label]

            for old_style_name, gids in existing_style.groupby(existing_style).groups.items():

                new_style_name = f'{old_style_name}-{label}'

                spec = dict(
                    lighter=plot.resaturate_color(plot.lighten_color(label_color), .25),
                    light=plot.resaturate_color(plot.lighten_color(label_color), .5),
                    main=label_color,
                    dark=plot.lighten_color(label_color, 1.125),
                    darker=plot.lighten_color(label_color, 1.25),
                )

                self.styles.loc[new_style_name] = plot.style_mix(old_style_name, self.styles, **spec)
                self.edges.loc[gids, 'style'] = new_style_name

    def style_bkg_nodes(self, bkg_nodes, value='lighter'):
        """force all of the given nodes to be colored on 'lighter'"""
        styles = self.nodes.loc[bkg_nodes, 'style'].astype(str)
        self.nodes.loc[bkg_nodes, 'style'] = styles.map(lambda x: f'{x}-bkg')

        for name in styles.unique():
            self.styles.loc[f'{name}-bkg'] = plot.style_mix(
                name, base_styles=self.styles, light=value, main=value, dark=value, darker=value)

    def style_selected_gids(self, all_spikes, which_gids, style_name):
        """force all of the given nodes to be colored on 'lighter'"""

        which_gids = np.asarray(which_gids)
        if np.ndim(which_gids) == 0:
            which_gids = [which_gids.item()]

        gids = all_spikes.loc[self.nodes.index, 'gid']
        self.nodes.loc[gids.isin(which_gids), 'style'] = style_name
