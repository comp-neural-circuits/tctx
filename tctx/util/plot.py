import numpy as np
import pandas as pd
import numba

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as LSCMap
import matplotlib.ticker
import matplotlib.cm
import matplotlib.colors
import matplotlib.transforms
import matplotlib.patches
import matplotlib.gridspec
import matplotlib.figure
import matplotlib.patheffects
import colorsys
import json
import os.path
import collections
import matplotlib.path


# noinspection PyUnusedLocal
def default_pbar(iterable, total=None, desc=None):
    return iterable


##################################################################################################
# Styles


def resample_colormap(cmap, vmin=0, vmax=1, n=256):
    """clip a colormap
    This reduces the dynamic range but it is useful to get rid of yellow values at the end of
    viridis/plasma/magma/inferno, which can't be seen well on a screen or on projectors


    call like:
        cmap = plot.resample_colormap('magma', vmax=.9)
    """
    if isinstance(cmap, str):
        cmap = matplotlib.cm.get_cmap(cmap, n)

    return matplotlib.colors.ListedColormap(cmap(np.linspace(vmin, vmax, n)))


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """

    c = color
    if not isinstance(c, np.ndarray) and c in matplotlib.colors.cnames:
        c = matplotlib.colors.cnames[color]

    c = colorsys.rgb_to_hls(*matplotlib.colors.to_rgb(c))

    new_color = colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

    return tuple(np.maximum(0, new_color))


def resaturate_color(color, amount=0.5):
    """
    Saturates the given color by setting saturation to the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    """

    if not isinstance(color, np.ndarray) and color in matplotlib.colors.cnames:
        color = matplotlib.colors.cnames[color]

    hls = colorsys.rgb_to_hls(*matplotlib.colors.to_rgb(color))

    new_hls = hls[0], hls[1], amount

    new_color = colorsys.hls_to_rgb(*new_hls)

    return tuple(np.minimum(np.maximum(0, new_color), 1))


connectivity_inh_cmap = LSCMap(
    name='Connectivity',
    N=2048,
    segmentdata={
        'red': ((0.0, 1.0, 1.0),
                (1.0, .5, 1.0)),

        'green': ((0.0, 1.0, .75),
                  (1.0, 0.0, 0.0)),

        'blue': ((0.0, 1.0, .75),
                 (1.0, 0.0, 0.0))})

connectivity_exc_cmap = LSCMap(
    name='Connectivity',
    N=2048,
    segmentdata={
        'blue': ((0.0, 1.0, 1.0),
                 (1.0, .5, 1.0)),

        'green': ((0.0, 1.0, .75),
                  (1.0, 0.0, 0.0)),

        'red': ((0.0, 1.0, .75),
                (1.0, 0.0, 0.0))})


cmap_greys = LSCMap(
    name='Connectivity',
    N=2048,
    segmentdata={
        'blue': ((0.0, 1.0, .75),
                 (1.0, 0., 1.0)),

        'green': ((0.0, 1.0, .75),
                  (1.0, 0.0, 0.0)),

        'red': ((0.0, 1.0, .75),
                (1.0, 0.0, 0.0))})


WEIGHT_CMAP = LSCMap.from_list(
    'weight',
    ['xkcd:cerulean', 'xkcd:dark mustard', 'xkcd:orange'],
)


group_style = collections.namedtuple('group_style', 'background highlight highlight2 cmap short_name long_name')


exc_style = group_style(
    background='xkcd:faded blue',
    highlight='xkcd:ocean blue',
    highlight2='xkcd:cerulean',
    cmap=connectivity_exc_cmap,
    short_name='exc',
    long_name='excitatory'
)


inh_style = group_style(
    background='xkcd:faded pink',
    highlight='xkcd:carmine',
    highlight2='xkcd:scarlet',
    cmap=connectivity_inh_cmap,
    short_name='inh',
    long_name='inhibitory'
)

combined_style = group_style(
    background='xkcd:grey',
    highlight='xkcd:charcoal',
    highlight2='xkcd:dark grey',
    cmap=cmap_greys,
    short_name='all',
    long_name='all'
)


def custom_marker_rect(width, height):
    """
    Use a rectangle with fill and edge in your scatter plots.
    In normalized units.  The overall marker will be rescaled by *s* in the scatter call.
    Result can be used like:
    ax.scatter(x, y, marker=custom_marker_rect(1, 2)
    """
    half_width = width * .5
    half_height = height * .5

    marker = np.array([
        [-half_width, -half_height],
        [-half_width, half_height],
        [half_width, half_height],
        [half_width, -half_height],
        [-half_width, -half_height],
    ])

    return marker


def custom_marker_pipette(length=5., width=2., stem_in=.25, stem_out=.1, theta_deg=30.):
    """
        Create a path that can represent a pipette. Use as a marker in scatter plot:

            f, ax = plt.subplots()
            for theta in np.linspace(0, -360, 5 + 1)[:-1]:
                ax.scatter(
                    [0],
                    [0],
                    s=500,
                    marker=pipette_marker(theta_deg=theta),
                    facecolor='xkcd:light purple',
                    edgecolor='xkcd:dark purple',
                )
    """
    base = np.array([
        [-.5, 1],
        [0, 0],
        [.5, 1],
        [0, 1 - stem_in],
        [0, 1 + stem_out],
    ])

    theta = np.deg2rad(theta_deg)

    scale = np.array([width, length])

    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])

    marker = base * scale @ rotation

    codes = [
        matplotlib.path.Path.MOVETO,
        matplotlib.path.Path.LINETO,
        matplotlib.path.Path.LINETO,
        matplotlib.path.Path.MOVETO,
        matplotlib.path.Path.LINETO,
    ]

    return matplotlib.path.Path(marker, codes=codes)


def plot_pipette(ax, x, y, theta_deg=-30, **kwargs):
    """plot a single pipette marker"""
    default = dict(
        s=200,
        linewidth=.5,
        facecolor=styles_df.loc['purple', 'light'],
        edgecolor=styles_df.loc['purple', 'darker'],
    )

    for k, v in default.items():
        kwargs.setdefault(k, v)

    ax.scatter(
        x, y,
        marker=custom_marker_pipette(theta_deg=theta_deg),
        **kwargs,
    )


def custom_marker_spike(width=.5, height=2):
    """
    Use a rectangle with fill and edge in your scatter plots.
    In normalized units.  The overall marker will be rescaled by *s* in the scatter call.
    Result can be used like:
    ax.scatter(x, y, marker=custom_marker_rect(1, 2)
    """
    half_width = width * .5
    half_height = height * .5

    marker = np.array([
        [-half_width, -half_height],
        [-half_width, half_height],
        [half_width, half_height],
        [half_width, -half_height],
        [-half_width, -half_height],
    ])

    codes = [
        matplotlib.path.Path.MOVETO,
        matplotlib.path.Path.LINETO,
        matplotlib.path.Path.LINETO,
        matplotlib.path.Path.LINETO,
        matplotlib.path.Path.CLOSEPOLY,
    ]

    # noinspection PyTypeChecker
    return matplotlib.path.Path(marker, codes=codes)


SPIKE_MARKER = custom_marker_spike()


def load_styles_df():
    """
    Load styles and generate aliases for schemes and color maps.

    :return: a pd.DataFrame that looks like

            scheme marker     dark   darker    light  lighter     main  cmap cmap_from_white highlight background
        name
        e      blue      ^  #154975  #053053  #4E7EA6  #8EB3D1  #2D6390  obj       obj        #4E7EA6    #154975
        e2e    blue      ^  #154975  #053053  #4E7EA6  #8EB3D1  #2D6390  obj       obj        #4E7EA6    #154975
        red     red      o  #A3143D  #740021  #E66389  #F29FB7  #C9355F  obj       obj        #E66389    #A3143D

    """
    colors_filepath = os.path.abspath(os.path.join(__file__, '../../../data/processed/colors.json'))

    with open(colors_filepath, 'r') as fd:
        colors = json.load(fd)

    schemes = pd.DataFrame(colors).T  # index=ids, cols=properties (dark, main, cmap, ...)

    schemes.columns.name = 'hue'
    schemes.index.name = 'scheme'

    cmaps = {}

    for name, hues in schemes.iterrows():
        sorted_keys = ['lighter', 'light', 'main', 'dark', 'darker']

        color_list = [hues[k] for k in sorted_keys]

        cmaps.setdefault('cmap', {})[name] = LSCMap.from_list(str(name), color_list)
        cmaps.setdefault('cmap_r', {})[name] = LSCMap.from_list(str(name), color_list[::-1])

        for i in range(len(sorted_keys)):
            color_list_with_white = ['white'] + color_list[:i + 1]
            cmap_name = 'cmap_white_to_' + sorted_keys[i]

            cmaps.setdefault(cmap_name, {})[name] = LSCMap.from_list(
                cmap_name, color_list_with_white)

            cmaps.setdefault(cmap_name + '_r', {})[name] = LSCMap.from_list(
                cmap_name, color_list_with_white[::-1])

        cmaps.setdefault('cmap_from_white', {})[name] = cmaps['cmap_white_to_main'][name]

    for k, cm in cmaps.items():
        schemes[k] = pd.Series(cm)

    schemes['highlight'] = schemes['light']
    schemes['background'] = schemes['dark']

    mapping = {
        'blue': ['e', 'e2e', 'e2x', 'e-e', 'e2e2e'],
        'red': ['i', 'i2e', 'i2x', 'i-e'],
        'yellow': ['i2i', 'i-i', 'e2i2i'],
        'green': ['e2i', 'e-i', '?', 'unknown', 'e2i2e'],
        'grey': ['combined', 'all', 'default', 'total'],
        'cerulean': ['weak', 'all_weak_below_10'],
        'mustard': ['mid', 'only_mid'],
        'orange': ['strong', 'only_strong_top_quarter'],
        'purple': ['induced', 'original', 'full'],
    }

    mapping_df = {}
    for source in schemes.index:
        mapping_df[source] = source
        for alias in mapping.get(source, []):
            mapping_df[alias] = source

    mapping_df = pd.DataFrame({'scheme': pd.Series(mapping_df)})
    mapping_df.index.name = 'name'
    mapping_df = mapping_df.join(schemes, on='scheme', how='outer')

    markers = pd.DataFrame({'marker': pd.Series({
        'e': '^', 'e2e': '^', 'e-e': '^',
        'e2i': 'D', 'e-i': 'D',
        'i': 'o', 'i2e': 'o', 'i-e': 'o',
        'i2i': 's', 'i-i': 's',
        'induced': '^',
    })}, index=mapping_df.index)

    markers['marker_space'] = markers['marker']
    markers['marker_time'] = [SPIKE_MARKER] * len(markers)
    markers.fillna('o', inplace=True)

    mapping_df = mapping_df.join(markers, on='name')

    return mapping_df


styles_df = load_styles_df()


def lookup_style(style='default') -> pd.Series:
    """find a style by name"""

    if style is None:
        style = 'default'

    if isinstance(style, str):
        style = styles_df.loc[style]

    return style


def style_mix(base_name, base_styles=styles_df, **alts):
    """An easy way to mix styles in a one liner.
    Alts will map an attribute name to the new value. This can be (in order of preference):
        - the same attribute from a different style (if it exists)
        - a different attribute from a the same style (if it exists)
        - a  specific value.

    examples:
        # a copy of the 'green' style with the 'marker_space' value from 'e'
        style_mix('green', marker_space='e')

        # a copy of the 'green' style with the 'marker_space' value equal to 'marker_time'
        style_mix('green', marker_space='marker_time')

        # a copy of the 'green' style with the 'marker_space' value with a specific value
        style_mix('green', marker_space='|')
    """
    new_style = base_styles.loc[base_name].copy()

    for name, alt in alts.items():

        if not isinstance(alt, np.ndarray) and alt in base_styles.index:
            new_style[name] = base_styles.loc[alt, name]

        elif not isinstance(alt, np.ndarray) and alt in base_styles.columns:
            new_style[name] = base_styles.loc[base_name, alt]

        else:
            new_style[name] = alt

    return new_style


ALL_LABEL_ALIAS = {
    'input_whitenoise_mean': (r'$\mu_{in}$', 'pA'),
    'input_whitenoise_std': (r'$\sigma_{in}$', 'pA'),
    'input_rate': ('$FR_{in}$', 'Hz'),
    'input_weight': ('$w_{in}$', 'nS'),
    'distance': ('distance', r'$\mu m$'),
    'x': ('x', r'$\mu m$'),
    'y': ('y', r'$\mu m$'),
    'z': ('z', r'$\mu m$'),
    'xbin': (r'$\mu m$', ''),
    'xbins': (r'$\mu m$', ''),
    'tbin': (r'$ms$', ''),
    'tbins': (r'$ms$', ''),
    'median_first_spike': ('median first spike', 'ms'),
    'std_first_spike': ('std first spike', 'ms'),
    'std_spike': ('std spike', 'ms'),
    'IQR_spike': ('IQR spike', 'ms'),
    'IQR_first_spike': ('IQR first spike', 'ms'),
    'first_spike': ('first spike', 'ms'),
    'median_last_spike': ('median last spike', 'ms'),
    'last_spike': ('last spike', 'ms'),
    'mean_spike': ('mean spike', 'ms'),
    'median_spike': ('median spike', 'ms'),
    'fr_diff': (r'$\Delta FR$', 'Hz'),
    'frm': (r'$\Delta FR$', 'Hz'),
    'frm_norm': (r'norm $\Delta FR$', 'Hz'),
    'source_frm': (r'source $\Delta FR$', 'Hz'),
    'target_frm': (r'target $\Delta FR$', 'Hz'),
    'Afr': (r'$\Delta FR$', 'Hz'),
    'delay_std': ('delay std', 'ms'),
    'delstd': ('delay std', 'ms'),
    'delstd_norm': ('norm jitter', 'ms'),
    'source_delstd': ('source jitter', 'ms'),
    'target_delstd': ('target jitter', 'ms'),
    'delay': ('delay', 'ms'),
    'delay_in_window': ('delay', 'ms'),
    'delay_from_induced': ('delay', 'ms'),
    'target_delay_from_induced': ('target delay', 'ms'),
    'source_delay_from_induced': ('source delay', 'ms'),
    'weight': ('weight', 'nS'),
    'level_mean_mean': ('mean # syn. jumps', ''),
    'cell_hz': ('mean FR/cell', 'Hz'),
    ('cell_hz', 'total'): ('mean FR/cell', 'Hz'),
    'cell_hz_total': ('mean FR/cell', 'Hz'),
    ('cell_hz', 'e'): ('mean FR/e-cell', 'Hz'),
    'cell_hz_e': ('mean FR/e-cell', 'Hz'),
    ('cell_hz', 'i'): ('mean FR/i-cell', 'Hz'),
    'cell_hz_i': ('mean FR/i-cell', 'Hz'),

    'cell_hz_pre_total': ('mean FR/cell (pre)', 'Hz'),
    'cell_hz_pre_e': ('mean FR/e-cell (pre)', 'Hz'),
    'cell_hz_pre_i': ('mean FR/i-cell (pre)', 'Hz'),
    'spikes_pre_total': ('#spikes (pre)', ''),
    'spikes_pre_e': ('#e-spikes (pre)', ''),
    'spikes_pre_i': ('#i-spikes (pre)', ''),

    'cell_hz_induction_total': ('mean FR/cell', 'Hz'),
    'cell_hz_induction_e': ('mean FR/e-cell', 'Hz'),
    'cell_hz_induction_i': ('mean FR/i-cell', 'Hz'),
    'spikes_induction_total': ('# spikes', ''),
    'spikes_induction_e': ('# e-spikes', ''),
    'spikes_induction_i': ('# i-spikes', ''),

    'cell_hz_baseline_total': ('baseline mean FR/cell', 'Hz'),
    'cell_hz_baseline_e': ('baseline mean FR/e-cell', 'Hz'),
    'cell_hz_baseline_i': ('baseline mean FR/i-cell', 'Hz'),
    'spikes_baseline_total': ('# baseline spikes', ''),
    'spikes_baseline_e': ('# baseline e-spikes', ''),
    'spikes_baseline_i': ('# baseline i-spikes', ''),

    'cell_hz_effect_total': ('effect mean FR/cell', 'Hz'),
    'cell_hz_effect_e': ('effect mean FR/e-cell', 'Hz'),
    'cell_hz_effect_i': ('effect mean FR/i-cell', 'Hz'),
    'spikes_effect_total': ('# effect spikes', ''),
    'spikes_effect_e': ('# effect e-spikes', ''),
    'spikes_effect_i': ('# effect i-spikes', ''),

    'cell_hz_post_total': ('mean FR/cell (post)', 'Hz'),
    'cell_hz_post_e': ('mean FR/e-cell (post)', 'Hz'),
    'cell_hz_post_i': ('mean FR/i-cell (post)', 'Hz'),
    'spikes_post_total': ('#spikes (post)', ''),
    'spikes_post_e': ('#e-spikes (post)', ''),
    'spikes_post_i': ('#i-spikes (post)', ''),

    ('spikes', 'total'): ('# spikes', ''),
    'spikes_total': ('# spikes', ''),
    ('spikes', 'e'): ('# e-spikes', ''),
    'spikes_e': ('# e-spikes', ''),
    ('spikes', 'i'): ('# i-spikes', ''),
    'spikes_i': ('# i-spikes', ''),
    'all_weak_below_10': ('all-weak', ''),
    'only_strong_top_quarter': ('only-strong', ''),
    'only_mid': ('only-mid', ''),
    'time_change': (r'$\Delta t$', 'ms'),
    'delstd_norm_change': (r'$\Delta$ norm. delay std', 'ms'),
    'frm_norm_change': (r'$\Delta$ norm. $\Delta FR$', 'ms'),
    'original': ('full', ''),
    'foll_count': ('foll. count', '#cells'),
    'mean_foll_jitter': ('mean jitter', 'ms'),
    'last_foll_activation_time': ('last foll. act.', 'ms'),
    'furthest_follower_distance': ('farthest foll.', 'um'),
    'toffset': (r'$\Delta$ t', 'ms'),
}


def label_alias(name, unit=True):
    if isinstance(name, (pd.Series, pd.DataFrame)):
        name = name.name

    if name is None:
        alias = '???', ''

    elif name in ALL_LABEL_ALIAS:
        alias = ALL_LABEL_ALIAS[name]

    else:
        if isinstance(name, (tuple, list)):
            return [label_alias(n, unit=unit) for n in name]

        else:
            assert isinstance(name, str)

            alias = name.replace('_', ' '), ''

    full_name = alias[0]

    if unit and alias[1]:
        full_name = f'{full_name} ({alias[1]})'

    return full_name


def label_alias_unit(name):
    if name in ALL_LABEL_ALIAS:
        alias = ALL_LABEL_ALIAS[name]
        return alias[1]
    else:
        return '???'


##################################################################################################
# Handy


def set_figure_size_pixels(figure, x, y):
    figure.set_size_inches(x / figure.dpi, y / figure.dpi)


def set_window_title(figure, title):
    figure.canvas.set_window_title(title.replace('\n', ' '))


def remove_spines(ax):
    for sp in ax.spines.values():
        sp.set_visible(False)


def simplify_plot_1axis(ax, grid=False):

    remove_spines(ax)
    ax.spines["bottom"].set_visible(True)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.tick_params(axis="both", which="both",
                   bottom=True, labelbottom=True,
                   left=False, labelleft=True,
                   top=False, labeltop=False,
                   right=False, labelright=False)
    if grid:
        ax.set_axisbelow(True)
        ax.grid(True, axis='y', color='xkcd:light grey', linestyle='--')

    else:
        ax.grid(False)


def simplify_plot_2axis(ax, grid=False):

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.tick_params(axis="both", which="both",
                   bottom=True, labelbottom=True,
                   left=True, labelleft=True,
                   top=False, labeltop=False,
                   right=False, labelright=False)

    if grid:
        ax.set_axisbelow(True)
        ax.grid(True, axis='y', color='xkcd:light grey', linestyle='--')

    else:
        ax.grid(False)


def crop_spine(ax, name='x', opposite=False):
    ticks, spine_name, lim = {
        'x': (ax.get_xticks(), 'bottom' if not opposite else 'top', ax.get_xlim()),
        'y': (ax.get_yticks(), 'left' if not opposite else 'right', ax.get_ylim()),
    }[name]

    ticks = np.sort(ticks)
    first = ticks[lim[0] <= ticks]
    last = ticks[lim[1] >= ticks]
    if len(first) > 0 and len(last) > 0:
        first = first[0]
        last = last[-1]
        ax.spines[spine_name].set_bounds(first, last)


##################################################################################################
# Full plotting


named_marks = {
    'min': np.min,
    'max': np.max,
    'average': np.average,
    'median': np.median,
    'strongest': lambda x: x[np.argmax(np.abs(x))],
}


def simple_hist(data, name, bins=None, unit='', marks=('average',), ax=None,
                colors=('xkcd:faded blue', 'xkcd:ocean blue'), ylabel='count', norm=False, subtitle=None,
                mark_fmt='.2f'):

    bins = 10 if bins is None else bins

    heights, bins = np.histogram(data, bins=bins)
    # height = height / float(len(ratios))
    bin_widths = bins[1:] - bins[:-1]

    txt = ['n={}'.format(len(data))]

    if len(np.unique(bin_widths.round(decimals=6))) == 1:
        txt.append('bin width={:.2f}'.format(bin_widths[0]))

    title = name + '{}\n({})'.format((' ({0})'.format(unit) if unit else ''),
                                     ', '.join(txt))

    if subtitle is not None:
        title += '\n' + subtitle

    if ax is None:
        figure, ax = plt.subplots(tight_layout=True)
        set_window_title(figure, title)

    if norm:
        ylabel = 'normalized ' + ylabel
        heights = heights / np.sum(heights)

    if isinstance(colors, group_style):
        colors = (colors.background, colors.highlight)

    ax.bar(bins[:-1], heights, bins[1:] - bins[:-1], align='edge', color=colors[0])
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if isinstance(marks, str):
        marks = (marks,)

    if marks is None:
        marks = tuple()

    for name in marks:
        v = named_marks[name](data)
        ax.axvline(v, linestyle='--', color=colors[1],
                   label=('{0}={1:' + mark_fmt + '}{2}').format(name, v, (' ({0})'.format(unit) if unit else '')))

    if marks:
        ax.legend()

    simplify_plot_1axis(ax)

    return ax


def plot_traces(ax, t, traces, cmap, ylim=None, **kwargs):
    trace_count = len(traces)
    for trace, c in zip(traces, np.linspace(0, 1, trace_count)):
        ax.plot(t, trace, color=cmap(c), **kwargs)

    if ylim is not None:
        ax.set_ylim(*ylim)


def traces(
        ax, lines, x=None, c=None, cmap='viridis', norm=None, pbar=default_pbar, sort_ascending=None,
        orientation='horizontal',
        max_sample=None,
        labels=None,
        **kwargs
):
    """
        lines is expected to contain the lines as COLUMNS
        lines will be plotted in the order specified by c
        By default, c is linspace(0, 1) following the order of the columns in lines

        :param max_sample: int. Indicate how many lines to show at maximum. They will be selected randomly.
            If not provided, all lines are plot (default).
    """

    assert orientation in ('horizontal', 'vertical')

    if isinstance(lines, np.ndarray):
        lines = pd.DataFrame(lines.T)

    if c is None:

        cols = lines.columns
        # noinspection PyTypeChecker
        if isinstance(cols, pd.IntervalIndex):
            # noinspection PyUnresolvedReferences
            cols = lines.columns.mid

        if np.issubdtype(cols.dtype, np.number):
            c = cols

        else:
            c = np.linspace(0, 1, lines.shape[1])

    if isinstance(c, (np.ndarray, pd.Index)):
        c = pd.Series(c, index=lines.columns)

    if labels is not None:
        if isinstance(labels, (np.ndarray, list, tuple)):
            labels = pd.Series(labels, index=lines.columns)

    # noinspection PyUnresolvedReferences
    c = c.loc[c.index.intersection(lines.columns)]

    if x is None and isinstance(lines, pd.DataFrame):
        x = lines.index

    # noinspection PyTypeChecker
    if isinstance(x, pd.IntervalIndex):
        # noinspection PyUnresolvedReferences
        x = x.mid

    if isinstance(x, pd.TimedeltaIndex):
        x = x.values

    cmap = lookup_cmap(cmap)

    if norm is None:
        # cv = c.values
        # diff = np.diff(cv) * .5
        # boundaries = np.concatenate([[cv[0]], cv]) + np.concatenate([[-diff[0]], diff, [diff[-1]]])
        # norm = matplotlib.colors.BoundaryNorm(boundaries, len(cv))
        norm = matplotlib.colors.Normalize(vmin=c.min(), vmax=c.max())

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(c.values)

    if sort_ascending is not None:
        c = c.sort_values(ascending=sort_ascending)
        lines = lines[c.index]

    if max_sample is not None:
        c = c.sample(min(max_sample, len(c)), replace=False)

    label = kwargs.pop('label', None)

    for i, (col, trace_c) in enumerate(pbar(c.items(), total=len(c))):

        if labels is not None:
            label = labels.loc[col]

        if orientation == 'horizontal':
            ax.plot(x, lines[col].values, color=sm.cmap(sm.norm(trace_c)), label=label, **kwargs)
        else:
            ax.plot(lines[col].values, x, color=sm.cmap(sm.norm(trace_c)), label=label, **kwargs)

    return sm


def lookup_cmap(cmap):
    """return a cmap object, possibly identified by a string, prioritising custom styles"""

    if isinstance(cmap, str):
        if cmap in styles_df.index:
            cmap = styles_df.loc[cmap, 'cmap']
        else:
            cmap = matplotlib.cm.get_cmap(cmap)

    return cmap


def ymark(ax, v, label, color, axis='y', **kwargs):
    func = {'x': ax.axvline, 'y': ax.axhline}[axis]

    full_label = None
    if label is not None:
        if isinstance(label, str):
            full_label = label
        else:
            full_label = '{1} = {0:.3f} {2}'.format(v, *label)

    kwargs.setdefault('linestyle', '--')
    func(v, label=full_label, color=color, **kwargs)


def ymark_inline(ax, v, label, color, xalign='left', text_yoffset=None, text_format='{label} = {value:g}', **kwargs):
    line_kwargs = {}
    for k in ['linewidth', 'linestyle']:
        if k in kwargs:
            line_kwargs[k] = kwargs.pop(k)

    ymark(ax, v, None, color, **line_kwargs)
    ylim = ax.get_ylim()
    vrange = np.max(ylim) - np.min(ylim)

    if text_yoffset is None:
        text_yoffset = vrange * .025

    kwargs.setdefault('fontsize', 14)

    trans = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)

    if xalign == 'right':
        x = 1
        kwargs.setdefault('horizontalalignment', 'right')

    elif xalign == 'right_out':
        x = 1
        kwargs.setdefault('horizontalalignment', 'left')
        kwargs.setdefault('clip_on', False)

    else:
        assert xalign == 'left'
        x = 0
        kwargs.setdefault('horizontalalignment', 'left')

    ax.text(x, v + text_yoffset, text_format.format(label=label, value=v), color=color, transform=trans, **kwargs)


def no_ticks(ax):
    ax.tick_params(axis='both',
                   bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labeltop=False, labelleft=False, labelright=False)


def adjust_grid_with_cbar(ax_grid, cax, cbar_wspace=.01, cbar_right=.13, cbar_width=.02, **kwargs):

    cbar_total_width = (cbar_wspace + cbar_right + cbar_width)
    figure = ax_grid.flatten()[0].figure
    figure.subplots_adjust(right=1 - cbar_total_width, **kwargs)

    # Bounds are: (x0, y0, width, height)
    lowest = min([ax.get_position().bounds[1] for ax in ax_grid.flatten()])
    highest = max([ax.get_position().bounds[1] + ax.get_position().bounds[3] for ax in ax_grid.flatten()])

    cax.set_position([
        1 - cbar_total_width + cbar_wspace,  # left
        lowest,  # bottom
        cbar_width,  # width
        highest - lowest,  # height
    ])


def set_cbar_fontsize(cbar, label_size=6, tick_size=6):
    # noinspection PyProtectedMember
    cbar.set_label(cbar._label, size=label_size)
    cbar.ax.tick_params(labelsize=tick_size)


def plot_raster_density_2d(
        ax, x, y, bins, blur_std=None, sort_ascending=None,
        density=False,
        **kwargs
):
    """plot a scatter where the color is automatically generated by the local density of the points"""

    h, xedges, yedges = np.histogram2d(x, y, bins, density=density)

    if blur_std is not None:
        import scipy.ndimage.filters
        h = scipy.ndimage.filters.gaussian_filter(h, blur_std)

    xidcs = np.digitize(x, xedges) - 1
    yidcs = np.digitize(y, yedges) - 1

    xidcs[xidcs >= len(xedges) - 1] = len(xedges) - 2
    yidcs[yidcs >= len(yedges) - 1] = len(yedges) - 2

    c = h[(xidcs, yidcs)]

    if sort_ascending is not None:
        sorting = np.argsort(c)
        if not sort_ascending:
            sorting = sorting[::-1]
        c = c[sorting]
        x = np.asarray(x)[sorting]
        y = np.asarray(y)[sorting]

    default_kwargs = dict(
        edgecolor='none',
        s=10,
    )
    default_kwargs.update(kwargs)

    s = ax.scatter(
        x=x, y=y, c=c,
        **default_kwargs,
    )

    return s


def random_map_values(values):
    """
    return a map from values to the same values, where the unique values have been shuffled

    for example, if you want to colour by some kind of ID that is sequential and you want to use a
    colormap, which are nice to work with, but you don't want the colors to be sequential. This
    will give you a shuffling map that keeps the group identity of every value.

    :param values: a pd.Series or np.array
    :return: pd.Series widh index and data being unique entries of "values"
    """
    unique_values = np.unique(values)
    shuffled_unique = unique_values.copy()
    np.random.shuffle(shuffled_unique)
    shuffled_unique = pd.Series(index=unique_values, data=shuffled_unique)

    return shuffled_unique


def axs_grid_hide_bottom_spine(axs_grid, crop_y=True):
    """hide bottom spine of all subplots except the one at the bottom of the grid"""

    for ax in axs_grid[:-1, :].flatten():
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(labelbottom=False, bottom=False, labeltop=False, top=False)

    for ax in axs_grid[-1, :].flatten():
        ax.tick_params(labelbottom=True, bottom=True, labeltop=False, top=False)

    if crop_y:
        for ax in axs_grid.flatten():
            crop_spine(ax, 'y')


def axs_grid_hide_left_spine(axs_grid, crop_x=True):
    """hide bottom spine of all subplots except the one at the bottom of the grid"""

    for ax in axs_grid[:, 1:].flatten():
        ax.spines['left'].set_visible(False)
        ax.tick_params(labelleft=False, left=False, labelright=False, right=False)

    for ax in axs_grid[:, 0].flatten():
        ax.tick_params(labelleft=True, left=True, labelright=False, right=False)

    if crop_x:
        for ax in axs_grid.flatten():
            crop_spine(ax, 'x')


def axs_grid_set_labels(axs_grid, xlabel=None, ylabel=None, title=None, **kwargs):
    """
    set x and y labels on subplots on the edges of the grid

    :param axs_grid: a 2D axs grid
    :param xlabel: a string (will be repeated) or a list (one por element)
    :param ylabel: a string (will be repeated) or a list (one por element)
    :param title: a string (will be repeated) or a list (one por element)
    :return:
    """

    structure = [
        ('set_xlabel', xlabel, axs_grid[-1, :]),
        ('set_ylabel', ylabel, axs_grid[:, 0]),
        ('set_title', title, axs_grid[0, :]),
    ]

    for key, values, axs_line in structure:

        if values is not None:
            if not isinstance(values, (list, tuple, np.ndarray)):
                values = [values] * len(axs_line)

            values = [label_alias(v) for v in values]

            for val, ax in zip(values, axs_line):
                getattr(ax, key)(val, **kwargs)


def _generate_scatter_jitter(scat, blur=5, bins=100, std_scale=3., std_min=.1, std_max=np.inf):
    import scipy.ndimage

    scat = np.asarray(scat)

    h, edges = np.histogram(scat[~np.isnan(scat)], bins=bins)

    bin_idcs = np.clip((np.digitize(scat, edges) - 1), 0, len(edges) - 2)
    stds = scipy.ndimage.gaussian_filter(h, blur)
    stds = (stds / np.max(stds)) / std_scale
    stds = stds[bin_idcs]
    stds = np.clip(stds, std_min, std_max)
    dot_noise = np.random.normal(size=len(bin_idcs), loc=0, scale=stds)

    return np.clip(dot_noise, -1, 1)


def plot_scatter_n_box(
        ax, center, values, c=None, q1=.25, q2=.5, q3=.75, width=.8, style='default',
        s=3, alpha=1, edgecolor='none', zorder=1000,
        noise_scale=10,
        rect_facecolor='light',
        rect_edgecolor='light',
        median_color='darker',
        facecolor='dark',
        noise_hist=True,
        **scat_kwargs):
    """
    Plot a single scatter of values around a center, with a box behind indicating quantiles of the data.
    """
    plot_box(
        ax, center, values,
        q1=q1, q2=q2, q3=q3,
        width=width, style=style, zorder=zorder,
        rect_facecolor=rect_facecolor,
        rect_edgecolor=rect_edgecolor,
        median_color=median_color,
    )

    plot_scatter_jittered(
        ax, center, values, c=c, width=width, style=style,
        s=s, alpha=alpha, edgecolor=edgecolor, zorder=zorder,
        noise_scale=noise_scale,
        facecolor=facecolor,
        noise_hist=noise_hist,
        **scat_kwargs
    )


def plot_scatter_jittered(
        ax, center, values, c=None, width=.8, style='default',
        s=3, alpha=1, edgecolor='none', zorder=1000,
        noise_scale=10,
        facecolor='dark',
        noise_hist=True,
        **scat_kwargs
):
    """
    Plot a single scatter of values around a center.
    """
    if isinstance(style, str):
        style = styles_df.loc[style]
    assert isinstance(style, (pd.Series, dict))

    if c is not None:
        scat_kwargs['c'] = c
    else:
        scat_kwargs['facecolor'] = style.get(facecolor, facecolor)

    left = center - width * .5
    right = center + width * .5

    if noise_hist:
        xnoise = _generate_scatter_jitter(values, std_scale=noise_scale / 10 * 2)
        xnoise = xnoise * .5 * width * .8 + center

    else:
        xnoise = np.random.normal(size=len(values), scale=(right - left) / noise_scale, loc=center)
        xnoise = np.clip(xnoise, left, right)

    ax.scatter(
        xnoise, values,
        zorder=zorder + 2,
        s=s, alpha=alpha,
        edgecolor=edgecolor,
        **scat_kwargs
    )


def plot_box(
        ax, center, values,
        q1=.25, q2=.5, q3=.75,
        width=.8, style='default', zorder=1000,
        rect_facecolor='light',
        rect_edgecolor='light',
        median_color='darker',
):
    """
    Plot a box indicating quantiles of the data.
    """
    if isinstance(style, str):
        style = styles_df.loc[style]
    assert isinstance(style, (pd.Series, dict))

    left = center - width * .5
    right = center + width * .5

    q_values = [
        q(values) if callable(q) else np.nanquantile(values, q)
        for q in (q1, q2, q3)
    ]
    v1, v2, v3 = q_values
    ax.add_patch(
        matplotlib.patches.Rectangle(
            (left, v1),
            width=right - left,
            height=(v3 - v1),
            facecolor=style.get(rect_facecolor, rect_facecolor),
            edgecolor=style.get(rect_edgecolor, rect_edgecolor),
            zorder=zorder,
        )
    )
    ax.plot(
        [left, right],
        [v2, v2],
        color=style.get(median_color, median_color),
        zorder=zorder + 1,
    )


def mean_roll_by(x: np.array, y: np.array, win: tuple, xvals: np.array):
    """
    for every value in xvals, take the mean of all ys whose corresponding x falls within a given local window
    :param x:
    :param y:
    :param win:
    :param xvals:
    :return:
    """
    if isinstance(x, pd.Series):
        x = x.values

    if isinstance(y, pd.Series):
        y = y.values

    assert len(x) == len(y)

    return _mean_roll_by(x, y, win, xvals)


@numba.njit
def _mean_roll_by(x: np.array, y: np.array, win: tuple, xvals: np.array):
    """ fast implementation of mean_roll_by """
    rolled = np.empty(len(xvals))

    for i, x_center in enumerate(xvals):
        vmin = x_center + win[0]
        vmax = x_center + win[1]

        rolled[i] = np.nanmean(y[(vmin <= x) & (x <= vmax)])

    return rolled


def plot_rolled_mean(
        ax, x, y, xaxis_samples=101, xaxis_qrange=(.01, .99), win_width=None, style='default',
        linewidth=2,
        **kwargs,
):
    """
    Plot a trace generated by a mean on a sliding window
    """
    xaxis = np.linspace(
        np.nanquantile(x, xaxis_qrange[0]),
        np.nanquantile(x, xaxis_qrange[1]),
        xaxis_samples
    )

    if win_width is None:
        vrange = np.nanmax(x) - np.nanmin(x)
        win_width = .1 * vrange

    win = (-.5 * win_width, .5 * win_width)

    rolled = mean_roll_by(x, y, win, xaxis)

    style = lookup_style(style)

    ax.plot(
        xaxis, rolled,
        color=style['dark'],
        linewidth=linewidth * .75,
        path_effects=[
            matplotlib.patheffects.Stroke(linewidth=linewidth, foreground=style['darker']),
            matplotlib.patheffects.Normal(),
        ],
        **kwargs,
    )


def set_log_ticks(ax, vmax, include_zero=True):
    """
    Set nice log10 ticks on the y axis
    This uses "1k" instead of 1000.
    """
    ticks_major = np.power(10, np.arange(0, np.ceil(np.log10(vmax)) + 1))

    if include_zero:
        ticks_major = np.append(0, ticks_major)

    ax.tick_params(left=True, labelleft=True, which='major', length=3)
    ax.tick_params(left=True, labelleft=False, which='minor', length=2)

    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks_major))
    ax.set_yticklabels([f'{t:g}' if t < 1000 else f'{t:g}'[:-3] + 'k' for t in ticks_major])

    ticks_minor = np.concatenate([
        np.arange(0, 10, 1) * base
        for base in ticks_major[1 if include_zero else 0:-1]
    ])

    ax.yaxis.set_minor_locator(matplotlib.ticker.FixedLocator(ticks_minor))

    ax.spines['left'].set_bounds(0 if include_zero else 1, vmax)
