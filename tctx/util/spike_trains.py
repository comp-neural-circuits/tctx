"""a collection of functions to process multi-neuron spike trains using pandas DataFrames"""
import numpy as np
import numba
import pandas as pd


ms_to_s = .001


class Sorting:
    """a way to sort indexable items. For example, for neurons in raster plots.

    A sorting is just a map from a set of real indices (for example cell's identifiers)
    to an ordered abstract value. It is stored as a series.
    """
    def __init__(self, series, description=None):
        if series is not None:
            assert not np.any(series.index.duplicated())
            # TODO decide:
            # assert not np.any(series.duplicated())

        self.series = series
        self.description = description

    def get_label(self, linebreak=True, item='neurons'):
        """return a description for this sorting"""
        sep = "\n" if linebreak else " "

        if self.series is not None:
            if self.description is not None:
                return f'{item} sorted{sep}{self.description}'
            else:
                return f'sorted {item}'
        else:
            return f'{item} ids'

    def collapse(self, indices):
        """assign new values to the sorting so it only contains
        the indices given while preserving the order"""
        if self.series is None:
            collapsed_sorting = indices

        else:
            collapsed_sorting = self.series.index[self.series.index.isin(indices)]

        return Sorting(
            series=pd.Series(
                index=collapsed_sorting,
                data=np.arange(len(collapsed_sorting))),
            description=self.description)

    def reverse(self):
        if self.series is None:
            return self

        return Sorting(
            series=pd.Series(
                index=self.series.index,
                data=self.series.values[::-1]).sort_values(),
            description=self.description)

    def apply(self, indices):
        """apply this sorting to an existing index
        Note that the items returned are not the same index, but the rank the given ones should
        be given. If you want the sorted indices use sort()
        """
        if self.series is None:
            return indices

        return self.series.reindex(indices).values

    def sort(self, indices):
        if isinstance(indices, pd.Series):
            index = indices.index
            values = indices.values
            argsort = np.argsort(self.apply(values))

            return pd.Series(index=index[argsort], data=values[argsort])

        else:
            values = indices

            mapped = self.apply(values)

            # note that if a value is missing from this sorting, it should be removed
            valid = ~np.isnan(mapped)
            argsort = np.argsort(mapped[valid])

            return values[valid][argsort]

    @staticmethod
    def from_series(series, description=None, ascending=True):
        new_series = pd.Series(
            index=series.sort_values(ascending=ascending).index,
            data=np.arange(len(series)))

        new_series.name = series.name

        if description is None and series.name is not None:
            description = series.name.replace('_', ' ')

        return Sorting(new_series, ('by ' + description) if description is not None else None)

    @staticmethod
    def from_multiple_series(description=None, ascending=True, **all_series):
        df = pd.DataFrame(all_series).dropna().sort_values(list(all_series.keys()), ascending=ascending)

        if description is None:
            names = [name.replace('_', ' ') for name in all_series.keys()]
            if len(names) > 1:
                description = ', '.join(names[:-1]) + ' & ' + names[-1]
            else:
                description = names[0]

        new_series = pd.Series(
            index=df.index,
            data=np.arange(len(df)))

        new_series.name = 'multiple_sorting'

        return Sorting(new_series, 'by ' + description)

    @staticmethod
    def none():
        return Sorting(None, None)

    @staticmethod
    def random(values, width=(0., 1.)):
        values = np.unique(values)

        randomised = np.random.rand(len(values)) * (width[1] - width[0]) + width[0]

        return Sorting(
            series=pd.Series(data=randomised, index=values),
            description='randomly'
        )


@numba.njit
def get_wrapped_distance_points(xy, origin, side):
    diff = xy - np.expand_dims(origin, axis=0)
    diff = np.abs(diff)

    diff = np.minimum(diff, side - diff)
    distance = np.sqrt(np.sum(np.square(diff), axis=-1))
    return distance


def wrap_around_dimension(values, full_side, new_origin):
    half_side = full_side * .5
    offset = values - new_origin

    if new_origin > half_side:
        mask = offset < -half_side
        offset[mask] = offset[mask] + full_side

    else:
        mask = offset >= half_side
        offset[mask] = offset[mask] - full_side

    return offset


def center_cells(x, y, origin, side_um) -> pd.DataFrame:
    x = wrap_around_dimension(
        x,
        side_um,
        origin[0])

    y = wrap_around_dimension(
        y,
        side_um,
        origin[1])

    return pd.DataFrame({'x': x, 'y': y})


time_key = 'delay_from_induced'
win_key = 'win_idx'


########################################################################################################################
# Windows to define sections of experiments


class ExclusiveWindows:
    """Fast classification of events in exclusive windows"""

    def __init__(self, windows, by=None):
        """
        :param windows: a windows DataFrame
        :param by: a categorical column that will be used to classify the windows
        """
        self.windows = windows

        edges = windows[['start', 'stop']].values.flatten()
        assert np.all(np.diff(edges) >= 0), 'windows monotonically increasing'

        is_real_window = np.zeros(len(edges), dtype=np.bool_)
        is_real_window[0::2] = np.ones(len(windows), dtype=np.bool_)

        # If it falls outside any valid window,
        # the index will be -1.
        # Using this instead of pandas nan to represent missing data so we can stick to integers.
        invalid_value = windows.index.max() + 1
        win_idx = np.ones(len(edges), dtype=windows.index.dtype) * invalid_value
        win_idx[0::2] = windows.index

        not_empty = np.concatenate([(np.diff(edges) != 0), [True]])
        edges = edges[not_empty]
        is_real_window = is_real_window[not_empty]
        win_idx = win_idx[not_empty]

        self.edges = edges  # the edges of the bins representing window and inter-window periods
        self.is_real_window = is_real_window  # whether the bin is a window (True) or an inter-window period
        self.win_idx = win_idx  # the index of the window (invalid_value if it's not)
        self.invalid_value = invalid_value  # the index that represents out-of-window

        if by is not None:
            if isinstance(by, str):
                by = windows[by]

            by = by.astype('category')
            self.by_codes = by.cat.codes.values
            self.by_cat_names = by.dtype.categories.values
            self.by_name = by.name

        else:
            self.by_codes = None
            self.by_cat_names = None
            self.by_name = None

    def __str__(self):
        return self.windows.__str__()

    def __repr__(self):
        return self.windows.__repr__()

    def _repr_html_(self):
        return self.windows._repr_html_()

    @staticmethod
    @numba.njit
    def _classify_time_points_raw(ts, edges, is_real_window, win_idx, invalid_value):
        """
        :returns: the index of the window where each ts falls.
        """
        bin_idcs = np.digitize(ts, edges)

        # in any bin
        mask = (bin_idcs >= 1) & (bin_idcs < len(edges))

        # shift idcs to represent left edge
        bin_idcs = bin_idcs - 1

        # skip "fake" bins that represent between-windows
        bin_idcs[~mask] = 0
        mask = mask & is_real_window[bin_idcs]

        idcs = win_idx[bin_idcs].copy()
        idcs[~mask] = invalid_value

        return idcs

    def classify_spikes(self, spikes):
        """
        compute the relative time of each spike, depending on which window it falls into
        :returns: DF be like

                         gid  delay  win_idx
            spike_idx
            1          85409   58.7        0
            2          20825   59.4        0
            ...          ...    ...      ...
            248507     62476  122.9       98
            248509     29716  134.5       98

        """
        gids = spikes.gid.values
        times = spikes.time.values

        idcs = self._classify_time_points_raw(times, self.edges, self.is_real_window, self.win_idx, self.invalid_value)
        mask = idcs != self.invalid_value

        times = times[mask]
        idcs = idcs[mask]

        delays = times - self.windows.ref.values[idcs]
        mark_idx = self.windows.index.values[idcs]

        mark_name = self.windows.index.name
        if mark_name is None:
            mark_name = 'win_idx'

        df = {
            'gid': gids[mask],
            'delay': delays,
            mark_name: mark_idx,
        }

        if self.by_name is not None:
            df['cat'] = self.windows['cat'].values[idcs]

        return pd.DataFrame(df, index=spikes.index[mask])

    # @numba.njit
    def _count_spikes_raw(self, times, gids, win_cats):
        """
        :returns: an array of shape (GIDS, CATEGORIES)
        that contains the number of spikes of each category for each gid.
        """
        idcs = self._classify_time_points_raw(times, self.edges, self.is_real_window, self.win_idx, self.invalid_value)

        mask = idcs != self.invalid_value

        gids = gids[mask]
        idcs = idcs[mask]

        cats_count = np.max(win_cats) + 1

        counts = np.bincount(gids * cats_count + win_cats[idcs])

        # it may happen that the last gid doesn't have anything in the last category
        # resulting in a non-reshapable array. Just fill these with 0s.
        missing = len(counts) % cats_count
        if missing != 0:
            counts = np.concatenate([counts, np.zeros(missing, dtype=counts.dtype)])

        counts = counts.reshape((len(counts) // cats_count, cats_count))

        return counts

    def count_spikes(self, spikes):
        """
        :returns: a dataframe of shape (GIDS, CATEGORIES)
        that contains the number of spikes of each category for each gid.
        """
        counts = self._count_spikes_raw(
            spikes.time.values,
            spikes.gid.values,
            self.by_codes,
        )

        counts = pd.DataFrame(counts)
        counts.columns = self.by_cat_names[counts.columns]
        counts.columns.name = 'cat'
        counts.index.name = 'gid'

        return counts

    def get_length_by_cat(self):
        """return a series matching each category to the total time covered by its windows"""
        return (self.windows.stop - self.windows.start).groupby(self.windows.cat).sum()

    @classmethod
    def build_between(cls, breaks, start=None, stop=None):
        """
        Create sequential windows between some break points.
        The windwos are most likely not equal sized.
        :param breaks: timepoints that define the windows
        :param start: if present, add a window between "start" to the start of the first window
        :param stop: if present, add a window between the stop of the last window to "stop"
        :return:
        """

        if isinstance(breaks, pd.Series):
            breaks = breaks.values

        assert isinstance(breaks, np.ndarray)

        if start is not None:
            endpoint = breaks.min()
            if start < endpoint:
                breaks = np.concatenate([[start], breaks])

        if stop is not None:
            endpoint = breaks.max()
            if stop > endpoint:
                breaks = np.concatenate([breaks, [stop]])

        breaks = np.sort(np.unique(breaks))

        df = pd.DataFrame(
            {
                'start': breaks[:-1],
                'stop': breaks[1:],
                'ref': breaks[:-1],
            },
        )

        df.index.name = 'win_idx'
        df.name = 'windows'

        return cls(df)


def make_windows(marks, win_ms):
    """
    build a dataframe containing several time windows around a series of time points
    for each row entry there are the start/stop points and the reference
    (sometimes used to compute the relative time).

    :param marks: a series of time points round which the windows are built
    :param win_ms: a bi-tuple of time deltas, example: (-50, 150.)
    :return: a df that looks like:
            start    stop     ref
        0  1250.0  1400.0  1300.0
        1  1550.0  1700.0  1600.0
        2  1850.0  2000.0  1900.0
        3  2150.0  2300.0  2200.0
        4  2450.0  2600.0  2500.0
    """
    assert win_ms[0] < win_ms[1]

    if isinstance(marks, (list, tuple, np.ndarray)):
        marks = pd.Series(marks)

    marks = marks.sort_values()

    all_windows = marks.values[:, np.newaxis] + np.array(win_ms)

    df = pd.DataFrame(
        {
            'start': all_windows[:, 0],
            'stop': all_windows[:, 1],
            'ref': marks,
        },
        index=marks.index)

    if df.index.name is None:
        df.index.name = 'win_idx'

    df.name = 'windows'

    return df


def assign_spikes_to_windows(spikes, windows, pbar=None):
    """
    give a set of time windows, categorize the spikes that fall within each time window

    Note that windows may overlap in which case the same spike may be added multiple times.
    This is why spike_idx is not the index of the resulting table.

    Window is interpreted as left inclusive [0, 1), ie a spike clasifies if
        windows.start <= (spike.time - mark.time) < windows.stop

    You can use this for example to select spikes happening right before a pulse in a long recording:

        extract_relative_spike_time(spikes, pulses.time, win_ms=(-50, 0))

    :param spikes: a dataframe of spikes containing time and other data.
    :param windows: a dataframe with [start, stop) as columns and windows as rows
    :return: looks like:
               spike_idx  abs_time  rel_time    gid  mark_idx
        0         30     556.3     -43.7  97439         1
        1         31     563.3     -36.7  98303         1
        2         32     568.1     -31.9  82495         1
        3         33     570.7     -29.3  99711         1
        4         34     574.4     -25.6  87647         1
     """

    pbar = (lambda x, total: x) if pbar is None else pbar

    spikes_in_windows = []

    for idx, (start, stop, ref) in pbar(windows.iterrows(), total=len(windows)):
        inside = (start <= spikes.time) & (spikes.time < stop)

        spikes_in_windows.append(pd.DataFrame({
            spikes.index.name if spikes.index.name is not None else 'spike_idx': spikes.index[inside],
            'abs_time': spikes.time[inside].values,
            time_key: spikes.time[inside].values - ref,
            'gid': spikes.gid[inside].values,
            win_key: idx
        }))

    if spikes_in_windows:
        spikes_in_windows = pd.concat(spikes_in_windows, axis=0, ignore_index=True)

    else:
        spikes_in_windows = pd.DataFrame([], columns=['spike_idx', 'abs_time', time_key, 'gid', win_key])

    assert spikes_in_windows.index.is_unique
    return spikes_in_windows


def bin_count(wins, times):
    """given a set of non-exclusive windows,
    count the number of elements of times that fall within each one"""

    start = wins.start.values[:, np.newaxis]
    stop = wins.stop.values[:, np.newaxis]

    if isinstance(times, (tuple, list)):
        times = np.array(times)

    if isinstance(times, pd.Series):
        times = times.values
    times = times[np.newaxis, :]

    contains = np.count_nonzero((start <= times) & (times < stop), axis=1)

    return contains


def are_windows_exclusive(windows):
    """
    :param windows: a windows pd.DF
    :return: bool: whether the windows monotonically increase
    """
    edges = windows[['start', 'stop']].sort_values(['start', 'stop']).values.flatten()
    return np.all(np.diff(edges) >= 0)


def merge_overlapping_windows(windows):
    """
    :param windows: a windows pd.DF
    :return: another DF guaranteed to have exclusive windows. Those that overlap are merged.
    """

    index = windows.sort_values('start').index

    remaining = []

    w1 = index[0]
    w0_start = windows.loc[w1, 'start']
    w0_stop = windows.loc[w1, 'stop']
    w0_ref = windows.loc[w1, 'ref']

    for w1 in index[1:]:
        if windows.loc[w1, 'start'] <= w0_stop < windows.loc[w1, 'stop']:
            w0_stop = windows.loc[w1, 'stop']

        else:
            remaining.append([w0_start, w0_stop, w0_ref])
            w0_start = windows.loc[w1, 'start']
            w0_stop = windows.loc[w1, 'stop']
            w0_ref = windows.loc[w1, 'ref']

    remaining.append([w0_start, w0_stop, w0_ref])

    df = pd.DataFrame(remaining, columns=['start', 'stop', 'ref'])
    df.index.name = 'win_idx'

    assert are_windows_exclusive(df)
    return df


def invert_windows(windows, start=None, stop=None):
    """
    Select all of the time except that covered by the given windows.
    You can use this to define a baseline time period that is far enough from key events:

    # select all time in the series excluding any 500 ms window after an induced spike
    baseline_wins = spt.invert_windows(
        spt.make_windows(patched_spikes.time, (0, +500)),
        start=series_window[0],
        stop=series_window[1]
    )

    :param windows:
    :param start: if present, add a window between "start" to the start of the first window
    :param stop: if present, add a window between the stop of the last window to "stop"
    :return:
    """
    windows = windows.sort_values('start')

    if not are_windows_exclusive(windows):
        windows = merge_overlapping_windows(windows)
        assert are_windows_exclusive(windows)

    df = pd.DataFrame({
        'start': windows.stop.values[:-1],
        'stop': windows.start.values[1:],
    })

    if start is not None:
        endpoint = windows.start.values[0]
        if start <= endpoint:
            df = df.append({
                'start': start, 'stop': endpoint}, ignore_index=True)

    if stop is not None:
        endpoint = windows.stop.values[-1]
        if stop >= endpoint:
            df = df.append({
                'start': endpoint, 'stop': stop}, ignore_index=True)

    df = df[df.start != df.stop]
    df.sort_values('start', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['ref'] = df['start']
    df.index.name = 'win_idx'

    return df


def filter_windows_exclusive_ref(wins, marks=None):
    """
    Return a subset of wins that do NOT contain the reference of any other.
    Its fine if they contain their own reference.

    You can use this to create windows after patched spikes that do not contain more than one patched spike
    (the one they were created around).

    after_patched_wins = spt.make_windows(patched_spikes.time, (0, +250))
    after_patched_wins = spt.filter_windows_exclusive_ref(after_patched_wins)

    :param wins:
    :param marks: if not specified, use the window's ref
    :return:
    """
    if marks is None:
        marks = wins.ref

    expected = marks.between(wins.start, wins.stop).map({True: 1, False: 0})

    selection = wins[bin_count(wins, wins.ref) == expected]

    return selection.copy()


def split(wins, delay=0, cat0=None, cat1=None):
    """
    split windows by a fixed delay relative to the REF
    for each window, 2 new ones will be created: [start, ref+delay] and [ref+delay, stop]
    optionally, give categories to the windows

    Note the ref is still the same for both windows, which is on purpose so you can still use it to compute
    delays compatible across windows
    """
    offsets = (wins.ref + delay).values

    df0 = pd.DataFrame({'start': wins.start.values, 'stop': offsets, 'ref': wins.ref.values})
    if cat0 is not None:
        df0['cat'] = cat0

    df1 = pd.DataFrame({'start': offsets, 'stop': wins.stop.values, 'ref': wins.ref.values})
    if cat1 is not None:
        df1['cat'] = cat1

    df = pd.concat([df0, df1], axis=0)

    df = df.sort_values(['start', 'ref', 'stop']).reset_index(drop=True).rename_axis(index='win_idx')
    df.name = 'windows'

    return df

