"""
code to track produced figures or results and classify them
"""

import contextlib
import numpy as np
import logging
import os
import socket
import sys
import inspect
import datetime
import yaml
import argparse
from tctx.util.defaults import DEFAULTS
from pathlib import Path

import filelock
logging.getLogger('py-filelock.filelock').setLevel(logging.WARNING)


@contextlib.contextmanager
def grab_all_to_file(filename):
    """
    We need some fancy pipe-ing to capture all of nest's output

    based on: https://stackoverflow.com/a/29834357
    """
    with open(filename, 'w') as log_file:
        with grab_all(log_file):
            yield


@contextlib.contextmanager
def grab_all(log_file):
    with grab_single(sys.stdout, log_file):
        with grab_single(sys.stderr, log_file):
            yield


@contextlib.contextmanager
def grab_single(original_stream, impostor_stream):
    original_stream.flush()

    fileno = original_stream.fileno()
    saved_original = os.dup(fileno)

    try:
        os.dup2(impostor_stream.fileno(), fileno)
        yield

    finally:
        os.dup2(saved_original, fileno)


@contextlib.contextmanager
def new_experiment(config=None, *args, **kwargs):
    """
    Context for tracking an experiment. It does:
    - Reads commandline args for general setup
    - Registers the experiment in a global registry (with  optional additional notes pulled from commandline)
    - Sets up global logging level (read from commandline)
    - Returns an ExperimentTracker object to keep track of created figures and processed data
    - Closes the tracker on exit.

    Use like:
        with new_experiment(mode='show') as tracker:
            # use the tracker
            pass

    :param config: object resulting from parsing options (see get_argparser)
    :param args: args to new_tracker
    :param kwargs: kwargs to new_tracker
    :return: ExperimentTracker context
    """
    tracker = new_tracker(*args, **kwargs)

    if config is None:
        config = get_argparser().parse_args()

    registry_extra = dict(
        notes=config.notes,
        pid=os.getpid(),
        cwd=os.getcwd(),
        hostname=socket.gethostname(),
    )

    if not config.redirect:
        grab = contextlib.nullcontext()
    else:
        log_file_name = str(tracker.get_data_path('run.log'))
        print('saving output to:', log_file_name)
        sys.stdout.flush()
        registry_extra['log_file_name'] = log_file_name
        grab = grab_all_to_file(log_file_name)

    with grab:
        # need to do this at this level so the output is correctly redirected
        # Alternatively I could use FileHandler objects which is cleaner but I worry
        # that the logging messages won't be correctly interleaved with print messages
        logging.basicConfig(level=config.loglevel.upper())
        logging.getLogger('matplotlib').setLevel(logging.WARNING)  # don't need to see debug from matplolib

        global_reg = GlobalRegistry.from_gpfs()

        global_reg.add(**registry_extra, tracker=tracker)

        with DEFAULTS.use_all_cores(minus=config.spare_cores):
            with DEFAULTS.update(parallel_pool=config.parallel_pool):

                yield tracker

        tracker.finish()


def new_tracker(base_path=None, mode='save show', extra_path=None, prefix_stack=True, timestamp=True, categories=None):
    """
    Create a tracker object to know where to store results and figures. The most complete path created would look like:
        base_path /
        [data | figures] /
        [interim | saved | external | processed] /
        caller_filename / caller_function /
        timestamp

    :param base_path: where to store all results
    :param extra_path: something to append to base_path
    :param mode: string containing none or multiple of:
        - save: save the figures
        - show: run plt.show at the end
    :param prefix_stack: use the callstack to extract the caller's file and function name and use them as subfolders.
    :param timestamp: use a timestamp as a subfolder.
    :param categories: list of categories (folders) to use. Also accepts a single string.
    :return: an ExperimentTracker object
    """
    if categories is None:
        categories = []

    elif isinstance(categories, str):
        categories = [categories]

    elif not isinstance(categories, list):
        categories = list(categories)

    if prefix_stack:
        caller = extract_caller()
        categories = list(caller) + categories

    if timestamp:
        categories.append(get_timestamp())

    if base_path is None:
        base_path = get_storage_base_path()

    if extra_path is not None:
        base_path = Path(base_path, extra_path)

    tracker = ExperimentTracker(
        base_path=str(base_path),
        categories=categories,
        mode=mode)

    return tracker


def new_tracker_nb(name, level=logging.DEBUG, prefix_stack=False, timestamp=False, **kwargs):
    """
    Create a tracker object that is notebook-friendly. See new_tracker for details.
    """
    logging.basicConfig(level=level)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)  # don't need to see debug from matplolib

    et = new_tracker(
        prefix_stack=prefix_stack,
        timestamp=timestamp,
        **kwargs,
    )

    et.categories.extend(name.replace(' ', '_').replace('-', '_').split('/'))
    return et


def extract_caller():
    """Look at the callstack and pick return the name of the highest file and function outside the track module"""
    stack = inspect.stack()

    # for frame_info in stack:
    #     print(frame_info.filename, frame_info.function)

    for frame_info in stack:
        ignored = ('track.py', 'contextlib')
        if all(name not in frame_info.filename for name in ignored):
            filename = Path(frame_info.filename).stem
            return filename, frame_info.function

    return 'unknown', 'unknown'


def get_argparser(*args, **kwargs):
    """Some sensible default options for command-line launching of simulations"""
    parser = argparse.ArgumentParser(*args, **kwargs)

    parser.add_argument('-l', '--loglevel', help='One of (debug, info, warning, error)',
                        default='debug', action='store', type=str)

    parser.add_argument('--redirect', help='Redirect stdout/err to a file.',
                        default=False, action='store_true')

    parser.add_argument('-r', '--notes', help='Additional notes regarding this particular run.',
                        default='', action='store', type=str)

    parser.add_argument('--spare-cores', help='Cores to spare.',
                        default=1, action='store', type=int)

    parser.add_argument('--parallel-pool', default='process', required=False,
                        help='Whether to use serial/thread-pool/process-pool for parallel tasks.', type=str,
                        choices=['serial', 'process', 'thread'])

    return parser


class GlobalRegistry:
    """
    We keep track of all the jobs that we have run.
    Note one job may contain multiple sims run in series.
    """

    def __init__(self, base_path):
        self.base_path = Path(base_path)

    @classmethod
    def from_gpfs(cls):
        return cls(get_storage_base_path())

    @property
    def _fullpath(self) -> Path:
        """get the full path to the yaml registry where we track all job executions"""
        return self.base_path / 'data' / 'registry.yml'

    @property
    def _lockpath(self) -> Path:
        """get the full path to the yaml registry where we track all job executions"""
        return Path(str(self._fullpath) + '.lock')

    def add(self, tracker, **extra):
        """
        register a new experiment in a global log in order
        to keep track of what was generated when and with what purpose

        This is not thread-safe in any way.
        """
        now = datetime.datetime.now()

        properties = extra.copy()
        properties['date'] = now.strftime('%Y.%m.%d')
        properties['time'] = now.strftime('%H:%M:%S:%f')
        properties['categories'] = tracker.categories
        properties['cmd'] = sys.argv
        properties['data'] = str(tracker.get_data_path('', ensure=False))
        properties['figures'] = str(tracker.get_figures_path('', ensure=False))

        full_text = '\n' + yaml.dump([properties])
        logging.debug('Registering experiment: %s', tracker.categories)

        with filelock.FileLock(str(self._lockpath)):
            if not self._fullpath.exists():
                logging.debug('Create registry of experiments: %s', self._fullpath)
                with open(self._fullpath, 'w') as f:
                    f.write('# Registry of experiments\n---\n')

            with open(self._fullpath, 'a') as f:
                f.write(full_text)

    def load(self) -> list:
        """load the full registry of all jobs"""
        with filelock.FileLock(str(self._lockpath)):
            with open(self._fullpath, 'r') as f:
                try:
                    # avoid warning about default loader
                    contents = yaml.load(f, Loader=yaml.FullLoader)

                except AttributeError:
                    # because of different versions of yaml
                    contents = yaml.load(f)

        return contents


def get_timestamp() -> str:
    """return a human-friendly timestamp text"""
    return datetime.datetime.now().strftime('%Y.%m.%d.%H.%M.%S.%f')


def get_storage_base_path():
    """find location of data assuming its relative to this package's installation"""
    return (Path(__file__) / '../../../../tctx/').resolve()


class ExperimentTracker(object):
    """track your experiment generated results and figures"""

    def __init__(self, base_path=None, categories=None, mode='save show'):
        if base_path is None:
            base_path = get_storage_base_path()

        self.base_path = Path(base_path).resolve()
        self.categories = categories or []
        self.mode = mode

        self.accessed = set()

    def get_dir(self, main_dir, state_category, subpath, ensure=False, is_dir=False) -> Path:
        """get the full path to an item, ensure the folder structure exists if necessary"""
        assert state_category in ('external', 'interim', 'raw', 'processed')

        subpath = Path(*(list(self.categories) + [subpath]))

        if not is_dir:  # is the item a directory itself or a file
            subpath = subpath.parent

        full_path = Path(self.base_path, main_dir, state_category, subpath).resolve()

        if ensure:
            full_path.mkdir(exist_ok=True, parents=True)

        return full_path

    def get_external_data_path(self, subpath):
        """gets a fixed path that is independent of this experiment"""
        return Path(self.base_path, 'data', 'external', subpath).resolve()

    def get_processed_data_path(self, subpath):
        """gets a fixed path that is independent of this experiment"""
        return Path(self.base_path, 'data', 'processed', subpath).resolve()

    def _get_specific_path(self, subpath, main_category, state_category, timestamp, ensure, is_dir) -> Path:
        """get the full path that is dependent on this experiment
        Ensure the folder structure exists if necessary"""

        assert main_category in ('data', 'figures'), \
            'unknown main category %s' % main_category

        assert state_category in ('interim', 'processed'), \
            'unknown state category %s' % state_category

        subpath = Path(subpath)

        if timestamp:
            subpath = subpath.parent / (get_timestamp() + '_' + subpath.name)

        if is_dir:
            final_path = self.get_dir(main_category, state_category, subpath, ensure=ensure, is_dir=is_dir)

        else:
            dir_path = self.get_dir(main_category, state_category, subpath, ensure=ensure, is_dir=is_dir)

            final_path = dir_path / subpath.name
            self.accessed.add(str(final_path))

        return final_path

    def get_data_path(self, subpath, state_category='interim', timestamp=False, ensure=True, is_dir=False):
        return self._get_specific_path(subpath, 'data', state_category, timestamp, ensure, is_dir=is_dir)

    def get_figures_path(self, subpath, state_category='interim', timestamp=False, ensure=True, is_dir=False):
        return self._get_specific_path(subpath, 'figures', state_category, timestamp, ensure, is_dir=is_dir)

    @contextlib.contextmanager
    def subcategory(self, name):
        self.categories.append(name)
        yield
        self.categories.pop()

    def finish(self):
        if 'show' in self.mode:
            # TODO undo workaround for broken installation
            #
            #     there is a conflict on the loading of a so library in the cluster
            #     that happens when we load
            #
            #         import nest
            #         import matplotlib.pyplot
            #         import scipy.stats
            #
            #     in THAT SPECIFIC ORDER. This needs to be fixed, possibly by reinstalling
            #     annaconda and recompiling nest. I suspect it's caused by changes
            #     in the cluster. In anycase, I need to run a few simulations for a
            #     deadline (poster) so workaround it for now.
            from matplotlib import pyplot as plt
            plt.show()

        logging.info('Accessed paths:\n%s', '\n'.join(sorted(list(self.accessed))))

    def finished_figure(self, figure, state_category='interim', extensions=('.pdf',),
                        rasterized=None, dpi=None, title=None, timestamp=False):

        while isinstance(figure, (tuple, list, set, np.ndarray)):
            figure = figure[0]

        if hasattr(figure, 'figure') and figure.figure is not None:
            figure = figure.figure

        title = figure.canvas.get_window_title() if title is None else title

        if title == 'image':  # the default canvas title from matplotlib
            logging.warning('non descriptive figure title')

        if 'save' in self.mode:
            for extension in extensions:
                full_path = self.get_figures_path(title.replace(' ', '_').replace('-', '_') + extension,
                                                  state_category=state_category, ensure=True, timestamp=timestamp)
                logging.info('saving %s', full_path)
                figure.savefig(full_path, transparent=True, dpi=dpi, rasterized=rasterized)

        if 'show' not in self.mode:
            figure.clear()
