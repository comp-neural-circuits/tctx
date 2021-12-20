import contextlib
import datetime
import logging
import time


@contextlib.contextmanager
def log_time(name='', level=logging.DEBUG, pre=False):
    if pre:
        logging.log(level, f'{name}...')
    tstart = time.time()
    yield
    logging.log(level, '%s total time %s', name, datetime.timedelta(seconds=time.time() - tstart))
