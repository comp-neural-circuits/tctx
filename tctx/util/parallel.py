"""
Code to launch multiple light independent tasks in parallel
"""
import logging
from tqdm.auto import tqdm
import concurrent.futures
import concurrent.futures.process
import contextlib

from tctx.util.defaults import DEFAULTS


def independent_tasks(func, all_params, mode=None, pbar=tqdm, **kwargs):
    if mode is None:
        mode = DEFAULTS['parallel_pool']

    assert mode in ['serial', 'process', 'thread']
    assert 'parallel_pool' not in kwargs.keys()

    if mode == 'serial':
        results = {}
        for i, params in enumerate(pbar(all_params, desc='completed')):
            results[i] = func(*params)
        return results

    else:
        return parallel_tasks(func, all_params, parallel_pool=mode, pbar=pbar, **kwargs)


def parallel_tasks(func, all_params, parallel_pool, max_workers=None, pbar=tqdm):

    if max_workers is None:
        max_workers = DEFAULTS['max_workers']

    logging.debug(
        'submitting %d jobs on parallel pool of %d %s', len(all_params), max_workers, parallel_pool)

    results = {}
    completed = [False] * len(all_params)

    pool_class = dict(
        process=concurrent.futures.ProcessPoolExecutor,
        thread=concurrent.futures.ThreadPoolExecutor,
    )[parallel_pool]

    try:
        with pool_class(max_workers=max_workers) as executor:
            futures_map = {}

            for i, params in enumerate(pbar(all_params, desc='submit')):
                future = executor.submit(func, *params)
                futures_map[future] = i

            try:
                for future in pbar(concurrent.futures.as_completed(futures_map.keys()),
                                   total=len(futures_map),
                                   desc='completed'):

                    i = futures_map[future]
                    result = future.result()
                    completed[i] = True

                    if result is not None:
                        results[i] = result

            except (KeyboardInterrupt, SystemExit):
                logging.warning('============ STOPPING ============ ')
                executor.shutdown(wait=False)

    except (KeyboardInterrupt, SystemExit):
        logging.warning('============ STOPPING ============ ')
        pass

    except concurrent.futures.process.BrokenProcessPool:
        logging.exception('============ Broken Process Pool ============ ')

    return results


def serial_tasks(func, all_params):
    # because sometimes it's easier to debug like this
    return {i: func(*params) for i, params in enumerate(tqdm(all_params))}


@contextlib.contextmanager
def limit_mkl_threads(limit):
    import mkl
    n = mkl.get_max_threads()

    try:
        mkl.set_num_threads(limit)
        yield

    finally:
        mkl.set_num_threads(n)
