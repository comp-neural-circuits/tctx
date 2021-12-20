from contextlib import contextmanager
import logging


class Defaults:
    def __init__(self, **kwargs):
        self.values = kwargs.copy()

    @contextmanager
    def update(self, **kwargs):
        """in order to control global parameters (like number of cores available)
        without having to pass it through every function

        use like:

            with sim.DEFAULTS.update(max_workers=16):
                 print(DEFAULTS.values)
            print(DEFAULTS.values)

        """
        prev_defaults = self.values.copy()

        try:
            logging.debug('using new defaults: %s', kwargs)
            self.values.update(kwargs)

            yield self

        finally:
            self.values = prev_defaults

    def use_all_cores(self, minus=0):
        """
        use like:

            with DEFAULTS.use_all_cores(minus=1):
                 print(DEFAULTS.values)
            print(DEFAULTS.values)

        """
        import multiprocessing
        core_count = multiprocessing.cpu_count()
        return self.update(max_workers=max(core_count - minus, 1))

    def __getitem__(self, item):
        return self.values[item]


DEFAULTS = Defaults(
    max_workers=32,
    grng_seed=1,  # nest
    parallel_pool='process',
)
