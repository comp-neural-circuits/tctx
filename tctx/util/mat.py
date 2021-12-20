"""code to explore Matlab files"""

import time

import numpy as np
import scipy.io


def explore(mat, tab_count=0, skip_meta=True):
    tabs = '  ' * tab_count
    if type(mat) == dict:
        for k in mat.keys():
            if not skip_meta or not k.startswith('_'):
                if isinstance(mat[k], (bytes, str, list)):
                    print('%s%s: %s' % (tabs, k, mat[k]))

                elif not mat[k].dtype.fields:
                    print('%s%s: %s %s' % (tabs, k, mat[k].shape, mat[k].dtype))
                else:
                    print('%s%s: %s' % (tabs, k, mat[k].shape))
                    explore(mat[k], tab_count + 1)

    elif type(mat) == np.ndarray or type(mat) == np.void:
        if mat.shape and max(mat.shape) > 0:
            if max(mat.shape) > 1:
                print('%s%s' % (tabs, mat.shape))
            idx = tuple([0] * len(mat.shape))
            explore(mat[idx], tab_count + 1)
        else:
            if mat.dtype.fields:
                for k, v in mat.dtype.fields.items():
                    print('%s%s: %s' % (tabs, k, v))

                    explore(mat[k], tab_count + 1)


def load_single(full_filename, silent=False):

    full_filename = str(full_filename)

    tstart = time.time()
    try:
        data = scipy.io.loadmat(full_filename)

    except NotImplementedError:
        import hdf5storage
        data = hdf5storage.loadmat(full_filename)

    if not silent:
        print('Contents of', full_filename)
        explore(data)
        print('took:', time.time() - tstart, 'seconds')
    return data
