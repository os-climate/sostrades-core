# -*- coding: utf-8 -*-

import pickle
from bz2 import BZ2File


def dump_compressed_pickle(fname, data):
    ''' writes data as a compressed pickle file
    '''
    with BZ2File(fname, 'w') as f:
        pickle.dump(data, f)

def load_compressed_pickle(fname):
    ''' decompress pickle and returns the data
    '''
    data = BZ2File(fname, 'rb')

    return pickle.load(data)
