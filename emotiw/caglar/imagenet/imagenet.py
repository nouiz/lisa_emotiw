import os
import gc
import warnings
try:
        import tables
except ImportError:
        warnings.warn("Couldn't import tables, so far SVHN is "
                            "only supported with PyTables")
import numpy
from theano import config
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils.serial import load
from pylearn2.utils.string_utils import preprocess

class Imagenet(dense_design_matrix.DenseDesignMatrixPytables):
    mapper = {
        'train': 0,
        'test': 1,
        'valid': 2
    }

    def __init__(self, which_set,
            path,
            center,
            size_of_receptive_field,
            stride,
            start,
            stop,
            mode=None,
            axes=('b', 0, 1, 'c'),
            preprocessor=None):

        assert which_set in self.mapper.keys()

        self.__dict__.update(locals())
        del self.self

        self.mode = mode

        if not os.path.isfile(path):
            raise ValueError("The path you have entered is not a valid path.")

        self.h5file = tables.openFile(file_n, mode = mode)

        """
        TODO
        """
