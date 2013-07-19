import os
import numpy as np
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils import serial
from pylearn2.utils.string_utils import preprocess
import tables

class ComboDatasetPyTable(dense_design_matrix.DenseDesignMatrixPyTables):
    def __init__(self, path = None, start = None, stop = None, shuffle=True,
                rng = None, seed = 132987, center = False,
                scale = False,
                axes=('b', 0, 1, 'c'), preprocessor=None,
                which_set = 'all'):

        if path is None:
            path = '/data/lisa/data/faces/EmotiW/preproc/'
            mode = 'r'
        else:
            mode = 'r+'


        path = preprocess(path)
        file_n = "{}{}.h5".format(path, which_set)
        if os.path.isfile(file_n):
            make_new = False
        else:
            make_new = True

        if make_new:
            self.make_data(path, shuffle, rng, seed, which_set, start, stop)

        self.h5file = tables.openFile(file_n, mode=mode)
        data = self.h5file.getNode('/', "Data")

        if not make_new and(start != None or stop != None):
            raise ValueError("Ah ah")

        self.axes = axes
        view_converter = dense_design_matrix.DefaultViewConverter((48, 48, 1), axes)
        super(ComboDatasetPyTable, self).__init__(X=data.X, y=data.y, view_converter=view_converter)
        assert not np.any(np.isnan(self.X))

        if preprocessor is not None:
            preprocessor.apply(self)


    @staticmethod
    def make_data(path, shuffle = True, rng = None, seed = 132987, which_set = 'all', start = None, stop = None):
        file_n = "{}{}.h5".format(path, which_set)

        orig_path = '/data/lisa/data/faces/EmotiW/preproc/'
        data_x = np.memmap(orig_path + 'all_x.npy', mode='r', dtype='float32')
        data_y = np.memmap(orig_path + 'all_y.npy', mode='r', dtype='uint8')
        data_x = data_x[::3]
        data_y = data_y[::3]
        data_x = data_x.reshape((data_y.shape[0], 48 * 48))

        one_hot = np.zeros((len(data_y),7), dtype=np.float32)
        one_hot[np.asarray(range(len(data_y))), data_y] = 1.
        data_y = one_hot

        if shuffle:
            rng = rng if rng else np.random.RandomState(seed)
            rand_idx = rng.permutation(len(data_x))
            data_x = data_x[rand_idx]
            data_y = data_y[rand_idx]

        if start is not None or stop is not None:
            if start is None:
                start = 0
            else:
                assert start >= 0
            if stop is None:
                stop = -1
            if stop != -1:
                assert stop > start
            data_x = data_x[start:stop]
            data_y = data_y[start:stop]

        h5file, node = ComboDatasetPyTable.init_hdf5(file_n, ((data_x.shape[0], data_x.shape[1]), (data_y.shape[0], 7)))
        ComboDatasetPyTable.fill_hdf5(h5file, data_x, data_y, node)
        h5file.close()

if __name__ == "__main__":
    ComboDatasetPyTable(start=0, stop=210000, which_set = 'train')
    ComboDatasetPyTable(start=210000, stop=-1, which_set = 'valid')
    ComboDatasetPyTable()

