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
            path = '/Tmp/zumerjer/'
            mode = 'r'
        else:
            mode = 'r'


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
        view_converter = dense_design_matrix.DefaultViewConverter((96, 96, 3), axes)
        super(ComboDatasetPyTable, self).__init__(X=data.X, y=data.y, view_converter=view_converter)
        assert not np.any(np.isnan(self.X))

        if preprocessor is not None:
            preprocessor.apply(self)


    @staticmethod
    def make_data(path, shuffle = True, rng = None, seed = 132987, which_set = 'all', start = None, stop = None):
        

        orig_path = '/data/lisa/data/faces/EmotiW/preproc/'
        orig_path = '/Tmp/zumerjer/'
        file_n = "{}{}.h5".format(orig_path, which_set)
        data_x = np.memmap(orig_path + 'complete_train_x.npy', mode='r', dtype='uint8')
        data_y = np.memmap(orig_path + 'complete_train_y.npy', mode='r', dtype='float32')
        numSamples = len(data_x)/(96*96*3)
        data_x = data_x.reshape((numSamples, -1))
        data_y = data_y.reshape((numSamples, -1))

        assert numSamples == len(data_y)
        data_x = data_x.reshape(-1,96,96,3)[:,::-1,:,:] #(b, 0, 1, c)
        data_y = data_y.reshape(-1, 98, 2)[:,:,::-1].reshape(-1, 98*2) #(num_samples, 2*num_targets)
        import Image
        print 'going to print image'
        img = data_x[0,:].reshape((96,96,3)).copy()
        for idx, _ in enumerate(data_y[0,:]):
            if idx % 2 == 1:
                img[round(data_y[0,idx-1]), round(data_y[0,idx])] = [0,0,0]
        Image.fromarray(img).show()
#        data_x = data_x.reshape((data_y.shape[0], 48 * 48))

#        one_hot = np.zeros((len(data_y),7), dtype=np.float32)
 #       one_hot[np.asarray(range(len(data_y))), data_y] = 1.
  #      data_y = one_hot

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

        h5file, node = ComboDatasetPyTable.init_hdf5(file_n, (data_x.shape, data_y.shape))
        ComboDatasetPyTable.fill_hdf5(h5file, data_x, data_y, node)
        h5file.close()

if __name__ == "__main__":
    # ComboDatasetPyTable(start=0, stop=210000, which_set = 'train')
    ComboDatasetPyTable()
    #ComboDatasetPyTable(start=210000, stop=-1, which_set = 'valid')
    #ComboDatasetPyTable()

