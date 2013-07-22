import numpy as np
from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets.tfd import TFD
from pylearn2.utils import serial

class GoogleTFDDataset(dense_design_matrix.DenseDesignMatrix):
    def __init__(self, start = None, stop = None, shuffle=False,
                rng = None, seed = 132987, center = False,
                scale = False,
                axes=('b', 0, 1, 'c'), preprocessor=None,
                which_ds = 'kaggle'):

        data_x, data_y = self.load_data(which=which_ds, center=center, scale=scale)
        tfd = TFD('train', one_hot=1, scale=scale)
        data_x = np.concatenate((data_x, tfd.X))
        data_y = np.concatenate((data_y, tfd.y))
        tfd = TFD('valid', one_hot=1, scale=scale)
        data_x = np.concatenate((data_x, tfd.X))
        data_y = np.concatenate((data_y, tfd.y))

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

        if center:
            data_x -= 0.5

        self.axes = axes
        view_converter = dense_design_matrix.DefaultViewConverter((48, 48, 1), axes)
        super(GoogleTFDDataset, self).__init__(X=data_x, y=data_y, view_converter=view_converter)
        assert not np.any(np.isnan(self.X))

        if preprocessor is not None:
            preprocessor.apply(self)

    @staticmethod
    def load_data(which='original', center=False, scale=False):
        if which == 'original':
            path = "/data/lisa/data/faces/GoogleDataset/Clean/latest.pkl"
            data = serial.load(path)
            data_x = data[0]
            data_y = data[1]
            assert len(data_x) == len(data_y)
            if center:
                data_x -= 0.5
        elif which == 'kaggle':
            path="/data/lisa/data/faces/EmotiW/preproc/samira/KGL-AFEW/"
            data_x = serial.load(path + 'train_kaggle_x.npy')
            data_y = serial.load(path + 'train_kaggle_y.npy')
            assert len(data_x) == len(data_y)
            if scale:
                data_x /= 255.
                if center:
                    data_x -= .5
            elif center:
                data_x -= 127.5

        one_hot = np.zeros((data_y.shape[0], 7), dtype='float32')
        for i in xrange(data_y.shape[0]):
            one_hot[i,data_y[i]] = 1.
        data_y = one_hot

        return data_x.reshape(data_x.shape[0], 48*48).astype('float32'), data_y



if __name__ == "__main__":
    #GoogleDataset(which_ds='original')
    GoogleTFDDataset(which_ds='kaggle', scale=1)
