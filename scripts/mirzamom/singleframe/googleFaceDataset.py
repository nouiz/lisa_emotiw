import numpy as np
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils import serial

class GoogleDataset(dense_design_matrix.DenseDesignMatrix):
    def __init__(self, start = None, stop = None, shuffle=False,
                rng = None, seed = 132987, center = False,
                axes=('b', 0, 1, 'c'), preprocessor=None):


        path = "/data/lisa/data/faces/GoogleDataset/Clean/latest.pkl"
        data = serial.load(path)
        data_x = data[0]
        data_y = data[1]
        assert len(data_x) == len(data_y)


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
        super(GoogleDataset, self).__init__(X=data_x, y=data_y, view_converter=view_converter)
        assert not np.any(np.isnan(self.X))

        if preprocessor is not None:
            preprocessor.apply(self)
