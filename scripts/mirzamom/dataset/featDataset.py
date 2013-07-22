import numpy as np
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils import serial
from theano import config

class FeatDataset(dense_design_matrix.DenseDesignMatrix):
    def __init__(self, path, shuffle=False, one_hot=False,
                rng = None, seed = 132987, preprocessor=None):

        data = serial.load(path)
        data_x = data['x']
        data_y = data['y']

        if shuffle:
            rng = rng if rng else np.random.RandomState(seed)
            rand_idx = rng.permutation(len(data_x))
            data_x = data_x[rand_idx]
            data_y = data_y[rand_idx]

        if one_hot:
            one_hot=np.zeros((data_y.shape[0],7)).astype(config.floatX)
            for i in xrange(data_y.shape[0]):
                one_hot[i, data_y[i]] = 1.
            data_y = one_hot

        #view_converter = dense_design_matrix.DefaultViewConverter((35))
        super(FeatDataset, self).__init__(X=data_x, y=data_y)
        assert not np.any(np.isnan(self.X))

        if preprocessor is not None:
            preprocessor.apply(self, can_fit=True)



if __name__ == "__main__":
    FeatDataset('train.pkl')
