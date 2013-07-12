from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

import numpy as np
from collections import defaultdict
import os
class FeaturesDataset(DenseDesignMatrix):

    def __init__(self,
            start = None,
            stop = None,
            base_path=None,
            which_set = None,
            features_path = None,
            one_hot = True,
            labels_path = None,
            gcn = None,
            eps = 1e-16,
            standardize = False,
            shuffle = False):

        assert features_path is not None
        assert labels_path is not None

        if base_path is not None:
            features_path = os.path.join(base_path, features_path)
            labels_path = os.path.join(base_path, labels_path)

        X = np.load(features_path)
        y = np.load(labels_path)

        if start is not None:
            assert isinstance(start, int)
            assert isinstance(stop, int)
            assert start >= 0
            assert start < stop
            assert stop <= X.shape[0]
            X = X[start:stop]
            y = y[start:stop]

        if gcn:
            assert isinstance(gcn,float)
            X = (X.T - X.mean(axis=1)).T
            X = (X.T / np.sqrt(np.square(X).sum(axis=1))).T
            X *= gcn

        if standardize:
            X = (X - np.mean(X, axis=0)) / np.maximum(np.std(X, axis=0), eps)


        if one_hot:
            dummy_y = np.zeros((y.shape[0], 7))
            for i,t in enumerate(y):
                dummy_y[i, t] = 1.
            y = dummy_y

        super(FeaturesDataset, self).__init__(X=X,
                                              y=y)

