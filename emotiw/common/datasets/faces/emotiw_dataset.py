import numpy as np
import functools

from emotiw.common.datasets.faces.facetubes import FaceTubeSpace
from pylearn2.space import CompositeSpace, VectorSpace
from pylearn2.datasets.dataset import Dataset

from .faceimages import basic_7emotion_names

import os


class EmotiwIterator(object):
    def __init__(self, dset):
        self.dset = dset
        self.cur_perm = 0
    
    def __len__(self):
        return self.dset.n_samples

    def __iter__(self):
        return self

    def next(self):
        if self.cur_perm >= len(self.dset.permutation):
            raise StopIteration()

        next_idx = self.dset.permutation[self.cur_perm]
        feature = self.dset.x[next_idx]
        target = self.dset.y[next_idx]

        if self.dset.one_hot:
            one_hot = np.zeros((7,), dtype=np.float32)
            one_hot[target] = 1.
            target = one_hot

        self.cur_perm += 1

        return (feature, target)


class EmotiwDataset(Dataset):
    def __init__(self, one_hot=False,
                    shuffle_rng=None, preproc=[], size=(48,48), num_channels=1, img_per_seq=3,
                    path=None):

        if path is None:
            path = '/data/lisa/data/faces/EmotiW/preproc/arranged_data'

        self.x = np.memmap(path + '_x.npy', mode='r', dtype='float32')
        self.y = np.memmap(path + '_y.npy', mode='r', dtype='uint8')
        self.y = self.y.view()
        self.y.shape = (len(self.y)/(img_per_seq), img_per_seq, 1)

        self.x = self.x.view()
        self.x.shape = (len(self.y), img_per_seq, size[0], size[1], num_channels)
        
        if shuffle_rng is None:
            shuffle_rng = np.random.RandomState((2013, 06, 11))
        elif not isinstance(shuffle_rng, np.random.RandomState):
            shuffle_rng = np.random.RandomState(shuffle_rng)

        self.permutation = shuffle_rng.permutation(len(self.y))
        self.one_hot = one_hot

        self.space = CompositeSpace(
            (FaceTubeSpace(shape=size,
                          num_channels=num_channels,
                          axes=('b', 't', 0, 1, 'c')),
            VectorSpace(dim=(self.one_hot and 7 or 1))))
        self.source = ('features', 'targets')
        self.data_specs = (self.space, self.source)

        self.n_samples = len(self.y)

    @functools.wraps(Dataset.iterator)
    def iterator(self):
        return EmotiwIterator(self)

    def get_data_specs(self):
        return self.data_specs
