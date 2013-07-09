import numpy as np
import functools

from pylearn2.space import CompositeSpace, VectorSpace, Conv2DSpace
from pylearn2.datasets.dataset import Dataset

from .faceimages import basic_7emotion_names


class EmotiwIterator(object):
    def __init__(self, dset):
        self.dset = dset
        self.cur_perm = 0
    
    def __len__(self):
        return self.dset.n_samples

    def __iter__(self):
        return self

    def __next__(self):
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
                    shuffle_rng=None, preproc=[], size=(96,96), num_channels=3, img_per_seq=3):

        self.x = np.memmap('/data/lisa/data/EmotiW/arranged_x.npy')
        self.y = np.memmap('/data/lisa/data/EmotiW/arranged_y.npy')
        self.x = self.x.view()
        self.x.shape = (len(self.y), img_per_seq, size[0], size[1], num_channels)

        if shuffle_rng is None:
            shuffle_rng = np.random.RandomState((2013, 06, 11))
        elif not isinstance(shuffle_rng, np.random.RandomState):
            shuffle_rng = np.random.RandomState(shuffle_rng)

        self.permutation = shuffle_rng.permutation(len(self.y))
        self.one_hot = one_hot

        self.space = CompositeSpace(
            Conv2DSpace(shape=size,
                          num_channels=num_channels,
                          axes=('b', 't', 0, 1, 'c')),
            VectorSpace(dim=(self.one_hot and 1 or 7)))
        self.source = ('features', 'targets')
        self.data_specs = (self.space, self.source)

        self.n_samples = len(self.x)

        @functools.wraps(Dataset.iterator)
        def iterator(self):
            return EmotiwIterator(self)

    def get_data_specs(self):
        return (self.space, self.source)
