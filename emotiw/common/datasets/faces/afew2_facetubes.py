"""
This file defines a Pylearn2 Dataset containing the face-tubes data for afew2

All the facetubes for a given clip will be returned as different examples,
each labeled with the label for the full clip.
"""
# External dependencies
import numpy as np

# In-house dependencies
from pylearn2.space import CompositeSpace, VectorSpace

# Current project
from .afew2 import AFEW2ImageSequenceDataset
from .faceimages import basic_7emotion_names
from .facetubes import FaceTubeDataset, FaceTubeSpace


class AFEW2FaceTubes(FaceTubeDataset):
    def __init__(self, which_set, preload_facetubes=True, one_hot=False,
                    shuffle_rng=None, preproc=[], size=(96,96)):
        if which_set == 'train':
            which_set = 'Train'
        elif which_set == 'valid':
            which_set = 'Val'
        if which_set not in ['Train', 'Val']:
            raise ValueError(
                "Unrecognized value for 'which_set': %s. "
                "Valid values are 'Train' and 'Val'." % which_set)

        if not preload_facetubes:
            raise NotImplementedError(
                "For now, we need to preload all facetubes")

        if shuffle_rng is None:
            shuffle_rng = np.random.RandomState((2013, 06, 11))
        elif not isinstance(shuffle_rng, np.random.RandomState):
            shuffle_rng = np.random.RandomState(shuffle_rng)

        dataset = AFEW2ImageSequenceDataset(preload_facetubes=False,
                preproc=preproc, size=size)
        train_idx, val_idx = dataset.get_standard_train_test_splits()[0]
        if which_set == 'Train':
            data_idx = train_idx
        elif which_set == 'Val':
            data_idx = val_idx
        else:
            raise AssertionError

        if preload_facetubes:
            features = []
            clip_ids = []
            targets = []

            for idx in data_idx:
                fts = dataset.get_facetubes(idx)
                tgt = basic_7emotion_names.index(dataset.get_label(idx))
                for ft in fts:
                    features.append(ft)
                    clip_ids.append(idx)
                    targets.append(tgt)

            permutation = shuffle_rng.permutation(len(features))
            self.features = [features[i] for i in permutation]
            self.clip_ids = np.asarray(clip_ids)[permutation]
            self.targets = np.asarray(targets)[permutation][:, np.newaxis]
            self.one_hot = one_hot
            if one_hot:
                one_hot = np.zeros((self.targets.shape[0], 7), dtype='float32')
                for i in xrange(self.targets.shape[0]):
                    one_hot[i, self.targets[i]] = 1.
                self.targets = one_hot
                target_dim = 7
            else:
                target_dim = 1

        self.data = (self.features, self.clip_ids, self.targets)
        self.space = CompositeSpace((
            FaceTubeSpace(shape=size,
                          num_channels=3,
                          axes=('b', 't', 0, 1, 'c')),
            VectorSpace(dim=1),
            VectorSpace(dim=target_dim)))
        self.source = ('features', 'clip_ids', 'targets')
        self.data_specs = (self.space, self.source)

        self.n_samples = len(self.features)
