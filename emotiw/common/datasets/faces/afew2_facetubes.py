"""
This file defines a Pylearn2 Dataset containing the face-tubes data for afew2

All the facetubes for a given clip will be returned as different examples,
each labeled with the label for the full clip.
"""
# In-house dependencies
from pylearn2.datasets import CompositeSpace, VectorSpace

# Current project
from .afew2 import AFEW2ImageSequenceDataset
from .faceimages import basic_7emotion_names
from .facetubes import FaceTubeDataset, FaceTubeSpace


class AFEW2FaceTubes(FaceTubeDataset):
    def __init__(self, which_set, preload_facetubes=True):
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

        dataset = AFEW2ImageSequenceDataset(preload_facetubes=False)
        train_idx, val_idx = dataset.get_standard_train_test_splits()[0]
        if which_set == 'Train':
            data_idx = train_idx
        elif which_set == 'Val':
            data_idx = val_idx
        else:
            raise AssertionError

        if preload_facetubes:
            self.features = []
            self.clip_ids = []
            self.targets = []

            for idx in data_idx:
                fts = dataset.get_facetubes(idx)
                tgt = basic_7emotion_names.index(dataset.get_label(idx))
                for ft in fts:
                    self.features.append(fts)
                    self.clip_ids.append(idx)
                    self.targets.append(tgt)

        self.data = (self.features, self.clip_ids, self.targets)
        self.space = CompositeSpace((
            FaceTubeSpace(shape=(96, 96),
                          num_channels=3,
                          axes=('b', 't', 0, 1, 'c')),
            VectorSpace(dim=1),
            VectorSpace(dim=1)))
        self.source = ('features', 'clip_ids', 'targets')
        self.data_specs = (self.space, self.source)

        self.n_samples = len(self.features)
