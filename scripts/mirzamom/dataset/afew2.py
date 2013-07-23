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
from emotiw.common.datasets.faces.afew2 import AFEW2ImageSequenceDataset
from emotiw.common.datasets.faces.faceimages import basic_7emotion_names
from emotiw.common.datasets.faces.facetubes import FaceTubeDataset, FaceTubeSpace
from pylearn2.utils import serial

class AFEW2FaceTubes(FaceTubeDataset):
    def __init__(self, which_set, preload_facetubes=True, one_hot=False,
                    shuffle_rng=None, preproc=[], size=(96,96), min_seq_length = 1,
                    forced_seq_size = None, source='original', prep=''):
        if which_set not in ['train', 'valid']:
            raise ValueError(
                "Unrecognized value for 'which_set': %s. "
                "Valid values are 'train' and 'valid'." % which_set)

        if prep not in ['', '_prep']:
            raise ValueError("Unrecognized preprocessing")

        if not preload_facetubes:
            raise NotImplementedError(
                "For now, we need to preload all facetubes")

        if shuffle_rng is None:
            shuffle_rng = np.random.RandomState((2013, 06, 11))
        elif not isinstance(shuffle_rng, np.random.RandomState):
            shuffle_rng = np.random.RandomState(shuffle_rng)


        features, clip_ids, targets = self.load_data(which_set, source, preproc, size, prep=prep)

        # disregard short sequences
        if min_seq_length > 1:
            _features = []
            _clip_ids = []
            _targets = []
            for feat, id, target in zip(features, clip_ids, targets):
                if feat.shape[0] >= min_seq_length:
                    _features.append(feat)
                    _clip_ids.append(id)
                    _targets.append(target)
            features = _features
            clip_ids = _clip_ids
            targets = _targets

        if forced_seq_size is not None:
            for feat in features:
                if feat.shape[0] % forced_seq_size != 0:
                    new_size = feat.shape[0] - (feat.shape[0] * forced_seq_size)
                    feat = feat[:new_size]

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
                          num_channels=1,
                          axes=('b', 't', 0, 1, 'c')),
            VectorSpace(dim=1),
            VectorSpace(dim=target_dim)))
        self.source = ('features', 'clip_ids', 'targets')
        self.data_specs = (self.space, self.source)

        self.n_samples = len(self.features)

    @staticmethod
    def load_data(which_set,source, preproc=None, size=None, prep = ''):

        if source == 'original':
            dataset = AFEW2ImageSequenceDataset(preload_facetubes=False,
                        preproc=preproc, size=size)
            train_idx, val_idx = dataset.get_standard_train_test_splits()[0]
            if which_set == 'train':
                data_idx = train_idx
            elif which_set == 'valid':
                data_idx = val_idx
            else:
                raise AssertionError

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
            return features, clip_ids, targets

        elif source == 'samira':
            path = "/data/lisa/data/faces/EmotiW/preproc/samira/KGL-AFEW/afew2_{}{}.pkl".format(which_set, prep)
            data = serial.load(path)
            return data['data_x'], data['clip_ids'], data['data_y']
        elif source == 'samira_iso':
            path = "/data/lisa/data/faces/EmotiW/preproc/samira/KGLIS-AFEWIS/afew2_{}.pkl".format(which_set)
            data = serial.load(path)
            return data['data_x'], data['clip_ids'], data['data_y']
        else:
            raise ValueError("Unknow source")



