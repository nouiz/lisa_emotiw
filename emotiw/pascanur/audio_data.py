"""
TODO:
    (1) More documentation
    (2) Unit Tests
    (3) Build a simple example (the bidirectional RNN !?)
    (4) Collect some feedback
"""

import cPickle
import glob
import numpy
import os
import sys

classes = ['Angry',
           'Disgust',
           'Fear',
           'Happy',
           'Neutral',
           'Sad',
           'Surprise']

class ListSequence(object):
    def __init__(self,
            path='../audio_features',
            pca=True,
            subset='full',
            which='train',
            one_hot = True,
            nbits = 32):
        """
        Class storing dataset as a list of sequences, suitable for
        sequences of different lengths.
        :param path: String
            Path to the dataset; If it points to a pickle file, just load
            that pickle file
        :param pca: Bool
            If we want the PCA version of the dataset or not
        :param subset: String
            The subset of features (one of 'raw', 'minimal', 'full')
        :param which: String
            Which dataset to load (one of 'train' or 'valid')
        :param one_hot: Bool
            If True, the target is given as a one-hot vector, otherwise as
            an index
        :param nbits: int
            Either 32 or 64. If 32 we will use float32/int32 otherwise we
            use float64/int64
        """
        if '.pkl' == path[-4:] and os.path.isfile(path):
            data = cPickle.load(open(path))
            self.data_x = data[0]
            self.data_y = data[1]
            self.index = -1
        else:
            assert which in ('train', 'valid')
            assert subset in ('raw', 'minimal', 'full')
            assert nbits in (32, 64)
            if which == 'train':
                # This is the convention / folder name picked by Mehdi or
                # the ones who made the data
                which = 'Train'
            else:
                which = 'Val'
            if pca:
                suffix = '%s.pca.pkl' % subset
            else:
                suffix = '%s.pkl' % subset
            audiofiles = glob.glob('%s/*/*/*.%s' % (path, suffix))

            data_x = []
            data_y = []
            for audiofile in audiofiles:
                data = cPickle.load(open(audiofile))
                target = [(cls in audiofile) for cls in classes]
                if not one_hot:
                    target = numpy.max(target)
                if which in audiofile:
                    data_x.append(
                            numpy.array(data,
                                dtype='float%d'%nbits))
                    data_y.append(
                            numpy.array(target,
                                dtype='int%d'%nbits))
            self.data_x = data_x
            self.data_y = data_y
        self.nbits = nbits
        self.n_examples = len(self.data_x)
        # hack .. should be replaced once a proper pylearn2 dataset class
        # is created. It is meant to keep track if parameters for the
        # iterator are set or not. I do not want to construct another class
        # for the iterator for now to keep the code short and easy to
        # parse, but the final version will do that (as any dataset in
        # pylearn 2 does)
        self.__iterator_set__ = False

    def set_iterator(self,
            order = 'sequence',
            rng = None):
        self.order = order
        if rng is None:
            rng = numpy.random.RandomState([123,43,53])
        self.rng = rng
        self.index = -1
        self.perm = self.rng.permutation(self.n_examples)
        self.__iterator_set__ = True

    def __iter__(self):
        if not self.__iterator_set__:
            self.set_iterator()
        return self

    def next():
        return self.get_example()

    def get_example(self):
        if not self.__iterator_set__:
            self.set_iterator()
        self.index += 1
        if self.order == 'rand':
            if self.index == self.n_examples:
                self.perm = self.rng.permutation(self.n_examples)
                self.index = 0
                index = self.perm[self.index]
            else:
                index = self.perm[self.index]
        else:
            if self.index == self.n_examples:
                self.index = 0
            index = self.index
        return self.data_x[index], self.data_y[index]

    def export_dense_format(self,
            sequence_length= 10,
            overlap = 5,
            batchsize = 32):
        """
        Clip sequences in batches of size `batchsize`, where each sequence
        has fixed length `sequence_length`. This subsequences are taken
        from the original data by cropping every `overlap`,
        `sequence_length` consecutive steps.
        """
        final_data_x = []
        final_data_y = []
        for sample_x, sample_y in zip(self.data_x, self.data_y):
            n_steps = sample_x.shape[0]
            for k in xrange(0,n_steps-sequence_length,overlap):
                final_data_x.append(sample_x[k:k+sequence_length])
                final_data_y.append(sample_y)

        final_data_x = numpy.array(final_data_x,
                dtype='float%d'%self.nbits)
        final_data_y = numpy.array(final_data_y,
                dtype='int%d'%self.nbits)

        final_data_x = numpy.transpose(final_data_x, [1,0,2])
        return DenseSequences(
                data=(final_data_x, final_data_y),
                nbits=self.nbits)


class DenseSequences(object):
    def __init__(self, path=None, data=None, nbits = 32):
        assert (path is None) or (data is None)
        assert nbits in (32, 64)
        if path is not None:
            data = numpy.load(path)
            self.data_x = data['x']
            self.data_y = data['y']
        else:
            self.data_x = data[0]
            self.data_y = data[0 ]
        self.nbits = nbits
        self.n_examples = self.data_x.shape[1]
        self.__iterator_set__ = False


    def set_iterator(self,
            order = 'sequence',
            rng = None,
            batchsize = 32):
        self.order = order
        if rng is None:
            rng = numpy.random.RandomState([123,43,53])
        self.rng = rng
        self.index = -1
        self.batchsize = batchsize
        self.n_batches = self.n_examples // self.batchsize
        self.offset_max = self.n_examples % self.batchsize
        self.perm = self.rng.permutation(self.n_batches)
        self.offset = 0
        self.__iterator_set__ = True

    def __iter__(self):
        # TODO export a different iterator class
        if not self.__iterator_set__:
            self.set_iterator()
        return self

    def next():
        return self.get_example()

    def get_batch(self):
        if not self.__iterator_set__:
            self.set_iterator()
        self.index += 1
        if self.order == 'rand':
            if self.index == self.n_batches:
                self.perm = self.rng.permutation(self.n_batches)
                self.offset = self.rng.randint(self.offset_max)
                self.index = 0
                index = self.perm[self.index]
            else:
                index = self.perm[self.index]
        else:
            if self.index == self.n_examples:
                self.index = 0
            index = self.index
        start = index * self.batchsize + self.offset
        end = (index + 1) * self.batchsize + self.offset
        return self.data_x[start:end], self.data_y[start:end]


    def save(self, filename):
        numpy.savez(filename, x = self.data_x, y = self.data_y)




if __name__=='__main__':
    # Simple test of constructing the objects
    # Encoded for tikuanyin
    liter = ListSequence(
            path='/home/pascanur/data/EmotiW/audio_features',
            pca=True,
            subset='full',
            which='train',
            one_hot = True,
            nbits = 32)
    diter = liter.export_dense_format()

