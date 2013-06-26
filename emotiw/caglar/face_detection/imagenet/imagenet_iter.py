import theano
import numpy
import h5py


class DatasetIter(object):
    def __init__(self, file, buffer_size=100000, dtype=None):
        self.file = file
        self.dtype = dtype if dtype != None else theano.config.floatX
        self.buffer_size = buffer_size

        self.current_index = 0
        self.shape = self.file["x"].shape

        self.buffer = None
        self.buffer_ptr = 0

    def minibatch_iterator(self, batch_size):
        """
        Returns an iterator that reads the dataset in chunks no bigger than
        batch_size.
        """
        read = 0
        self.current_index = 0
        while read < self.shape[0]:
            res = self.read(batch_size)
            read += res[0].shape[0]
            yield res

    def read(self, num):
        """
        Read a number of rows of features.
        """
        if self.buffer == None:
            self._fill_buffer()

        stop = min(self.buffer_ptr+num, self.buffer['features'].shape[0])

        features = self.buffer['features'][self.buffer_ptr:stop]
        targets = self.buffer['targets'][self.buffer_ptr:stop]

        self.buffer_ptr += features.shape[0]

        if self.buffer_ptr >= self.buffer['features'].shape[0]:
            self.buffer = None

        return numpy.asarray(features, dtype=self.dtype), (targets - 1)

    def _fill_buffer(self):
        """
        Fill the read buffer.
        """
        self.buffer = None

        features = self.file['x'][self.current_index:min(self.current_index+self.buffer_size,
            self.shape[0])]
        targets = self.file['y'][self.current_index:min(self.current_index+self.buffer_size,
            self.shape[0])]

        self.current_index = self.current_index + self.buffer_size

        if self.current_index >= self.shape[0]:
            self.current_index = 0

        self.buffer = {
            'features' : features,
            'targets' : targets,
        }
        self.buffer_ptr = 0

