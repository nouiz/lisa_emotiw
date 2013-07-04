import numpy

from pylearn2.datasets.dataset import Dataset
from pylearn2.utils.iteration import resolve_iterator_class
from pylearn2.space import CompositeSpace, VectorSpace, Conv2DSpace


class EmotiwArrangerIter(object):
    """
    Iterator for the data arranger class
    """
    _default_seed = (17, 2, 946)

    def __init__(self,
                 inst,
                 iter_mode=None,
                 batch_size=None,
                 topo=None,
                 targets=True,
                 rng=_default_seed):

        self.inst = inst
        self.n_face_per_batch = [int(batch_size * self.weights[i]) for i in xrange(len(self.weights))]
        self.batch_size = batch_size

        self.iterators = [inst.dataset[i].iterator(mode=iter_mode,
                                                   batch_size=self.n_face_per_batch[i],
                                                   topo=topo,
                                                   targets=targets) for i in xrange(len(inst.dataset))]
        self.rng = numpy.random.RandomState(rng)

        self.total_n_exs = self.inst.total_n_exs
        self.topo = topo
        self.targets = targets
        self.iter_mode = self.inst.subset_iterator

        self.img_idx = [0]*len(self.inst.datasets)

    @staticmethod
    def fisher_yates_shuffle(imgs, tgts):
        length = imgs.shape[0]
        for i in xrange(length):
            j = numpy.random.randint(0, length)
            imgs[i], imgs[j] = imgs[j], imgs[i]
            tgts[i], tgts[j] = tgts[j], tgts[i]
        return imgs, tgts

    @staticmethod
    def _mix_faces(imgs, tgts):
        #TODO: verify correctness. The original shuffled tgts and returned
        #face_tgts.
        face_imgs = numpy.concatenate(tuple(imgs))
        face_tgts = numpy.concatenate(tuple(tgts))
        EmotiwArrangerIter.fisher_yates_shuffle(face_imgs, face_tgts)
        return face_imgs, face_tgts

    def _pick_idx_given_rnd(self, rnd, weights, num_left):
        cumul_sum = [sum(weights[0:i]) * int(bool(num_left[i])) for i in xrange(len(weights))]
        #0 if none left, cumulative sum otherwise.

        #return the index if:
        # - the cumulative sum is larger than rnd (we search from left to
        # right)
        # - all the remaining elements are 0
        #
        # In other words, if all the elements from a given set
        # have been selected for this batch, do as if it didn't exist.
        # If this is the last non-0 elements (left), then this is the one we
        # should sample from.
        for idx, x in enumerate(cumul_sum):
            if x > rnd or (idx < len(cumul_sum)-1 and sum(cumul_sum[idx+1:])==0):
                return idx

    def next(self):
        images, targets, imgnos = [], [], []
        next_index = self.iter_mode.next()
        batch_idx = [0]*len(self.inst.datasets)
        vals = [it.next() for it in self.iterators]

        die_values = self.rng.rand(self.batch_size)

        for i in xrange(next_index.start, next_index.stop):
            die_value = die_values[sum(batch_idx)]
            pick_from = self._pick_idx_given_rnd(die_value, self.weights,
                    numpy.asarray(self.n_face_per_batch) - numpy.asarray(batch_idx))
            self.inst.face_table.insert_face_table(imgno=self.img_idx[pick_from])
            batch_idx[pick_from] += 1
            self.img_idx[pick_from] += 1
            the_vals = vals[pick_from]
            the_batch_idx = batch_idx[pick_from]
            #XXX

            images.append(the_vals[0][the_batch_idx])
            targets.append(the_vals[1][the_batch_idx])

            img_no = self.img_idx[pick_from]

            imgnos.append(img_no)

        images = numpy.asarray(images)
        targets = numpy.asarray(targets)
        return images, targets  # , imgnos

    def __iter__(self):
        return self


class EmotiwArranger(Dataset):
    """
    This dataset takes N dataset objects, and combines them online.
    """
    def __init__(self,
                 datasets,
                 weights):

        assert len(weights) == len(datasets)

        self.datasets = datasets
        total_weight = float(sum(weights))
        self.weights = [w/total_weight for w in weights]
        self.total_n_exs = sum([len(dataset) for dataset in datasets])

    def get_space(self):
        return CompositeSpace(Conv2DSpace(shape=(96, 96),
                                          num_channels=3,
                                          axes=('b', 't', 0, 1, 'c')),
                              VectorSpace(1))

    def get_data_specs(self):
        return (self.get_space(), self.get_data_source())

    def get_data_source(self):
        return ('features', 'targets')

    def get_data(self):
        raise NotImplementedError('get_data cannot be implemented for EmotiwArranger')

    def get_batch_design(self,
                         batch_size,
                         include_labels=False):
        """
        Method inherited from the Dataset.
        """
        iterator = self.iterator(mode='sequential',
                                 batch_size=batch_size,
                                 num_batches=None,
                                 topo=None)
        return iterator.next()

    def get_batch_topo(self, batch_size):
        """
        Method inherited from the Dataset.
        """
        raise NotImplementedError('Not implemented for sparse dataset')

    def iterator(self,
                 mode=None,
                 batch_size=None,
                 num_batches=None,
                 topo=None,
                 targets=None,
                 rng=None):
        """
        Method inherited from the Dataset.
        """
        self.mode = mode
        self.batch_size = batch_size
        self._targets = targets
        mode = resolve_iterator_class(mode)

        self.subset_iterator = mode(self.total_n_exs,
                                    batch_size,
                                    num_batches,
                                    rng=None)

        return EmotiwArrangerIter(self,
                                  mode,
                                  batch_size=batch_size)
