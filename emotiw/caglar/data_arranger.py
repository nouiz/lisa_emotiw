from face_bbox import FaceBBox
from emotiw.caglar.imagenet import Imagenet
from pylearn2.utils import iteration

class DataArrangerIter(iteration.FiniteDatasetIterator):
    """
    Iterator for the data arranger class
    """
    def __init__(self, inst, iter_mode=None, batch_size=None, topo=None, targets=False):
        self.inst = inst
        self.face_dataset_iter = inst.face_dataset.iterator(mode=iter_mode, batch_size=batch_size,
                topo=topo, targets=targets)

        self.nonface_dataset_iter = inst.nonface_dataset.iterator(mode=iter_mode, batch_size=batch_size,
                topo=topo, targets=targets)

        self.face_ratio = face_ratio
        self.total_n_exs = total_n_exs
        self.topo = topo
        self.targets = targets

    def next(self):
        pass

class DataArranger(Dataset):

    def __init__(self, face_dataset, nonface_dataset, face_ratio, face_table, total_n_exs):
        self.face_dataset = face_dataseet
        self.nonface_dataset = nonface_dataset
        self.face_ratio = face_ratio
        self.total_n_exs = total_n_exs
        self.face_table = face_table

    def get_design_matrix(self):
        return self.sparse_matrix

    def get_batch_design(self, batch_size, include_labels=False):
        """
        Method inherited from the Dataset.
        """
        self.iterator(mode='sequential', batch_size=batch_size, num_batches=None, topo=None)
        return self.next()

    def get_batch_topo(self, batch_size):
        """
        Method inherited from the Dataset.
        """
        raise NotImplementedError('Not implemented for sparse dataset')

    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None):
        """
        Method inherited from the Dataset.
        """
        self.mode = mode
        self.batch_size = batch_size
        self._targets = targets
        mode = resolve_iterator_class(mode)
        self.subset_iterator = mode(self.data_n_rows,
                                            batch_size, num_batches, rng=None)
        return DataArrangerIter(self, mode, batch_size)
