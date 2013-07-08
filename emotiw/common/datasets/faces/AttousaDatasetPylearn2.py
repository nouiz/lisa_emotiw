from pylearn2.datasets import Dataset
from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace


class AttousaDataset(Dataset):
    def get_data_specs(self):
        return (self.get_space(), self.get_source())

    def get_space(self):
        return CompositeSpace(
                Conv2DSpace(
                    shape=(48 , 48),
                    num_channels=3,
                    axes=('b', 't', 0, 1, 'c')),
                VectorSpace(1))

    def get_source(self):
        return ('features', 'targets')

    def __init__(self):
        pass
