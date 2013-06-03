import os
import gc
import warnings
try:
        import tables
except ImportError:
        warnings.warn("Couldn't import tables, so far SVHN is "
                            "only supported with PyTables")
import numpy
from theano import config
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils.serial import load
from pylearn2.utils.string_utils import preprocess
from pylearn2.datasets import preprocessing

class Imagenet(dense_design_matrix.DenseDesignMatrixPyTables):
    mapper = {
        'train': 0,
        'test':  1,
        'valid': 2
    }

    def __init__(self,
	    which_set,
            path,
            center,
	    scale,
            start,
            stop,
	    imageShape = (256,256),
            mode='r',
            axes=('b', 0, 1, 'c'),
            preprocessor=None):

        assert which_set in self.mapper.keys()
	self.which_set = which_set
        self.__dict__.update(locals())
        del self.self

        self.mode = mode
	w,h = imageShape

	path_org = '/Tmp/gulcehrc/imagenet_256x256_filtered.h5'
        h5file_org = tables.openFile(path_org, mode = 'r')

	assert start != None and stop != None
	x_data = h5file_org.getNode('/', 'x')
	y_data = h5file_org.getNode('/', 'y')

	#create new h5file at the specified path
	self.h5file = tables.openFile(path, mode = mode, title = "ImageNet Dataset")
	if self.h5file.__contains__('/Data'):
		self.h5file.removeNode('/', "Data", 1)

	data = self.h5file.createGroup(self.h5file.root, "Data", "Data")
	atom = tables.Float32Atom() if config.floatX == 'float32' else tables.Float64Atom()
	filters = tables.Filters(complib='blosc', complevel=5)
	
	x = self.h5file.createCArray(data, 'X', atom = atom, shape = ((stop-start, 256*256)),
                                title = "Data values", filters = filters)
	y = self.h5file.createCArray(data, 'y', atom = atom, shape = ((stop-start, )),
                                title = "Data targets", filters = filters)

	#copy data from original h5file
	x[:] = x_data[start:stop]
	y[:] = y_data[start:stop]
	self.h5file.flush()
	    
	    
        # rescale or center if permitted
        if center and scale:
            data.X[:] -= 127.5
            data.X[:] /= 127.5
        elif center:
            data.X[:] -= 127.5
        elif scale:
            data.X[:] /= 255.

        view_converter = dense_design_matrix.DefaultViewConverter((w, h, 1),
                                                                        axes)
        super(Imagenet, self).__init__(X = data.X, y = None,
                                    view_converter = view_converter)
        if preprocessor:
            can_fit =False 
            if which_set in ['train']:
                can_fit = True
            preprocessor.apply(self, can_fit)


def test_works():
    path = '/Tmp/aggarwal/imagenetTemp.h5'
    train = Imagenet(which_set = 'train',
	    path = path,
            center=True,
	    scale=True,
            start=0,
            stop=1000,
	    imageShape =(256,256),
            mode='a',
            axes=('b', 0, 1, 'c'),
            preprocessor=None)
    

    pipeline = preprocessing.Pipeline()
    pipeline.items.append(preprocessing.GlobalContrastNormalizationPyTables())
    pipeline.items.append(preprocessing.LeCunLCN((256, 256), channels=[0], kernel_size=7))

    # apply preprocessing to train
    train.apply_preprocessor(pipeline, can_fit = True)


if __name__ == '__main__':
	test_works()
	
	
