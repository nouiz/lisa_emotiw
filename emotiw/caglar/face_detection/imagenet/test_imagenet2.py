from emotiw.caglar.imagenet import Imagenet
from emotiw.caglar.data_arranger import ExtractRandomPatches

from pylearn2.datasets import preprocessing
from pylearn2.utils.iteration import SequentialSubsetIterator

def test1():
    path_org = '/Tmp/gulcehrc/imagenet_256x256_filtered.h5'
    path = '/Tmp/gulcehrc/imagenetTemp2.h5'
    train = Imagenet(which_set = 'train',
	    path = path,
	    path_org = path_org,
            size_of_receptive_field=(8, 8),
            center=True,
	    scale=True,
            start=0,
            stop=1000,
	    imageShape =(256,256),
            mode='a',
            axes=('b', 0, 1, 'c'),
            preprocessor=None)

    import ipdb; ipdb.set_trace()

    randomPatches = ExtractRandomPatches(patch_shape=(8, 8), num_patches=100)
    data = randomPatches.apply(train)
    #import ipdb; ipdb.set_trace()

test1()
