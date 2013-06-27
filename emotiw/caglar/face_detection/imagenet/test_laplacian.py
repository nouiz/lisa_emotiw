from emotiw.caglar.laplacian import LaplacianPyramid

from imagenet import Imagenet
from pylearn2.utils.iteration import SequentialSubsetIterator

def test_works():
    nlevels = 5
    subsampling_rates=[2, 2, 2, 2, 2]
    img_shape = (256, 256)
    paths = ['/Tmp/aggarwal/imagenetl1.h5',
             '/Tmp/aggarwal/imagenetl2.h5',
             '/Tmp/aggarwal/imagenetl3.h5',
             '/Tmp/aggarwal/imagenetl4.h5',
             '/Tmp/aggarwal/imagenetl5.h5']


    obj = LaplacianPyramid(nlevels=nlevels,
                           subsampling_rates=subsampling_rates,
                           img_shape = img_shape,
                           paths = paths,
                           preprocess =True)

    path_org = '/Tmp/gulcehrc/imagenet_256x256_filtered.h5'
    path = '/Tmp/aggarwal/imagenetTemp.h5'
    train = Imagenet(which_set = 'train',
            path = path,
            path_org = path_org,
            center=True,
            scale=True,
            start=0,
            stop=1000,
            imageShape =(256,256),
            mode='a',
            axes=('b', 0, 1, 'c'),
            preprocessor=None)

    #testing
    batch_size = 10
    num_batches = 1
    mode = SequentialSubsetIterator
    targets1 = []
    targets2 = []

    for data in train.iterator(batch_size=batch_size, num_batches=num_batches, mode=mode,targets=False):
        obj.apply(data, canfit=True)

if __name__ == '__main__':
    test_works()

