from imagenet import *

def test1():

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


    pipeline = preprocessing.Pipeline()
    pipeline.items.append(preprocessing.GlobalContrastNormalizationPyTables())
    pipeline.items.append(preprocessing.LeCunLCN((256, 256), channels=[0], kernel_size=7))

    # apply preprocessing to train
    train.apply_preprocessor(pipeline, can_fit = True)
    train.view_shape()

    #testing
    batch_size = 10
    num_batches = 1
    mode = SequentialSubsetIterator
    targets1 = []
    targets2 = []

    for data in train.iterator(batch_size=batch_size, num_batches=num_batches, mode=mode,targets=False):
	imgplot = plt.imshow(data.reshape((256,256)))
	plt.gray()
        plt.show()
	#print data.shape


test1()
