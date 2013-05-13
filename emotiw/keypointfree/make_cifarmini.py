"""
make mini CIFAR dataset, which is a subset of the CIFAR training set, and save it locally 

running this script will generate the following four txt files: 
cifar_mini_images_train.txt
cifar_mini_images_test.txt
cifar_mini_labels_train.txt
cifar_mini_labels_test.txt

you need access to the original CIFAR dataset to generate the mini cifar dataset. you may have to adjust the datadir string below, which has to point to the original CIFAR dataset. 
"""



import os
HOME = os.environ['HOME']

import numpy
import pylab

patchsize = 12
datadir = HOME+'/research/data/cifar/cifar-10-batches-py'

numpy.random.seed(1)

numtrain = 2000
numtest = 2000


def onehot(x,numclasses=None):
    """ Convert integer encoding for class-labels (starting with 0 !)
        to one-hot encoding. 
      
        If numclasses (the number of classes) is not provided, it is assumed 
        to be equal to the largest class index occuring in the labels-array + 1.
        The output is an array who's shape is the shape of the input array plus
        an extra dimension, containing the 'one-hot'-encoded labels. 
    """
    if x.shape==():
        x = x[None]
    if numclasses is None:
        numclasses = x.max() + 1
    result = numpy.zeros(list(x.shape) + [numclasses], dtype="int")
    z = numpy.zeros(x.shape)
    for c in range(numclasses):
        z *= 0
        z[numpy.where(x==c)] = 1
        result[...,c] += z
    return result


def crop_patches(image, keypoints, patchsize):
    patches = numpy.zeros((len(keypoints), patchsize**2))
    for i, k in enumerate(keypoints):
        patches[i, :] = image[k[0]-patchsize/2:k[0]+patchsize/2, k[1]-patchsize/2:k[1]+patchsize/2].flatten()
    return patches


def dispims_color(M, border=0, bordercolor=[0.0, 0.0, 0.0], *imshow_args, **imshow_keyargs):
    """ Display an array of rgb images. 

    The expected array shape is 
    numimages x numpixelsY x numpixelsX x 3
    """
    bordercolor = numpy.array(bordercolor)[None, None, :]
    numimages = len(M)
    M = M.copy()
    for i in range(M.shape[0]):
        M[i] -= M[i].flatten().min()
        M[i] /= M[i].flatten().max()
    height, width, three = M[0].shape
    assert three == 3
    n0 = numpy.int(numpy.ceil(numpy.sqrt(numimages)))
    n1 = numpy.int(numpy.ceil(numpy.sqrt(numimages)))
    im = numpy.array(bordercolor)*numpy.ones(((height+border)*n1+border,(width+border)*n0+border, 1),dtype='<f8')
    for i in range(n0):
        for j in range(n1):
            if i*n1+j < numimages:
                im[j*(height+border)+border:(j+1)*(height+border)+border,
                   i*(width+border)+border:(i+1)*(width+border)+border,:] = numpy.concatenate((
                  numpy.concatenate((M[i*n1+j,:,:,:],
                         bordercolor*numpy.ones((height,border,3),dtype=float)), 1),
                  bordercolor*numpy.ones((border,width+border,3),dtype=float)
                  ), 0)
    pylab.imshow(im, *imshow_args, **imshow_keyargs)
    pylab.show()


images = numpy.load(datadir+'/data_batch_1')['data'].astype("float").reshape(10000, 3, 1024)
labels = numpy.array(numpy.load(datadir+'/data_batch_1')['labels']).astype("int32")
images = images.reshape(10000, 3*1024)
images = numpy.uint8(images)

trainimages = images[:numtrain]
testimages = images[numtrain:numtrain+numtest]
trainlabels = onehot(labels[:numtrain])
testlabels = onehot(labels[numtrain:numtrain+numtest])


numpy.savetxt("cifar_mini_images_train.txt", trainimages, fmt="%u")
numpy.savetxt("cifar_mini_labels_train.txt", trainlabels, fmt="%u")
numpy.savetxt("cifar_mini_images_test.txt", testimages, fmt="%u")
numpy.savetxt("cifar_mini_labels_test.txt", testlabels, fmt="%u")



