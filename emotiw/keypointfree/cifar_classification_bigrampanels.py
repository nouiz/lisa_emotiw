import numpy 
import numpy.random
import pylab
import dispims_color
import logreg 
import harris 
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import patchpair_encoder 
import train

numclasses = 10 
patchsize = 6
stride = 3
border = patchsize + patchsize/2
numhid = 50

datadir = '/data/lisa/data/cifar10/cifar-10-batches-py'                                                 


rng  = numpy.random.RandomState(1)


def rgb2gray(image):
  return numpy.sum(image*numpy.array([0.3,0.59,0.11])[None,None,:],2)


def onehot(x, numclasses=None):
    if numclasses is None:
        numclasses = x.max() + 1
    result = numpy.zeros(list(x.shape) + [numclasses], dtype="uint8")
    z = numpy.zeros(x.shape, dtype="uint8")
    for c in range(numclasses):
        z *= 0
        z[numpy.where(x==c)] = 1
        result[...,c] += z
    return result


def crop_patches_color(image, keypoints, patchsize):
    patches = numpy.zeros((len(keypoints), 3*patchsize**2))
    for i, k in enumerate(keypoints):
        patches[i, :] = image[k[0]-patchsize/2:k[0]+patchsize/2, k[1]-patchsize/2:k[1]+patchsize/2,:].flatten()
    return patches


def pca(data, var_fraction):
    """ principal components, retaining as many components as required to 
        retain var_fraction of the variance 

    Returns projected data, projection mapping, inverse mapping, mean"""
    from numpy.linalg import eigh
    u, v = eigh(numpy.cov(data, rowvar=1, bias=1))
    v = v[:, numpy.argsort(u)[::-1]]
    u.sort()
    u = u[::-1]
    u = u[u.cumsum()<u.sum()*var_fraction]
    numprincomps = u.shape[0]
    V = ((u**(-0.5))[:numprincomps][numpy.newaxis,:]*v[:,:numprincomps]).T
    W = (u**0.5)[:numprincomps][numpy.newaxis,:]*v[:,:numprincomps]
    return V, W
    #return numpy.dot(V,data), V, W




##USE MINI CIFAR DATA
#print "loading data"
#trainims = numpy.loadtxt("cifar_mini_images_train.txt")
#testims = numpy.loadtxt("cifar_mini_images_test.txt")
#trainlabels = numpy.loadtxt("cifar_mini_labels_train.txt").astype("int")
#testlabels = numpy.loadtxt("cifar_mini_labels_test.txt").astype("int")
#allims = numpy.concatenate((trainims,testims), 0)
#
#numtrain = numpy.int(trainims.shape[0]*0.75)  #use 3/4 for training 
#numvali = numpy.int(trainims.shape[0]*0.25)   #use 1/4 for validation 

##USE CIFAR DATA
numtrain = 10000
numvali = 2000
numtest = 2000
print "loading data"                                                                                     
allims = numpy.concatenate([(numpy.load(datadir+'/data_batch_'+b)['data'].astype("float").reshape(10000, 3, 1024)) for b in ["1", "2", "3", "4", "5"]], 0)                                                        
R = numpy.random.permutation(allims.shape[0])                                                            
allims = allims[R][:numtrain+numvali]                                                                            
testims = (numpy.load(datadir+'/test_batch')['data'].astype("float").reshape(10000, 3, 1024))[:numtest]  
allims = numpy.concatenate((allims, testims), 0)                                                         
del testims                                                                                              
alltrainlabels = onehot(numpy.concatenate([numpy.load(datadir+'/data_batch_'+b)['labels'] for b in ["1", "2", "3", "4", "5"]], 0).astype("int32")[R][:numtrain+numvali])
testlabels = onehot(numpy.array(numpy.load(datadir+'/test_batch')['labels']).astype("int32")[:numtest])
trainims = allims[:numtrain]                                                                             




#WHITENING
patches = numpy.concatenate([crop_patches_color(im.reshape(3, 32, 32).transpose(1,2,0), numpy.array([rng.randint(patchsize/2, 32-patchsize/2, 20), rng.randint(patchsize/2, 32-patchsize/2, 20)]).T, patchsize) for im in trainims]).astype("float32")
R = rng.permutation(patches.shape[0])
patches = patches[R, :]
meanstd = patches.std()
patches -= patches.mean(1)[:,None]
patches /= patches.std(1)[:,None] + 0.1 * meanstd
#patches -= patches.mean(0)[None,:]
#patches /= patches.std(0)[None,:] 
pca_backward, pca_forward = pca(patches.T, .99)
print "done"


print "extracting patch pairs"
alldata = []
print "xxxxx", 
for i, image in enumerate(allims):
    print "\b\b\b\b\b\b{0:5d}".format(i), 
    patch_pairs = []
    image = image.reshape(3, 32, 32).transpose(1,2,0)
    #use random keypoints:
    keypoints = numpy.array([(i,j) for i in numpy.arange(border,32-border, stride) for j in numpy.arange(border,32-border, stride)])
    #rng.shuffle(keypoints)
    keypoints = keypoints.repeat(8, 0)
    #-use random keypoints 

    #keypoints = harris.get_harris_points(rgb2gray(image), min_distance=3, border=patchsize+border, threshold=0.1)
    #numpy.random.shuffle(keypoints)
    ##add some random positions so we will certainly be able to use the image:
    #keypoints.append(numpy.array([12,12]))
    #keypoints.append(numpy.array([20,20]))
    #keypoints = numpy.array(keypoints)

    patches_1 = crop_patches_color(image, keypoints, patchsize)
    patches_1 -= patches_1.mean(1)[:,None] 
    patches_1 /= patches_1.std(1)[:,None] + 0.1 * meanstd
    patches_1 = numpy.dot(patches_1, pca_backward.T)
    #keypointdeltas = numpy.vstack((numpy.random.randint(-1, 2, len(patches_1)), numpy.random.randint(-1, 2, len(patches_1)))).T
    keypointdeltas = numpy.kron(numpy.ones((len(keypoints)/8,1)), numpy.vstack((numpy.array([-1,-1,-1,0,0,1,1,1]),numpy.array([-1,0,1,-1,1,-1,0,1]))).T)
    patch2positions = keypoints+patchsize*keypointdeltas
    patches_2 = crop_patches_color(image, patch2positions, patchsize)
    patches_2 -= patches_2.mean(1)[:,None] 
    patches_2 /= patches_2.std(1)[:,None] + 0.1 * meanstd
    patches_2 = numpy.dot(patches_2, pca_backward.T)
    keypoint_deltas = numpy.zeros((patches_1.shape[0], 3*3), dtype="float32")
    for k in range(len(patches_1)):
        keypoint_deltas[k, (keypointdeltas[k][0]+1)*3+(keypointdeltas[k][1]+1)] = 1.0
    keypoint_deltas = numpy.concatenate((keypoint_deltas[:,:4], keypoint_deltas[:,5:]), 1)
    patch_pairs = numpy.hstack((patches_1, patches_2)).astype("float32")
    alldata.append(numpy.hstack((keypoint_deltas, patch_pairs)))
print "done"
print len(alldata)



print "instantiating model"
numpy_rng  = numpy.random.RandomState(1)
theano_rng = RandomStreams(1)
model = patchpair_encoder.Patchpairencoder(numvisD=8, numvisY=pca_forward.shape[1], numhid=numhid, 
                       output_type="real", corruption_type="zeromask", corruption_level=0.0,
                       weightcost=0.0, contraction=0.5, 
                       numpy_rng=numpy_rng, theano_rng=theano_rng)

bigramtrainfeatures = numpy.concatenate(alldata[:numtrain]).astype("float32")
trainer = train.GraddescentMinibatch_unloaded(model, bigramtrainfeatures, batchsize=100, loadsize=1000000, learningrate=0.05) 


print "training model"
for epoch in xrange(10):
    trainer.step()
    trainer.set_learningrate(trainer.learningrate*0.5)
    if True: #epoch % 10 == 0:
        pylab.figure(1)
        pylab.clf()
        c = 0 
        for i in range(3):
            for j in range(3):
                pylab.subplot(3,3,i*3+j+1)
                if c<4:
                    W = model.wdfh.get_value()[c,:model.numvisY, :]
                    W_ = numpy.dot(pca_forward, W)
                    dispims_color.dispims_color(W_.T.reshape(-1, patchsize, patchsize, 3))
                    pylab.draw(); pylab.show()
                elif c>4:
                    W = model.wdfh.get_value()[c-1,:model.numvisY, :]
                    W_ = numpy.dot(pca_forward, W)
                    dispims_color.dispims_color(W_.T.reshape(-1, patchsize, patchsize, 3))
                    pylab.draw(); pylab.show()
                c += 1
        pylab.figure(2)
        pylab.clf()
        c = 0 
        for i in range(3):
            for j in range(3):
                pylab.subplot(3,3,i*3+j+1)
                if c<4:
                    W = model.wdfh.get_value()[c,model.numvisY:, :]
                    W_ = numpy.dot(pca_forward, W)
                    dispims_color.dispims_color(W_.T.reshape(-1, patchsize, patchsize, 3))
                    pylab.draw(); pylab.show()
                elif c>4:
                    W = model.wdfh.get_value()[c-1,model.numvisY:, :]
                    W_ = numpy.dot(pca_forward, W)
                    dispims_color.dispims_color(W_.T.reshape(-1, patchsize, patchsize, 3))
                    pylab.draw(); pylab.show()
                c += 1

print "done"


print "extracting features with bigram model" 
allbigramfeatures = []
print "xxxxx", 
for i, d in enumerate(alldata):
    print "\b\b\b\b\b\b{0:5d}".format(i), 
    allbigramfeatures.append(model.uncorruptedhiddens(d).mean(0))

print "done"

alltrainfeatures = numpy.vstack(allbigramfeatures[:numtrain+numvali])
testfeatures = numpy.vstack(allbigramfeatures[numtrain+numvali:])
trainfeatures = alltrainfeatures[:numtrain]
valifeatures = alltrainfeatures[numtrain:]
trainlabels = alltrainlabels[:numtrain]
valilabels = alltrainlabels[numtrain:]
del allbigramfeatures



#CLASSIFICATION 
#weightcosts = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.0]
weightcosts = [0.01, 0.001, 0.0001, 0.0]
valicosts = []
lr = logreg.Logreg(numclasses, trainfeatures.shape[1])
lr.train(trainfeatures.T, trainlabels.T, numsteps=100, verbose=False, weightcost=weightcosts[0])
lr.train_cg(trainfeatures.T, trainlabels.T, weightcost=weightcosts[0], maxnumlinesearch=100)
valicosts.append(lr.zeroone(valifeatures.T, valilabels.T))
for wcost in weightcosts[1:]:
    lr.train(trainfeatures.T, trainlabels.T, numsteps=100,verbose=False,weightcost=wcost)
    lr.train_cg(trainfeatures.T, trainlabels.T, weightcost=wcost, maxnumlinesearch=100)
    valicosts.append(lr.zeroone(valifeatures.T, valilabels.T))

winningwcost = weightcosts[numpy.argmin(valicosts)]

lr.train(alltrainfeatures.T, alltrainlabels.T, numsteps=100, verbose=False, weightcost=winningwcost)
lr.train_cg(alltrainfeatures.T, alltrainlabels.T, weightcost=winningwcost, maxnumlinesearch=100) 

performance = 1.0 - lr.zeroone(testfeatures.T, testlabels.T)
print "logreg test performance: ", performance
print "winning weightcost: ", winningwcost

