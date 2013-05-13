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
border = patchsize/2
maxdelta = 5
fixation_mask_size = 3
numhid = 100
#pooling = "quadrants"
numpatches_per_image = 500

rng  = numpy.random.RandomState(1)


def rgb2gray(image):
  return numpy.sum(image*numpy.array([0.3,0.59,0.11])[None,None,:],2)


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




#FETCH DATA 
print "loading data"
trainims = numpy.loadtxt("cifar_mini_images_train.txt")
testims = numpy.loadtxt("cifar_mini_images_test.txt")
trainlabels = numpy.loadtxt("cifar_mini_labels_train.txt").astype("int")
testlabels = numpy.loadtxt("cifar_mini_labels_test.txt").astype("int")
allims = numpy.concatenate((trainims,testims), 0)

numtrain = numpy.int(trainims.shape[0]*0.75)  #use 3/4 for training 
numvali = numpy.int(trainims.shape[0]*0.25)   #use 1/4 for validation 



#SHOW SOME IMAGES
imstoshow = numpy.zeros((30, 32, 32, 3))
for c in range(numclasses):
    imstoshow[c*3:(c+1)*3] = trainims[trainlabels[:,c]==1][:3].reshape(3,3,32,32).transpose(0,2,3,1)

dispims_color.dispims_color(imstoshow, 1)




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
#pcadata, pca_backward, pca_forward = pca(patches.T, .9)
#featurelearndata = pcadata.T.astype("float32")
#del pcadata
#numpatches = featurelearndata.shape[0] 
print "done"




#EXTRACT PATCH PAIRS 
print "extracting patch pairs from images"
alldata = []
print "xxxxx", 
for i, image in enumerate(allims):
    print "\b\b\b\b\b\b{0:5d}".format(i), 
    patch_pairs = []
    keypoint_deltas = []
    image = image.reshape(3, 32, 32).transpose(1,2,0)

    #use random keypoints:
    keypoints = numpy.array([(i,j) for i in numpy.arange(border,32-border) for j in numpy.arange(border,32-border)])
    rng.shuffle(keypoints)
    #-use random keypoints 

    #use harris keypoints:
    #keypoints = harris.get_harris_points(rgb2gray(image), min_distance=3, border=border, threshold=0.1)
    #rng.shuffle(keypoints)
    ##add some random positions to ensure we will be able to use the image:
    #keypoints.append(numpy.array([12,12]))
    #keypoints.append(numpy.array([15,15]))
    #keypoints = numpy.array(keypoints)
    #-use harris keypoints

    patches = crop_patches_color(image, keypoints, patchsize)
    patches -= patches.mean(1)[:,None] 
    patches /= patches.std(1)[:,None] + 0.1 * meanstd
    patches = numpy.dot(patches, pca_backward.T)
    #for patch_index_1 in range(len(patches)):
    keypoints_with_replacements = numpy.arange(len(patches))[rng.randint(0, len(patches)-1, numpatches_per_image)]
    for patch_index_1, patch_index_2 in zip(keypoints_with_replacements[::-1], keypoints_with_replacements[1:]):
        #for patch_index_2 in range(len(patches)):
        #patch_index_2 = (patch_index_1+1) % len(patches)
        if numpy.sum(numpy.abs(keypoints[patch_index_1]-keypoints[patch_index_2]) >= (maxdelta,maxdelta)) != 0:
            continue      #include only keypoints that are fairly close to each other 
        patch_pairs.append(numpy.hstack((patches[patch_index_1], patches[patch_index_2])))
        keypoint_deltas.append(numpy.zeros((fixation_mask_size, fixation_mask_size)))
        index = numpy.floor(((keypoints[patch_index_1]-keypoints[patch_index_2]+maxdelta)/(2.0*maxdelta))*fixation_mask_size)
        keypoint_deltas[-1][index[0], index[1]] = 1.0
        keypoint_deltas[-1] = keypoint_deltas[-1].flatten()
        #add the symmetric fixation pair, too:
        patch_pairs.append(numpy.hstack((patches[patch_index_2], patches[patch_index_1])))
        keypoint_deltas.append(numpy.zeros((fixation_mask_size, fixation_mask_size)))
        index = numpy.floor(((keypoints[patch_index_2]-keypoints[patch_index_1]+maxdelta)/(2.0*maxdelta))*fixation_mask_size)
        keypoint_deltas[-1][index[0], index[1]] = 1.0
        keypoint_deltas[-1] = keypoint_deltas[-1].flatten()
    if len(patch_pairs) < 2:
        continue
    patch_pairs = numpy.vstack(patch_pairs).astype("float32")
    keypoint_deltas = numpy.vstack(keypoint_deltas).astype("float32")
    alldata.append(numpy.hstack((keypoint_deltas, patch_pairs)))
print "done"
print len(alldata)

print "instantiating model"
numpy_rng  = numpy.random.RandomState(1)
theano_rng = RandomStreams(1)
model = patchpair_encoder.Patchpairencoder(numvisD=9, numvisY=pca_forward.shape[1], numhid=numhid, 
                       output_type="real", corruption_type="zeromask", corruption_level=0.0,
                       weightcost=0.0, contraction=0.5, 
                       numpy_rng=numpy_rng, theano_rng=theano_rng)

bigramtrainfeatures = theano.shared(numpy.concatenate(alldata[:numtrain]).astype("float32"))
trainer = train.GraddescentMinibatch(model, bigramtrainfeatures, batchsize=100, learningrate=0.01, normalizefilters=False) 


print "training model"
for epoch in xrange(100):
    trainer.step()
    if epoch % 10 == 0:
        pylab.figure(1)
        pylab.clf()
        for i in range(3):
            for j in range(3):
                pylab.subplot(3,3,i*3+j+1)
                W = model.wdfh.get_value()[i*3+j,:model.numvisY, :]
                W_ = numpy.dot(pca_forward, W)
                dispims_color.dispims_color(W_.T.reshape(-1, patchsize, patchsize, 3))
                pylab.draw(); pylab.show()
        pylab.figure(2)
        pylab.clf()
        for i in range(3):
            for j in range(3):
                pylab.subplot(3,3,i*3+j+1)
                W = model.wdfh.get_value()[i*3+j,model.numvisY:, :]
                W_ = numpy.dot(pca_forward, W)
                dispims_color.dispims_color(W_.T.reshape(-1, patchsize, patchsize, 3))
                pylab.draw(); pylab.show()

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
alltrainlabels = trainlabels
valilabels = trainlabels[numtrain:]
trainlabels = trainlabels[:numtrain]
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



