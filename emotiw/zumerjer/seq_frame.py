"""
Get sequence classification reuslts based on single based classifiers
"""

import numpy
from pylearn2.utils import py_integer_types
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from pylearn2.datasets import preprocessing
import theano.tensor as T
from theano import function
import sys
#from emotiw.common.datasets.faces.afew2_facetubes import AFEW2FaceTubes
from collections import Counter
from seq_dset import SeqDataset
from emotiw.common.datasets.faces.faceimages import basic_7emotion_names

def get_classifier(model):

    X = model.get_input_space().make_batch_theano()
    X.name = 'X'

    model.layers = model.layers[:-1]
    y = model.fprop(X)
    y.name = 'y'

    return function([X],y)

def accuracy(predictions, target):

    #mx = numpy.argmax(predictions, axis=1)
    #mask = numpy.zeros(predictions.shape)

    #for i in range(mask.shape[0]):
        #mask[i, mx[i]] = 1.
    #predictions = mask * predictions
    #y_hat = numpy.argmax(predictions.sum(axis=0))

    if target is None:
        return None

    max = predictions.max(axis=0)
    hapoo = numpy.log(numpy.exp(predictions-max[numpy.newaxis,:]).sum(axis=0))
    y_hat = numpy.argmax(hapoo)
    #print y_hat, target
    return int(y_hat != target)


def framewise_accuracy(predictions, target):
    if target is None:
        return None

    y_hat = numpy.argmax(predictions,axis=1)
    y =numpy.ones(y_hat.shape) * target
    #print Counter(y_hat), target
    #ipdb.set_trace()
    return y_hat != target


def get_stats2(predictions, target):

    y_hat = numpy.argmax(predictions,axis=1)
    y =numpy.ones(y_hat.shape) * target
    counts = Counter(y_hat)
    stats = []
    for i in xrange(7):
        if i in counts.keys():
            stats.append(counts[i])
        else:
            stats.append(0)

    return numpy.array(stats)/float(len(predictions))

def get_stats(predictions, target, scale = True):

    stats=[]
    if numpy.isinf(predictions).sum() > 0:
        print 'Found sum Inf values'
        predictions[numpy.isinf(predictions)] = 0    

    #min 
    stats.append(predictions.min(axis=0))

    #Mean
    stats.append(predictions.mean(axis=0))
    
    #median
    stats.append(numpy.median(predictions, axis=0))

    #std
    stats.append(numpy.std(predictions, axis=0))

    max = predictions.max(axis=0)
#   stats.append(numpy.log(numpy.exp(predictions-max[numpy.newaxis,:]).sum(axis=0)))

    #
    stats.append(max)

    #
    stats.append(numpy.sqrt((predictions ** 2).sum(axis=0)))

    #
#    mx = numpy.argmax(predictions, axis=1)
 #   mask = numpy.zeros(predictions.shape)
  #  for i in range(mask.shape[0]):
   #     mask[i, mx[i]] = 1.
  #  stats.append((mask * predictions).sum(axis=0))


    #
   # if target is not None:
   #     y_hat = numpy.argmax(predictions,axis=1)
   #     y =numpy.ones(y_hat.shape) * target
   #     counts = Counter(y_hat)
        #print counts, target
   #     _stats = []
   #     for i in xrange(7):
   #         if i in counts.keys():
   #             _stats.append(counts[i])
   #         else:
   #             _stats.append(0)

   #     stats.append(numpy.array(_stats)/float(len(predictions)))
    #else:
    #    stats.append(None)

    #if scale:
    #    _stats = []
    #    for item in stats:
    #        if item.max() - item.min() != 0:
    #            _stats.append((item - item.min()) / (item.max() - item.min()))
    #        elif item.max() != 0:
    #            _stats.append(item / item.max())
    #        else: _stats.append(item)
    #    stats = _stats

    return numpy.concatenate(stats)

def get_data(which_set, clip):
    data = SeqDataset(which_set=which_set)

    # organize axis
    data_axes = data.get_data_specs()[0].components[0].axes
    if 't' in data_axes:
        data_axes = [axis for axis in data_axes if axis not in ['t']]

    #targets = numpy.argmax(data.targets, axis=1)
    return data.get_clip(clip) + (data_axes,)


def run_model(which_set, classifier, batch_size, model_axes, data_axes=('b', 0, 1, 'c')):
    dset = SeqDataset(which_set=which_set)
    #indices = dset.get_filtered_indices(perturbations=['0'], flips=[False])
    indices = range(dset.numSamples)
    #indices = [2,3,4]
    #print indices
    targets_ints = []
    stats = []
    fileName = []

    for n in indices:
        features, targets, fname = dset.get_clip(n)
        misclass = []
        frame_misclass = []
        
        feature = features.reshape(len(features), 48, 48, 1)
        target = None
        if targets is not None:
            target = basic_7emotion_names.index(targets.lower().replace("angry", "anger"))
        #feature = feature / 255.
        #feature = feature.astype('float32')
        if data_axes != model_axes:
            feature = feature.transpose(*[data_axes.index(axis) for axis in model_axes])

        num_samples = feature.shape[3]
        predictions = []
        for i in range(num_samples / batch_size):
            # TODO FIX ME, after grayscale
            predictions.append(classifier(feature[0,:,:,i*batch_size:(i+1)*batch_size][numpy.newaxis,:,:,:])) #XXX:numpy.newaxis

        # for modulo we pad with garbage
        if batch_size > num_samples:
            modulo = batch_size - num_samples
        else:
            modulo = num_samples % batch_size

        if modulo != 0:
            # TODO FIX ME, after grayscale
            shape = [1, feature.shape[1], feature.shape[2], modulo]
            padding = numpy.ones((shape)).astype('float32')
            # TODO FIX ME, after grayscale
            feature = numpy.concatenate((feature[0,:,:,(num_samples/batch_size) * batch_size:][numpy.newaxis,:,:,:], padding), axis = 3) #XXX:axis,numpy.newaxis
            predictions.append(classifier(feature)[:batch_size - modulo])
        targets_ints.append(target)
        predictions = numpy.concatenate(predictions, axis=0)
        misclass.append(accuracy(predictions, target))
        frame_misclass.append(framewise_accuracy(predictions, target))
        stats.append(get_stats(predictions, target))
        fileName.append(fname)

   # error = numpy.sum(misclass) / float(len(features))
   # print "clip wise: ", error, 1-error

   # frame_misclass = numpy.concatenate(frame_misclass)
   # error = frame_misclass.sum() / float(len(frame_misclass))
   # print "frame wise: ", error, 1-error

    return numpy.vstack(stats), targets_ints, fileName

def save(stats, targets, fnames, save_path):

    #dataDict = {}
    #for i in range(len(stats)):
    #    peturbation = metaData[i]['perturbation']
    #    flip = metaData[i]['flipped']
    #    stat = stats[i,:]
    #    if not perturbation in dataDict:
    #        dataDict[perturbation].append((flip,
        
    #d = zip(stats, targets)
    #data=dict(zip(clipIDs, d))
    serial.save(save_path, {'x': stats, 'y': targets, 'path':fnames})


if __name__ == "__main__":

    _, model_path = sys.argv
    model = serial.load(model_path)
    classifier = get_classifier(model)

    batch_size = model.batch_size
    model_axes = model.input_space.axes
    if model_axes != ('c', 0, 1, 'b'):
        raise ValueError("Model axis is not supoorted")

    for which_set in ['train', 'val', 'test']:
        stats, targets, fnames = run_model(which_set, classifier, batch_size, model_axes)
        print stats.shape
        print targets
        save_path = '{}.pkl'.format(which_set)
        save(stats, targets, fnames, save_path)
