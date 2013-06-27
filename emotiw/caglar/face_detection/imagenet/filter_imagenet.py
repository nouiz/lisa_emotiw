from dataset_256x256 import *
from PIL import Image
import os
import h5py
from imagenet_iter import DatasetIter
import time

import warnings

"""
Convert the images to grayscale.
"""
def rgb2gray(rgb):
    r, g, b = numpy.rollaxis(rgb[...,:3], axis = -1)
    return 0.299 * r + 0.587 * g + 0.114 * b

def load_excluded_targets(filename="excluded_cats.txt"):
    excluded_targets = {}
    lines = [line.strip() for line in open(filename)]
    for line in lines:
        if line in TARGETS:
            excluded_targets[line] = TARGETS[line]
    return excluded_targets

def filter_dataset(n_files=1261405, src_file="imagenet_256x256_train.h5", dst_file="imagenet_256x256_filtered.h5", batch_size=1000, limit=None):
    numpy.random.seed(0xbeef)
    filtered_data = h5py.File(dst_file, "w")
    unfiltered_data = h5py.File(src_file, "r")

    train_iter = DatasetIter(unfiltered_data)

    excluded_targets = load_excluded_targets()

    filtered_data.create_dataset("x", (n_files, 256*256), 'uint8')
    filtered_data.create_dataset("y", (n_files,), 'uint32')

    inds = range(n_files)
    numpy.random.shuffle(inds)
    print "Started saving the imagenet file to directory %s " % dst_file

    i = 0
    begin = time.time()
    for features, targets in train_iter.minibatch_iterator(batch_size):
        if limit:
            if limit == i:
                break
        if targets[0] in  excluded_targets.values():
            continue
        img = features[0].reshape((3, 256, 256)).T
        gray = rgb2gray(img)
        filtered_data['x'][i] = gray.flatten()
        filtered_data['y'][i] = targets[0]
        i += 1
    end = time.time()
    print "Elapsed time (%.2fs)" % (end-begin)
