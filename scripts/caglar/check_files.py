import glob
import numpy

from tables import *
from PIL import Image
import os
import warnings

"""
Get the bounding boxes.
"""
def get_bounding_boxes(path, img_path):
    rval = {}
    count = 0
    batches = os.walk(path).next()[1]

    for batch in batches:
        list = glob.glob("{}{}/*.txt".format(path, batch))
        batch_holder = {}
        for item in list:
            name = int(item.split('/')[-1].rstrip('.txt').split('-')[-1])
            face_path = "{}/{}/{}.png".format(img_path, batch, name)
            if not os.access(face_path, os.R_OK):
                print("%s"%face_path)
            break
        rval[batch] = batch_holder
    return rval, count

get_bounding_boxes("/data/lisa/data/faces/GoogleDataset/images/facesCoordinates/", "/data/lisa/data/faces/GoogleDataset/images/")
