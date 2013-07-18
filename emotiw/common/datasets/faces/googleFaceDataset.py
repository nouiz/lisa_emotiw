# Copyright (c) 2013 University of Montreal, Pierre-Luc Carrier
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The names of the authors and contributors to this software may not be
#       used to endorse or promote products derived from this software without
#       specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import numpy as np
import sys
import glob
import cPickle
from scipy import io as sio
import unicodedata
import csv

from faceimages import FaceImagesDataset

# Subclasses of FaceImagesDataset
class GoogleFaceDataset(FaceImagesDataset):

    def __init__(self):
        super(GoogleFaceDataset,self).__init__("GFD", "faces/GoogleDataset/")

        # Load the dataset's pickle file
        data = cPickle.load(open(self.absolute_base_directory+"Clean/latest.pkl","rb"))
        #import ipdb
        #ipdb.set_trace()
        data = data[1:]

        # print data
        self.labels = data[0]
        self.ids = data[2]
        self.queryIds = data[3]
        self.tags = data[4]
        self.tagNames = data[7]
        self.X1 = data[8]
        self.Y1 = data[9]
        self.X2 = data[10]
        self.Y2 = data[11]

        self.set_picasa_path_substitutions(
            {"faces/GoogleDataset/images/":"faces/GoogleDataset/images/facesCoordinates/",
             '.png':'.txt',
             '.jpg':'.txt',
             }
            , csv_delimiter=',')

        self.set_ramanan_path_substitutions(
            {"faces/GoogleDataset/images/":"faces/GoogleDataset/ramanan_keypoints/"})

    def get_name(self):
        return "GoogleFaceDataset"

    def __len__(self):
        return len(self.labels)

    def get_original_image_path_relative_to_base_directory(self, i):
        return "images/" + str(self.queryIds[i]) + "/" + str(self.ids[i]) + ".png"

    def get_index_from_image_filename(self, imgFileName):
        filename = imgFileName.split("/")[-1]
        imageId = int(filename[:-4])
        return (self.ids == imageId).argmax()

    def get_original_bbox(self, i):
        return [(self.X1[i], self.Y1[i],
                 self.X2[i], self.Y2[i])]

        # return [[self.X1[i], self.Y1[i],
        #          self.X2[i] - self.X1[i],
        #          self.Y2[i] - self.Y1[i]]]

    def get_picasa_bbox(self, i):
        """Returns a list of bounding boxes precomputed by picasa.
        Calls get_picasa_path_from_image_path
        to locate the file containing the precomputed bounding box info"""
        imagepath = self.get_original_image_path(i)
        # pdb.set_trace()
        bboxpath = self.get_picasa_path_from_image_path(imagepath)
        if bboxpath is not None and os.path.exists(bboxpath):
            bboxes = []
            with open(bboxpath) as f:
                reader = csv.reader(f, delimiter=self.picasa_csv_delimiter)
                for row in reader:
                    picasaBatchNumber, idxInPicasaBatch, row, col, height, width = row
                    if picasaBatchNumber!="picasaBatchNumber":
                        print bboxpath, ":", picasaBatchNumber, idxInPicasaBatch, row, col, height, width
                        row, col, height, width = float(row), float(col), float(height), float(width)
                        bbox = [col, row, col+width, row+height]
                        print "bbox:", bbox
                        bboxes.append(bbox)
            return bboxes

        return None


    def get_7emotion_index(self, i):
        return self.labels[i].argmax()

    def get_detailed_emotion_label(self, i):
        return self.tagNames[self.tags[i].argmax()]


if __name__ == "__main__":
    GoogleFaceDataset()
