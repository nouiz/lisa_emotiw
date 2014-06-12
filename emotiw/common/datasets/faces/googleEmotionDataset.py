# Copyright (c) 2013 University of Montreal
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

import cPickle
import csv
import numpy as np
import os
from faceimages import FaceImagesDataset
from imageseq import ImageSequenceDataset


class GoogleEmotionDataset(ImageSequenceDataset):
    def __init__(self):
        dataPath = "faces/GoogleEmotionDataset/"
        # Load the dataset's pickle file
        pklFile = "/data/lisa/data/" + dataPath + "assignmentData.pkl"
        self.data = cPickle.load(open(pklFile, "rb"))
        self.sequences = []

        allSets = zip(*self.data[0] + self.data[1] + self.data[2])
        self.ids = range(0, len(allSets[0]))
        self.sequences.append(GoogleEmotionSequence(allSets, self.data[3]))
        super(GoogleEmotionDataset, self).__init__('Google emotion frames')

    def __len__(self):
        return len(self.sequences)

    def get_sequence(self, i):
        return self.sequences[i]

    def get_standard_train_test_splits(self):
        trainIdx = self.ids[:len(self.data[0])]
        validIdx = self.ids[len(trainIdx):len(trainIdx) + len(self.data[1])]
        testIdx = self.ids[-len(self.data[2]):]
        return [(trainIdx, ), (validIdx, ), (testIdx, )]


class GoogleEmotionSequence(FaceImagesDataset):

    def __init__(self, data, labels):
        dataPath = "faces/GoogleEmotionDataset/"
        super(GoogleEmotionSequence, self).__init__("GED", dataPath)

        self.queryIds = data[0]
        self.ids = data[1]
        self.labels = labels
        self.tags = data[2]
        self.tagNames = ["anger", "bored", "concern", "crying",
                         "disappointed", "discouraged", "disgust",
                         "displeased", "elation", "fear", "happy", "nervous",
                         "neutral", "sad", "screaming", "shame", "surprise",
                         "tired"]
        self.X1 = data[3]
        self.Y1 = data[4]
        self.X2 = data[5]
        self.Y2 = data[6]

        # Mapping from 18 emotions to 7
        self.emo_18_to_7 = {'anger': 0, 'bored': 0, 'concern': 6, 'crying': 4,
                            'disappointed': 4, 'discouraged': 4, 'disgust': 1,
                            'displeased': 0, 'elation': 3, 'fear': 2,
                            'happy': 3, 'nervous': 6, 'neutral': 6, 'sad': 4,
                            'screaming': 2, 'shame': 6, 'surprise': 5,
                            'tired': 0}

        self.set_picasa_path_substitutions(
            {"faces/GoogleEmotionDataset/": dataPath+"facesCoordinates/",
             '.png': '.txt',
             '.jpg': '.txt',
             },
            csv_delimiter=',')

        self.set_ramanan_path_substitutions(
            {"faces/GoogleEmotionDataset": "faces/GoogleDataset/ramanan_keypoints/"})

    def get_name(self):
        return "GoogleEmotionDataset"

    def __len__(self):
        return len(self.ids)

    def get_original_image_path_relative_to_base_directory(self, i):
        return str(self.queryIds[i]) + "/" + str(self.ids[i]) + ".png"

    def get_index_from_image_filename(self, imgFileName):
        filename = imgFileName.split("/")[-1]
        imageId = int(filename[:-4])
        return (self.ids == imageId).argmax()

    def get_original_bbox(self, i):
        return [(int(self.X1[i]), int(self.Y1[i]),
                 int(self.X1[i]) + int(self.X2[i]),
                 int(self.Y1[i]) + int(self.Y2[i]))]

    def get_picasa_bbox(self, i):
        """Returns a list of bounding boxes precomputed by picasa.
        Calls get_picasa_path_from_image_path
        to locate the file containing the precomputed bounding box info"""
        imagepath = self.get_original_image_path(i)
        bboxpath = self.get_picasa_path_from_image_path(imagepath)
        if bboxpath is not None and os.path.exists(bboxpath):
            bboxes = []
            with open(bboxpath) as f:
                reader = csv.reader(f, delimiter=self.picasa_csv_delimiter)
                for row in reader:
                    picasaBatchNumber, idxInPicasaBatch, row, col, height, width = row
                    if picasaBatchNumber != "picasaBatchNumber":
                        print bboxpath, ":", picasaBatchNumber, idxInPicasaBatch, row, col, height, width
                        row, col, height, width = float(row), float(col), float(height), float(width)
                        bbox = [col, row, col+width, row+height]
                        print "bbox:", bbox
                        bboxes.append(bbox)
            return bboxes

        return None

    def get_detailed_emotion_label(self, i):
        queryId = int(self.queryIds[i])
        tag = int(self.tags[i])
        tagIndex = self.labels[queryId][tag].argmax()
        return self.tagNames[tagIndex]

    def get_7emotion_index(self, i):
        emo_18 = self.get_detailed_emotion_label(i)
        return self.emo_18_to_7[emo_18]


if __name__ == "__main__":
    GoogleEmotionDataset()
