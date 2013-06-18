# Copyright (c) 2013 University of Montreal, Pascal Vincent,
# Pascal Lamblin, Mehdi Mirza
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

import glob
import os
import os.path

from emotiw.common.utils.pathutils import locate_data_path
from imageseq import ImageSequenceDataset
from faceimages import FaceImagesDataset, basic_7emotion_names

# Subclasses of FaceImagesDataset

class AFEWImageSequence(FaceImagesDataset):
    def __init__(self, dataset_name, relative_image_base_directory,
                 image_glob, emotionName):
        super(AFEWImageSequence, self).__init__(dataset_name, relative_image_base_directory)

        self.imageRelativePath = []  # Relative path to images
        self.emotionIndex = basic_7emotion_names.index(emotionName.lower())
        self.imageIndex = {}

        # Fetch images
        images_abspath = glob.glob(os.path.join(self.absolute_base_directory, image_glob))
        # Sort the frames
        images_abspath.sort()

        rel_startpos = len(self.absolute_base_directory)
        if not self.absolute_base_directory.endswith('/'):
            rel_startpos += 1 # must skip the separating /
        self.imageRelativePath = [ path[rel_startpos:] for path in images_abspath ]

        # Builds the data
        idx = 0
        for relpath in self.imageRelativePath:            
            self.imageIndex[relpath] = idx
            idx += 1

    def get_index_from_image_filename(self, imgFileName):
        return self.imageIndex[imgFileName]

    def __len__(self):
        return len(self.imageRelativePath)

    def get_original_image_path_relative_to_base_directory(self, i):
        return self.imageRelativePath[i]

    def get_7emotion_index(self, i):
        return self.emotionIndex

    def get_keypoints_location(self, i):
        return self.get_ramanan_keypoints_location(i)
    

class AFEWImageSequenceDataset(ImageSequenceDataset):
    emotionNames = {"Angry": "anger", "Disgust": "disgust", "Fear": "fear", "Happy": "happy",
                    "Neutral": "neutral", "Sad": "sad", "Surprise": "surprise"}

    # These directories are relative to the data path.
    base_dir = "faces/AFEW/images"
    picasa_boxes_base_dir = "faces/AFEW/picasa_boxes"

    def __init__(self, name="AFEW"):
        super(AFEWImageSequenceDataset,self).__init__(name)

        self.absolute_base_directory = locate_data_path(self.base_dir)
        self.absolute_picasa_boxes_base_directory = locate_data_path(
                self.picasa_boxes_base_dir)

        self.imagesequences = []
        self.labels = []
        self.trainIndexes = []
        self.validIndexes = []
        # For each emotion subfolder
        idx = 0
        directories = os.listdir(self.absolute_base_directory)
        directories.sort()
        for emotionName in directories:
            # Check if it is a emotion subfolder
            abs_emotionDir = os.path.join(self.absolute_base_directory, emotionName)
            rel_emotionDir = os.path.join(self.base_dir, emotionName)
            if not os.path.isdir(abs_emotionDir) or emotionName not in self.emotionNames.keys():
                continue
            # abs_picasa_bbox_dir = os.path.join(self.absolute_picasa_boxes_base_directory, emotionName)
            # rel_picasa_bbox_dir = os.path.join(self.picasa_boxes_base_dir, emotionName)

            # Find all images
            fileNames = glob.glob(os.path.join(abs_emotionDir, "*.png"))

            # Find all unique sequences
            uniqueSequence = list(set([name.split("-")[0] for name in fileNames]))
            uniqueSequence.sort()

            # For each unique sequence
            for sequence in uniqueSequence:
                # Load the Image Sequence object
                seq = AFEWImageSequence("AFEW", rel_emotionDir,
                                        "{0}-*.png".format(sequence),
                                        self.emotionNames[emotionName])
                seq.set_picasa_path_substitutions(
                    {self.base_dir:self.picasa_boxes_base_dir,
                     '-':'_',
                     '.png':'.txt',
                     '.jpg':'.txt',
                     }, csv_delimiter=' ')
                self.imagesequences.append(seq)
                # Save label
                self.labels.append(self.emotionNames[emotionName])
                # Save if in train or valid
                self.trainIndexes.append(idx)

                idx += 1

        return

    def __len__(self):
        return len(self.imagesequences)

    def get_sequence(self, i):
        return self.imagesequences[i]

    def get_label(self, i):
        return self.labels[i]

    def get_standard_train_test_splits(self):
        # Only one fold
        return [(self.trainIndexes, self.validIndexes)]

