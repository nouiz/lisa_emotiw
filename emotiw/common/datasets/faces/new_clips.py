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
"""
new_clips contains the data extracted from DVDs by Atousa Torabi,
labeled by the LISA lab.
"""
import os

#import numpy as np

from emotiw.common.utils.pathutils import locate_data_path
import afew
import afew2


class NewClipsImageSequenceDataset(afew2.AFEW2ImageSequenceDataset):
    # These directories are relative to the data path
    # Since there was no room at LISA for the data to be in the usual datapath,
    # there is a symlink instead.
    base_dir = "faces/new_clips/ExtractedFrames"
    picasa_boxes_base_dir = base_dir
    face_tubes_base_dir = base_dir

    def __init__(self, preload_facetubes=False, preproc=[], size=(96, 96)):
        """
        If preload_facetubes is True, all facetubes will be loaded when the
        data set is built.
        """

        if preproc:
            raise NotImplementedError()

        super(afew.AFEWImageSequenceDataset, self).__init__("NewClips")

        self.absolute_base_directory = locate_data_path(self.base_dir)
        #self.absolute_picasa_boxes_base_directory = locate_data_path(
        #        self.picasa_boxes_base_dir)
        #self.face_tubes_base_directory = locate_data_path(
        #        self.face_tubes_base_dir)

        self.preload_facetubes = preload_facetubes
        self.preproc = preproc
        self.size = size
        self.imagesequences = []
        self.labels = []
        self.seq_info = []
        self.trainIndexes = []
        self.validIndexes = []

        # No validation split for this dataset
        # Find all clips to keep (not marked "REJECT" during labeling)
        labels_file = os.path.join(self.absolute_base_directory, 'labels.txt')
        f = open(labels_file, 'r')
        clips_and_targets = [(clip_name[:-4], target)
                             for clip_name, target in [
                                 l.split() for l in f.readlines()]
                             if target != 'REJECT']

        idx = 0
        for clip_name, target in clips_and_targets:
            #abs_img_dir = os.path.join(self.absolute_base_directory,
            #                           clip_name)
            rel_img_dir = os.path.join(self.base_dir,
                                       clip_name)
            im_seq = afew.AFEWImageSequence(
                "NewClips",
                rel_img_dir,
                '*.png',
                self.emotionNames[target])

            #im_seq.set_picasa_path_substitutions(
            #    #???
            #    )

            #im_seq.set_ramanan_path_substitutions(
            #    # ???
            #    )

            self.imagesequences.append(im_seq)

            # Save clip_name of sequence
            self.seq_info.append(clip_name)

            # If needed, load facetubes
            if self.preload_facetubes:
                self.facetubes.append(self.load_facetubes(clip_name))

            # Save label
            self.labels.append(self.emotionNames[target])

            # Save split
            self.trainIndexes.append(idx)

            idx += 1

    def get_facetubes(self, i):
        if self.preload_facetubes:
            return self.facetubes[i]
        else:
            return self.load_facetubes(self.seq_info[i])

    def load_facetubes(self, clip_name):
        # Not available yet
        return None
