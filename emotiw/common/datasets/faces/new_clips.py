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
import cPickle
import glob
import os

import numpy as np

from emotiw.common.utils.pathutils import locate_data_path
import afew
import afew2


class NewClipsImageSequenceDataset(afew2.AFEW2ImageSequenceDataset):
    # These directories are relative to the data path
    # Since there was no room at LISA for the data to be in the usual datapath,
    # there is a symlink instead.
    base_dir = "faces/new_clips/ExtractedFrames"
    picasa_boxes_base_dir = "faces/new_clips/BoundingBox"
    face_tubes_base_dir = "faces/new_clips/picasa_face_tubes_%s_%s"
    face_tubes_boxes_base_dir = "faces/new_clips/pickle"
    emodict_dir = "faces/new_clips/Scripts"

    def __init__(self, preload_facetubes=False, preproc=[], size=(96, 96)):
        """
        If preload_facetubes is True, all facetubes will be loaded when the
        data set is built.
        """
        self.face_tubes_base_dir = self.face_tubes_base_dir%(size[0], size[1])
        self.facetubes_to_filter = None
        for opt in preproc:
            if opt == "smooth":
                # Use the bounding-boxes smoothed version of the face tubes.
                self.face_tubes_base_dir = ("faces/new_clips/smooth_picasa_face_tubes_%s_%s"
                                            "/numpy_arr_concatenated")%(size[0], size[1])
                self.face_tubes_boxes_base_dir = ("faces/new_clips/smooth_picasa_face_tubes_%s_%s"
                                                  "/picasa_tubes_pickles")%(size[0], size[1])

            if opt == "remove_background_faces":
                # Remove background faces as many as possible from the dataset.
                # Path to the dictionary giving for each dataset, the list of
                # face tubes corresponding to background faces/objects.
                abs_dir = locate_data_path("faces/new_clips")
                filename = os.path.join(abs_dir, "background_info.pkl")
                try:
                    f = open(filename, 'rb')
                    self.facetubes_to_filter = cPickle.load(f)
                    f.close()
                except IOError as e:
                    print e

        super(afew.AFEWImageSequenceDataset, self).__init__("NewClips")

        self.absolute_base_directory = locate_data_path(self.base_dir)
        self.absolute_picasa_boxes_base_directory = locate_data_path(
                self.picasa_boxes_base_dir)
        self.face_tubes_base_directory = locate_data_path(
                self.face_tubes_base_dir)
        self.face_tubes_boxes_base_directory = locate_data_path(
                self.face_tubes_boxes_base_dir)
        self.emodict_base_directory = locate_data_path(
                self.emodict_dir)

        self.preload_facetubes = preload_facetubes
        self.preproc = preproc
        self.size = size
        self.imagesequences = []
        self.labels = []
        self.seq_info = []
        self.trainIndexes = []
        self.validIndexes = []
        self.emo2numDict = {}
        self.num2emoDict = {}

        # Dictionary that gives a mapping from emotion name to clip id.
        file = open(os.path.join(self.emodict_base_directory,
                    "newFilenameTranslate.txt"), 'r')
        line = file.readline()

        while line != '':
            if line[0] != '=':
                ln = line.split(':')
                emo = ln[0].strip()
                num = ln[1].strip().split(' ')[0]
                self.emo2numDict[emo] = num
                self.num2emoDict[num] = emo
            line = file.readline()

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

            im_seq.set_picasa_path_substitutions(
                {self.base_dir: self.picasa_boxes_base_dir,
                 '.png': '.txt'},
                csv_delimiter=',')

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

    def get_bbox_coords(self, i):
        """
        Get a list of dictionary containing all facetubes' bounding boxes
        coordinates of clip i.  These bounding box coordinates are relative
        to the original picasa image (uncropped version).

        The dictionary gives the bounding boxes for all frames in a facetube.
        The key is the frame number and we associate it a list of 4 numbers
        representing the bounding box coordinates of that frame.

        Each 4-tuple is x1,y1,x2,y2 giving the coordinates of the top left
        corner and bottom right corner of a bounding box.  Coordinate system
        has its origin in the upper left corner of the image
        (horizontal_offset_in_pixels, vertical_offset_in_pixels).
        """
        rval = []
        clip_name = self.seq_info[i].split("/")[-1]
        clip_id = self.emo2numDict[clip_name]
        path = os.path.join(self.face_tubes_boxes_base_directory, "%s.pkl"%clip_name)
        try:
            f = open(path, 'rb')
            bbox_coords = cPickle.load(f)
            f.close()
            if clip_id in bbox_coords:
                rval = bbox_coords[clip_id]
        except IOError:
            pass
        return rval

    def get_facetubes(self, i):
        if self.preload_facetubes:
            return self.facetubes[i]
        else:
            return self.load_facetubes(self.seq_info[i])

    def load_facetubes(self, clip_name):
        #import pdb; pdb.set_trace()
        clip_id = self.emo2numDict[clip_name]
        npy_dir = self.face_tubes_base_directory
        #print 'npy_dir:', npy_dir
        #print 'clip_id:', clip_id
        npy_glob = os.path.join(npy_dir, '{0}-*.npy'.format(clip_id))
        npy_files = glob.glob(npy_glob)
        #print 'npy_glob:', npy_glob
        # sort the filenames of tubes
        npy_files.sort()
        # Filter out the background faces filenames if required.
        if self.facetubes_to_filter:
            npy_files_to_be_kept = []
            if clip_id in self.facetubes_to_filter:
                facetubes_ids = self.facetubes_to_filter[clip_id]
                for npy_file in npy_files:
                    # Get only the (clip_id, facetube_id).
                    npy_basename = os.path.basename(npy_file)
                    root = os.path.splitext(npy_basename)[0]
                    background = [f for f in facetubes_ids if f.startswith('%s-'%root)]
                    # Check if the given facetube is associated to a background face.
                    if not len(background):
                        # Not a background face thus it will be kept.
                        npy_files_to_be_kept.append(npy_file)
                npy_files = npy_files_to_be_kept
        #print 'npy_files:', npy_files
        rval = []
        #import pdb; pdb.set_trace()
        for f in npy_files:
            rval.append(np.load(f))
        #import pdb; pdb.set_trace()
        return tuple(rval)
