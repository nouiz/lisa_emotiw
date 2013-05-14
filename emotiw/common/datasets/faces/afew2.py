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
AFEW2 contains the data distributed with the EmotiW challenge.
"""
import glob
import os

import numpy as np

from emotiw.common.utils.pathutils import locate_data_path
import afew


class AFEW2ImageSequenceDataset(afew.AFEWImageSequenceDataset):
    # These directories are relative to the data path.
    base_dir = "faces/EmotiW/images"
    picasa_boxes_base_dir = "faces/EmotiW/picasa_boxes"
    face_tubes_base_dir = "faces/EmotiW/picasa_face_tubes_96_96"

    def __init__(self, preload_facetubes=False):
        """
        If preload_facetubes is True, all facetubes will be loaded
        when the dataset is built, which takes around 1.2 GB.
        """
        super(AFEW2ImageSequenceDataset,self).__init__("AFEW2")

        self.absolute_base_directory = locate_data_path(self.base_dir)
        self.picasa_boxes_base_directory = locate_data_path(
                self.picasa_boxes_base_dir)
        self.face_tubes_base_directory = locate_data_path(
                self.face_tubes_base_dir)

        self.preload_facetubes = preload_facetubes
        self.imagesequences = []
        self.labels = []
        self.seq_info = []
        self.trainIndexes = []
        self.validIndexes = []

        # For each split (Train or Val)
        idx = 0
        splits = (("Train", self.trainIndexes),
                  ("Val", self.validIndexes))
        for split_name, split_index in splits:
            #print 'processing %s' % split_name
            for emo_name in sorted(self.emotionNames.keys()):
                #print '  %s' % emo_name
                # Directory containing the images for all clips of that
                # emotion in that split
                img_dir = os.path.join(self.absolute_base_directory,
                        split_name, emo_name)
                #print 'img_dir:', img_dir
                if not os.path.isdir(img_dir):
                    continue

                # Directory containing the picasa bounding boxes for clips
                # of that emotion in that split
                picasa_bbox_dir = os.path.join(
                        self.picasa_boxes_base_directory,
                        split_name, emo_name)

                # Find all image names
                img_names = glob.glob(os.path.join(img_dir, '*.png'))
                #print '%s img_names' % len(img_names)

                # Find all clips (sequences)
                unique_seq = sorted(set([img.split('-')[0]
                                         for img in img_names]))
                #print '%s unique_seq' % len(unique_seq)

                # For each clip
                for seq in unique_seq:
                    # Load the Image Sequence object
                    im_seq = afew.ImageSequence("AFEW2",
                            img_dir, "{0}-*.png".format(seq),
                            picasa_bbox_dir, self.emotionNames[emo_name],
                            csv_delimiter=',')
                    self.imagesequences.append(im_seq)

                    # Save (split, emotion, sequence ID) of sequence
                    seq_id = os.path.basename(seq)
                    self.seq_info.append((split_name, emo_name, seq_id))

                    # If needed, load facetubes
                    if self.preload_facetubes:
                        self.facetubes.append(self.load_facetubes(
                                split_name, emo_name, seq_id))

                    # Save label
                    self.labels.append(self.emotionNames[emo_name])

                    # Save split
                    split_index.append(idx)

                    idx += 1

                #print '  done, idx = %s' % idx

            #print 'done, idx = %s' % idx
        #print 'finished, idx = %s' % idx

    def get_facetubes(self, i):
        """
        Get a tuple of ndarrays containing all facetubes of clip i.

        The arrays have dimension (nframes, 96, 96, 3), as each frame
        has been resized to a 96x96 RGB image.

        The order of the tubes in that tuple is the same as the order
        of bounding boxes returned by ImageSequence.get_picasa_bbox.
        """
        if self.preload_facetubes:
            return self.facetubes[i]
        else:
            split_name, emo_name, seq_id = self.seq_info[i]
            return self.load_facetubes(split_name, emo_name, seq_id)

    def load_facetubes(self, split_name, emo_name, seq_id):
        npy_dir = os.path.join(self.face_tubes_base_directory,
                               split_name, emo_name)
        #print 'npy_dir:', npy_dir
        #print 'seq_id:', seq_id
        npy_glob = os.path.join(npy_dir, '{0}-*.npy'.format(seq_id))
        #print 'npy_glob:', npy_glob
        npy_files = glob.glob(npy_glob)
        # sort the filenames of tubes
        npy_files.sort()
        #print 'npy_files:', npy_files
        rval = []
        for f in npy_files:
            rval.append(np.load(f))
        return tuple(rval)

