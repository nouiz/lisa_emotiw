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
AFEW2Test contains the test data distributed with the EmotiW challenge.
"""
import cPickle
import glob
import os

from emotiw.common.utils.pathutils import locate_data_path
import afew
import afew2


class AFEW2TestImageSequenceDataset(afew2.AFEW2ImageSequenceDataset):
    # These directories are relative to the data path
    # Since there was no room at LISA for the data to be in the usual datapath,
    # there is a symlink instead.
    base_dir = "faces/EmotiWTest/Test_Vid_Distr/ExtractedFrame"
    picasa_boxes_base_dir = "faces/EmotiWTest/Test_Vid_Distr/BoundBoxData"
    face_tubes_base_dir = base_dir # "faces/EmotiWTest/Test_Vid_Distr/FaceTube"
    face_tubes_boxes_base_dir = "faces/EmotiWTest/Test_Vid_Distr/picasa_tubes_pickles"

    def __init__(self, preload_facetubes=False, preproc=[], size=(96, 96)):
        """
        If preload_facetubes is True, all facetubes will be loaded
        when the dataset is built.
        """
        self.facetubes_to_filter = None
        for opt in preproc:
            # TODO: fix paths
            if opt == "smooth":
                #raise NotImplementedError()
                # Use the bounding-boxes smoothed version of the face tubes.
                #self.face_tubes_base_dir = ("faces/EmotiWTest/smooth_picasa_face_tubes_%s_%s"
                #                            "/numpy_arr/concatenate")%(size[0], size[1])
                # TODO: return the correct bounding boxes coordinates
                # corresponding to the smoothed face tubes.  For the
                # moment, we are using the default picasa boxes coordinates.
                self.face_tubes_boxes_base_dir = ("faces/EmotiWTest/smooth_picasa_face_tubes_%s_%s"
                                                  "/picasa_tubes_pickles")%(size[0], size[1])

            if opt == "remove_background_faces":
                raise NotImplementedError("The option %s is not supported yet."%opt)
                # Remove background faces as many as possible from the dataset.
                # NOTE: for the moment, only the smoothed version of face tubes
                # is supported.
                # Path to the dictionary giving for each dataset, the list of
                # face tubes corresponding to background faces/objects.
                abs_dir = locate_data_path("faces/EmotiW")
                filename = os.path.join(abs_dir, "background_faces_info.pkl")
                try:
                    f = open(filename, 'rb')
                    background_faces_info = cPickle.load(f)
                    f.close()
                    # Retrieve the list of background faces (clip_id, facetube_id)
                    # for the given dataset that will be filtered out.
                    if self.face_tubes_base_dir in background_faces_info:
                        self.facetubes_to_filter = background_faces_info[self.face_tubes_base_dir]
                    else:
                        print ("The option %s doesn't support the dataset %s"
                               %(opt, self.face_tubes_base_dir))
                except IOError as e:
                    print e

        super(afew.AFEWImageSequenceDataset, self).__init__("AFEW2Test")

        self.absolute_base_directory = locate_data_path(self.base_dir)
        self.absolute_picasa_boxes_base_directory = locate_data_path(
                self.picasa_boxes_base_dir)
        self.face_tubes_base_directory = locate_data_path(
                self.face_tubes_base_dir)
        self.face_tubes_boxes_base_directory = locate_data_path(
                self.face_tubes_boxes_base_dir)

        self.preload_facetubes = preload_facetubes
        self.preproc = preproc
        self.size = size
        self.imagesequences = []
        self.labels = []
        self.seq_info = []
        self.trainIndexes = []  # unused
        self.validIndexes = []  # unused
        self.testIndexes = []

        idx = 0
        # find all clips
        clip_names = glob.glob(os.path.join(self.absolute_base_directory, '*'))
        clip_names.sort()

        # For each clip
        for clip_name in clip_names:
            rel_img_dir = os.path.join(self.base_dir, clip_name)
            im_seq = afew.AFEWImageSequence(
                "AFEW2Test",
                rel_img_dir,
                '*.png',
                None)

            im_seq.set_picasa_path_substitutions(
                {self.base_dir: self.picasa_boxes_base_dir,
                 '_.png': '.txt',
                 '_.jpg': '.txt'},
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

            # No label
            self.labels.append(None)

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
        seq_id = self.seq_info[i].split("/")[-1]
        path = os.path.join(self.face_tubes_boxes_base_directory, "%s.pkl"%seq_id)
        try:
            f = open(path, 'rb')
            bbox_coords = cPickle.load(f)
            f.close()
            if seq_id in bbox_coords:
                rval = bbox_coords[seq_id]
        except IOError:
            pass
        return rval

    def get_facetubes(self, i):
        if self.preload_facetubes:
            return self.facetubes[i]
        else:
            return self.load_facetubes(self.seq_info[i])

    def load_facetubes(self, clip_name):
        # Not available yet
        return None
