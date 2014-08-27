# Copyright (c) 2014 University of Montreal, Thomas Rohee
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
import Pyro4
import subprocess
import struct

keypoints_name = ['left_eye_center', 'right_eye_center', 'nose_tip',
                  'mouth_left_corner', 'mouth_right_corner']


class DeepConvCascade(object):

    def __init__(self):
        self.local_path = os.path.join('C:/', 'deepconvcascade')
        self.imgList_local = os.path.join(self.local_path, 'imagelist.txt')
        self.bboxes_local = os.path.join(self.local_path, 'bbox.txt')
        self.bin_local = os.path.join(self.local_path, 'result.bin')

    def detect_face(self):
        subprocess.call(['FacePartDetect.exe', 'data',
                         self.imgList_local, self.bboxes_local])

    def detect_keypoints(self):
        subprocess.call(['TestNet.exe', self.bboxes_local, self.local_path,
                         'Input', self.bin_local])

    def format_bbox_file(self, img_name, data):
        """
        Format the file containing bounding box informations
        to be passed to the keypoints detector.

        Parameters
        ----------
        img_name: str
            name of the image (with extension)
        data: str
            bounding box informations with the form (imagepath int int int int)
        """

        with open(self.bboxes_local, 'w+') as fbbox:
            # remove path
            bboxes = data.split(' ')[1:]
            for i in range(0, len(bboxes), 4):
                cur_bbox = bboxes[i:i+4]
                fbbox.write(img_name + ' ' + ' '.join(cur_bbox) + '\n')

    def get_bbox_data(self):
        """ Return current bouding box data """
        with open(self.bboxes_local, 'r') as fbbox:
            data = fbbox.read()

        return data

    def get_keypoints(self, image_path, image_data):
        """
        Call necessary methods to return keypoints associated
        with the given imagepath.

        Parameters
        ----------
        image_path: str
            absolute path to the image from which to detect keypoints
        image_data: bin
            binary data of the image to process
        """
        print 'Using serializer', Pyro4.config.SERIALIZER

        base_path, ext = os.path.splitext(image_path)
        image_name = 'image' + ext
        img_local_path = os.path.join(self.local_path, image_name)

        with open(img_local_path, 'wb') as f:
            f.write(image_data)

        self.update_imagelist(img_local_path)
        self.detect_face()
        bbox_data = self.get_bbox_data()

        if self.is_face_detected(bbox_data):
            self.format_bbox_file(image_name, bbox_data)
            self.detect_keypoints()
            keypoints = self.kpts_from_binary()
        else:
            keypoints = []

        return keypoints

    def is_face_detected(self, data):
        return len(data.split(' ')) > 1

    def kpts_from_binary(self):
        """ Read keypoints coordinates from the generated binary file. """
        keypoints = []

        with open(self.bin_local, 'rb') as fresult:
            # read the binary file sequentially
            imageNum = struct.unpack('i', fresult.read(4))[0]  # C int (int32)
            pointNum = struct.unpack('i', fresult.read(4))[0]  # C int (int32)
            for i in range(imageNum):
                # C signed char (int8)
                valid = struct.unpack('b', fresult.read(1))

            for j in range(imageNum):
                kpts = {}
                for i in range(pointNum):
                    # C double (float64), 2 for each point
                    kpts[keypoints_name[i]] = struct.unpack('dd',
                                                            fresult.read(16))
                keypoints.append(kpts)

        return keypoints

    def update_imagelist(self, img_local_path):
        """
        Update the file, containing the images to process,
        used by the face detector.
        """
        with open(self.imgList_local, 'w+') as flist:
            flist.write('1\n')
            flist.write(img_local_path)


deepConvCascade = DeepConvCascade()

Pyro4.config.SERIALIZER = 'marshal'
daemon = Pyro4.Daemon(port=61605)
uri = daemon.register(deepConvCascade, 'deepconvcascade')
print uri
daemon.requestLoop()
