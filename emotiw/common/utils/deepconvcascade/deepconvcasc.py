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

import Pyro4
import subprocess
import struct


class DeepConvCascade(object):

    def __init__(self):
        self.path = ''
        self.bin_fname = 'result.bin'
        self.imglist_fname = 'imagelist.txt'
        self.bboxes_fname = 'bbox.txt'

    def detect_face(self):
        subprocess.call(['FacePartDetect.exe', 'data',
                         self.imglist_fname, self.bboxes_fname])

    def detect_keypoints(self):
        subprocess.call(['TestNet.exe', self.bboxes_fname, self.path,
                         'Input', self.bin_fname])

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

        with open(self.bboxes_fname, 'w+') as fbbox:
            # remove path
            bboxes = data.split(' ')[1:]
            for i in range(0, len(bboxes), 4):
                cur_bbox = bboxes[i:i+4]
                fbbox.write(img_name + ' ' + ' '.join(cur_bbox) + '\n')

    def get_bbox_data(self):
        """ Return current bouding box data """
        with open(self.bboxes_fname, 'r') as fbbox:
            data = fbbox.read()

        return data

    def get_keypoints(self, imagepath):
        """
        Call necessary methods to return keypoints associated
        with the given imagepath.

        Parameters
        ----------
        imagepath: str
            absolute path to the image from which to detect keypoints
        """

        self.set_image_path(imagepath)
        img_name = imagepath.split('/')[-1]

        self.update_imagelist(img_name)
        self.detect_face()
        data = self.get_bbox_data()

        if self.is_face_detected(data):
            self.format_bbox_file(img_name, data)
            self.detect_keypoints()
            keypoints = self.kpts_from_binary()
        else:
            keypoints = None

        return keypoints

    def is_face_detected(self, data):
        return len(data.split(' ')) > 1

    def kpts_from_binary(self):
        """ Read keypoints coordinates from the generated binary file. """
        keypoints = []

        with open(self.bin_fname, 'rb') as fresult:
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
                    kpts[str(i)] = struct.unpack('dd', fresult.read(16))
                keypoints.append(kpts)

        return keypoints

    def set_image_path(self, imagepath):
        # Replace /data/lisa by Q: (letter of network drive mounted on burns)
        self.path = 'Q:/' + '/'.join(imagepath.split('/')[3:-1]) + '/'

    def update_imagelist(self, img_name):
        """
        Update the file, containing the images to process,
        used by the face detector.
        """

        with open(self.imglist_fname, 'w+') as flist:
            flist.write('1\n')
            flist.write(self.path + img_name)


deepConvCascade = DeepConvCascade()

daemon = Pyro4.Daemon()
ns = Pyro4.locateNS()
uri = daemon.register(deepConvCascade)
ns.register("deepConvCascade", uri)
daemon.requestLoop()
