# Copyright (c) 2013 University of Montreal, Thomas Rohee
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

import math
import os
from faceimages import FaceImagesDataset


class FDDB(FaceImagesDataset):

    def __init__(self):
        super(FDDB, self).__init__('FDDB', 'faces/FDDB/')

        self.img_infos = []
        base_dir = self.absolute_base_directory
        infos_dir = os.path.join(base_dir, 'FDDB-folds/')
        folds = [fold for fold in os.listdir(infos_dir)
                 if 'ellipseList' in fold]

        for fold in folds:
            infos = open(infos_dir + fold).readlines()
            infos = [item.rstrip('\n') for item in infos]
            nb_ellipses = 0
            cur_ellipses = []
            cur_path = ''

            for item in infos:
                # Item can be a path, a number of faces or ellipse
                if '2002/' in item or '2003/' in item:
                    if len(cur_ellipses) > 0:
                        self.img_infos.append((cur_path+'.jpg', cur_ellipses))
                        cur_ellipses = []
                    cur_path = item
                elif nb_ellipses != 0:
                    # format ellipse data
                    ellipse = [int(float(elem)+0.5) for elem in item.split()]
                    bbox = self.ellipse_to_bbox(ellipse)
                    cur_ellipses.append(bbox)
                    nb_ellipses -= 1
                else:
                    nb_ellipses = int(item)

    def get_name(self):
        return 'FDDB dataset'

    def __len__(self):
        return len(self.img_infos)

    def get_original_image_path_relative_to_base_directory(self, i):
        return self.img_infos[i][0]

    def get_original_bbox(self, i):
        return self.img_infos[i][1]

    def ellipse_to_bbox(self, ellipse):
        """
        Method to get 4 points from an ellipse.
        The points correspond to the smallest circumscribed parallelogram
        around the ellipse.
        """
        maj_rad = ellipse[0]
        min_rad = ellipse[1]
        angle = ellipse[2]
        xcenter = ellipse[3]
        ycenter = ellipse[4]
        cosin = math.cos(math.radians(-angle))
        sin = math.sin(math.radians(-angle))
        
        x1 = cosin * (-min_rad) - sin * (-maj_rad) + xcenter
        y1 = sin * (-min_rad) + cosin * (-maj_rad) + ycenter
        x2 = cosin * (min_rad) - sin * (-maj_rad) + xcenter
        y2 = sin * (min_rad) + cosin * (-maj_rad) + ycenter
        x3 = cosin * (min_rad) - sin * (maj_rad) + xcenter
        y3 = sin * (min_rad) + cosin * (maj_rad) + ycenter
        x4 = cosin * (-min_rad) - sin * (maj_rad) + xcenter
        y4 = sin * (-min_rad) + cosin * (maj_rad) + ycenter

        return (x1, y1, x2, y2, x3, y3, x4, y4)

if __name__ == "__main__":
    FDDB()
