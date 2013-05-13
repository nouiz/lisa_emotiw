# Wrapper to access headpose dataset given at
# https://sites.google.com/site/diegotosato/ARCO/iit
# coded by - abhi (abhiggarwal@gmail.com)

import os
import numpy as np
import sys
import glob
from scipy import io as sio
import unicodedata
import math

from emotiw.common.utils.pathutils import locate_data_path
from faceimages import FaceImagesDataset

class HIIT6HeadPose(FaceImagesDataset):
    def __init__(self):
        super(HIIT6HeadPose, self).__init__("HIIT6HeadPose", "faces/headpose/HIIT6HeadPose")

        print 'Working...'

        self.images = [] 
        self.tilt = []
        self.imageIndex = {}
        self.pan = []
        self.roll = []
        self.trainIndexes = []
        self.testIndexes = []
        self.relPaths = []
        
   
        idx = 0
        for root, subdirs, files in os.walk(self.absolute_base_directory):
            if 'rear' in root:
                continue
            else:
                #print root
                for file in files:
                    if os.path.splitext(file)[1].lower() in ('.png'):
                        basename = os.path.basename(root) 
                        if 'frnt' in basename :
                            self.pan.append(0)
                        elif 'frlf' in basename :
                            self.pan.append(math.sin(math.radians(45)))
                        elif 'frrg' in basename :
                            self.pan.append(math.sin(math.radians(-45)))
                        elif ( 'left' in basename):
                            self.pan.append(math.sin(math.radians(90)))
                        elif ( 'right' in basename):
                            self.pan.append(math.sin(math.radians(-90)))
                     
                        self.tilt.append(None)
                        self.roll.append(None)
                             
                    #print os.path.join(root, file)
                        self.images.append(file)
                        self.imageIndex[file] = idx
                        if 'Data' in basename:
                            self.trainIndexes.append(idx)
                            self.relPaths.append( os.path.join(self.absolute_base_directory, 'IIT6HeadPose', 'train', basename, file))
                        elif 'Test' in basename:
                            self.testIndexes.append(idx)
                            self.relPaths.append( os.path.join(self.absolute_base_directory, 'IIT6HeadPose', 'test', basename, file))
                        idx += 1
                                       
                    #analyse the name
         
    def __len__(self):
        return len(self.images)

    def get_pan_tilt_and_roll(self, i):
        if i >= 0 and i < len(self.images):
            return (self.pan[i], self.tilt[i], self.roll[i])
        else:
            return None 

    def get_original_image_path_relative_to_base_directory(self, i):
        return self.relPaths[i]

    def get_subject_id_of_ith_face(self, i):
        return None

    def get_head_pose(self, i):
        return None
    def get_index_from_image_filename(self, imgFileName):
        return self.imageIndex[imgFileName]


def testWorks():

    ncku = HIIT6HeadPose()
    print len(ncku)
    for index in range(76):    
        print ncku.get_original_image_path(index)
        print ncku.get_head_pose(index)
        print ncku.get_subject_id_of_ith_face(index)
        print ncku.get_index_from_image_filename(ncku.images[index])
        print ncku.get_pan_tilt_and_roll(index)


if __name__ == '__main__':
    testWorks() 
