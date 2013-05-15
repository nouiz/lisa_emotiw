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
import json

from emotiw.common.utils.pathutils import locate_data_path
from faceimages import FaceImagesDataset

# 4805 (24.025 percent) keypoints out of 20000 datasamples

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
        self.out = 0
   
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
                        if 'Data' in basename:
                            relPath = os.path.join('IIT6HeadPose','train', basename, file)
                            self.trainIndexes.append(idx)
                            self.relPaths.append(relPath)
                        elif 'Test' in basename:
                            relPath = os.path.join('IIT6HeadPose','test', basename, file)
                            self.testIndexes.append(idx)
                            self.relPaths.append(relPath)
                        self.images.append(relPath)
                        self.imageIndex[relPath] = idx

                        idx += 1
        self.read_json_keypoints()
         
    def get_keypoints_location(self, i):
        if i >= 0 and i < len(self.images):
            return self.keyPoints[i]
        else:
            return None

    def read_json_keypoints(self):
        self.keyPoints = []
        print 'length ofimageIndex'
        print len(self.imageIndex)
        for file in self.imageIndex :
             relPath = self.get_original_image_path_relative_to_base_directory(self.imageIndex[file])
             pathJson = os.path.join(self.absolute_base_directory, '..', 'mashapeKpts', 'HIIT6HeadPose', relPath)
             pathJson = os.path.splitext(pathJson)[0] + '.json'
             jsonData = open(pathJson)
             data = json.load(jsonData)
             if len(data) == 0:
                 self.keyPoints.append({})
             else:
                 keyDict = {}
                 self.out += 1
                 print self.out
                 for key in data[0]:
                     if key not in ['confidence', 'tid', 'attributes', 'height', 'width'] :
                         keyDict[key] = (data[0][key]['x'],data[0][key]['y'])
                         #print keyDict[key]
                     elif key in ['height', 'width']:
                         keyDict[key] = data[0][key]
                        # print keyDict[key]
                         
                 self.keyPoints.append(keyDict)

    
         
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
    print 'number of keypoints'
    print ncku.out
    for index in range(1):    
        print ncku.get_original_image_path(index)
        print ncku.get_head_pose(index)
        print ncku.get_subject_id_of_ith_face(index)
        print ncku.get_index_from_image_filename(ncku.images[index])
        print ncku.get_pan_tilt_and_roll(index)


if __name__ == '__main__':
    testWorks() 
