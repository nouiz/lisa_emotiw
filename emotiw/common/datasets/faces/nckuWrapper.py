# Wrapper to access headpose dataset given at
# http://robotics.csie.ncku.edu.tw/Databases/FaceDetect_PoseEstimate.htm#Our_Database_)
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

class NCKUHeadPose(FaceImagesDataset):
    def __init__(self):
        super(NCKUHeadPose, self).__init__("NCKUHeadPose", "faces/headpose/ncku")

        print 'Working...'

        self.images = [] 
        self.tilt = []
        self.listOfSubjectId = []
        self.poses =[]
        self.imageIndex = {}
        self.pan = []
        self.roll = []

        idx = 0
        for root, subdirs, files in os.walk(self.absolute_base_directory):
            for file in files:
                if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg'):
                    subjectID = int(file[2:4])
                    angle = file[5:-4]
                    
                    #because the dataset follows opposite convention to what we are using
                    if(angle[0] == '-'):
                        angle = +1 * int(angle[1:])
                    elif (angle[0] == '+'):
                        angle = -1 * int(angle[1:])
                    else:
                        angle = 0


                    if(angle >= -90 and angle <=-65):
                        self.poses.append(10)
                    elif(angle >= -60 and angle <=-25):
                        self.poses.append(5)
                    elif(angle >= -20 and angle <= 20):
                        self.poses.append(4)
                    elif(angle >= 25 and angle <= 60):
                        self.poses.append(3)
                    elif(angle >= 65 and angle <= 90):
                        self.poses.append(9)

                    self.tilt.append(0)
                    self.roll.append(0)
                    self.pan.append(math.sin(math.radians(angle)))
                    self.listOfSubjectId.append(subjectID)
                   # print file
                    self.images.append(file)
                    self.imageIndex[file] = idx
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
        return os.path.join(self.absolute_base_directory,'Subject' + str(self.listOfSubjectId[i]), self.images[i])

    def get_subject_id_of_ith_face(self, i):
        if i >= 0 and i < len(self.images):
            return self.listOfSubjectId[i]
        else:
            return None

    def get_head_pose(self, i):
        if i >= 0 and i < len(self.images):
            return self.poses[i]
        else:
            return None
    def get_index_from_image_filename(self, imgFileName):
        return self.imageIndex[imgFileName]


def testWorks():

    ncku = NCKUHeadPose()
    print len(ncku)
    for index in range(76):    
        print ncku.get_original_image_path(index)
        print ncku.get_head_pose(index)
        print ncku.get_subject_id_of_ith_face(index)
        print ncku.get_index_from_image_filename(ncku.images[index])
        print ncku.get_pan_tilt_and_roll(index)


if __name__ == '__main__':
    testWorks() 
