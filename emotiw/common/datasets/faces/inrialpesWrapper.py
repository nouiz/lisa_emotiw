# Wrapper to access headpose dataset given at
# http://www-prima.inrialpes.fr/perso/Gourier/Faces/HPDatabase.html
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

class InrialpesHeadPose(FaceImagesDataset):
    def __init__(self):
        super(InrialpesHeadPose, self).__init__("InrialpesHeadPose", "faces/headpose/InrialpesHeadPose")

        print 'Working...'

        self.images = [] 
        self.tilt = []
        self.roll = []
        self.listOfSubjectId = []
        self.poses =[]
        self.imageIndex = {}
        self.pan = []

        idx = 0
        for root, subdirs, files in os.walk(self.absolute_base_directory):
            for file in files:
                if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg'):
                    subjectID = int(file[6:8])
                    series = int(file[8])
                    num = int(file[9:11]) 
                    nextPos = 14

                    #Getting PanAngle
                    if(file[11] == '-'):
                        panAngle = -1 * int(file[12:14])
                    elif (file[11] == '+'):
                        if(file[12] == '0'):
                            panAngle = 0
                            nextPos = 13
                        else:
                            panAngle = int(file[12:14])

                    #Getting TiltAngle
                    if(file[nextPos] == '-'):
                        tiltAngle = -1 * int(file[(nextPos+1):-4])
                    elif (file[nextPos] == '+'):
                        tiltAngle = +1 * int(file[(nextPos+1):-4])

                    self.poses.append(0)
                    self.tilt.append(math.sin(math.radians(tiltAngle)))
                    self.pan.append(math.sin(math.radians(panAngle)))
                    self.roll.append(None)                
                    self.listOfSubjectId.append(subjectID)
                    self.images.append(file)
                    self.imageIndex[file] = idx
                    idx += 1

                    #analyse the name
         
    def __len__(self):
        return len(self.images)

    def get_pan_tilt_and_roll(self, i):
        return (self.pan[i], self.tilt[i], self.roll[i])

    def get_original_image_path_relative_to_base_directory(self, i):
        return os.path.join(self.absolute_base_directory,'Subject' + str(self.listOfSubjectId[i]), self.images[i])

    def get_subject_id_of_ith_face(self, i):
        if i >= 0 and i < len(self.images):
            return self.listOfSubjectId[i]
        else:
            return None

    def get_head_pose(self, i):
        if i >= 0 and i < len(self.images):
            pan, tilt, roll = self.get_pan_tilt_and_roll(i)
            panAngle = math.degrees(math.asin(pan))
            tiltAngle = math.degrees(math.asin(tilt))

            if(tiltAngle >= -90 and tiltAngle <=-60):
                #subject vertically looking up
                if(panAngle >= -90 and panAngle <= -30):
                    self.poses[i] = 8
                elif(panAngle >= -15 and panAngle <= 15):
                    self.poses[i] = 7
                elif(panAngle >= 30 and panAngle <= 90):
                    self.poses[i] = 6
            elif(tiltAngle >= -30 and tiltAngle <= 30):
                #subject verticall looking in the center 
                if(panAngle >= -90 and panAngle <= -60):
                    self.poses[i] = 10
                elif(panAngle >= -45 and panAngle <= -30):
                    self.poses[i] = 5
                elif(panAngle >= -15 and panAngle <= 15):
                    self.poses[i] = 4
                elif(panAngle >= 30 and panAngle <= 45):
                    self.poses[i] = 3
                elif(panAngle >= 60 and panAngle <= 90):
                    self.poses[i] = 9    
            else:
                #subject vertically looking up
                if(panAngle >= -90 and panAngle <= -30):
                    self.poses[i] = 2
                elif(panAngle >= -15 and panAngle <= 15):
                    self.poses[i] = 1
                elif(panAngle >= 30 and panAngle <= 90):
                    self.poses[i] = 0

            return self.poses[i]
        else:
            return None

    def get_index_from_image_filename(self, imgFileName):
        return self.imageIndex[imgFileName]


def testWorks():

    ncku = InrialpesHeadPose()
    print len(ncku)
    for index in range(10):    
        print ncku.get_original_image_path(index)
        print ncku.get_head_pose(index)
        print ncku.get_subject_id_of_ith_face(index)
        print ncku.get_index_from_image_filename(ncku.images[index])
        print ncku.get_pan_tilt_and_roll(index)


if __name__ == '__main__':
    testWorks() 
