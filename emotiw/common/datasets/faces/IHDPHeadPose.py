# Wrapper to access headpose dataset given at
# https://sites.google.com/site/diegotosato/ARCO/ihdp
# coded by - abhi (abhiggarwal@gmail.com)

import os
import numpy as np
import sys
import glob
from scipy import io as sio
import unicodedata
import json

#16599 keypoints available in only 25.038 percent data out of 66295 samples

from emotiw.common.utils.pathutils import locate_data_path
from faceimages import FaceImagesDataset

class IHDPHeadPose(FaceImagesDataset):
    def __init__(self):
        super(IHDPHeadPose, self).__init__("IHDPHeadPose", "faces/headpose/IHDPHeadPose")

        print 'Working...'

        self.images = [] 
        self.tiltAngle = []
        self.listOfSubjectId = []
        self.poses =[]
        self.imageIndex = {}
        self.pan = []
        self.tilt = []
        self.roll = []
        self.numTrain = 0
        self.numTest = 0
        self.out = 0
        #labels =sio.loadmat(os.path.join(self.absolute_base_directory, "or_label.mat"))
        import h5py
        labels = h5py.File(os.path.join(self.absolute_base_directory, "or_label_full.mat"))
        for dtype in ["train", "test"] :
            labeltype = labels["or_label_" + dtype]
            name = labeltype["name"]
            roll = labeltype["roll"]
            pan = labeltype["pan"]
            tilt = labeltype["tilt"]
            num = 0
            print dtype
            for i in range(name.shape[0]):
                nStr = ''.join(chr(t) for t in labels[name[i,0]].value)
                self.imageIndex[nStr] = i
                if dtype == "test":
                    self.imageIndex[nStr] = i + self.numTrain
                self.images.append(nStr)
                self.pan.append(labels[pan[i,0]].value[0,0])
                self.tilt.append(labels[tilt[i,0]].value[0,0])
                self.roll.append(labels[roll[i,0]].value[0,0])
                print nStr
                num += 1

            if dtype in ["train"] :
                self.numTrain = num
            else:
                self.numTest = num

        self.read_json_keypoints()
         
    def get_keypoints_location(self, i):
        if i >= 0 and i < len(self.images):
            return self.keyPoints[i]
        else:
            return None

    def read_json_keypoints(self):
        self.keyPoints = []
        for file in self.imageIndex :
             relPath = self.get_original_image_path_relative_to_base_directory(self.imageIndex[file])

             pathJson = os.path.join(self.absolute_base_directory, '..', 'mashapeKpts', 'IHDPHeadPose', relPath)
             jsonFlip = os.path.splitext(pathJson)[0]

             if('flip' in jsonFlip):
                 pathJson = jsonFlip[:-1] + '.json'
                 #print pathJson
             else:
                 pathJson = jsonFlip + '.json'

             jsonData = open(pathJson)
             data = json.load(jsonData)
             if len(data) == 0:
                 self.keyPoints.append({})
             else:
                 keyDict = {}
                 self.out += 1
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

    def get_standard_train_test_splits(self):
        return (range(self.numTrain), range(self.numTrain, len(self.images)))

    def get_pan_tilt_and_roll(self, i):
        return (self.pan[i], self.tilt[i], self.roll[i])

    def get_original_image_path_relative_to_base_directory(self, i):
        if i>= 0 and i< len(self.images) :
            if i < self.numTrain:
                return os.path.join('train', self.images[i])
            else:
                return os.path.join('test', self.images[i])
        else:
            return None

    def get_subject_id_of_ith_face(self, i):
            return None

    def get_head_pose(self, i):
            return None

    def get_index_from_image_filename(self, imgFileName):
        return self.imageIndex[imgFileName]


def testWorks():

    ncku = IHDPHeadPose()
    print len(ncku)
    print "number of keys"
    print ncku.out
    for index in range(1):    
        print ncku.get_original_image_path(index)
        print ncku.get_head_pose(index)
        print ncku.get_subject_id_of_ith_face(index)
        print ncku.get_index_from_image_filename(ncku.images[index])
        print ncku.get_pan_tilt_and_roll(index)
        print ncku.get_keypoints_location(index)
        #print ncku.get_standard_train_test_splits()


if __name__ == '__main__':
    testWorks() 
