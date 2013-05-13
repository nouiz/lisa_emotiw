# Wrapper to access headpose dataset given at
# https://sites.google.com/site/diegotosato/ARCO/ihdp
# coded by - abhi (abhiggarwal@gmail.com)

import os
import numpy as np
import sys
import glob
from scipy import io as sio
import unicodedata

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
        self.numTrain = []
        #labels =sio.loadmat(os.path.join(self.absolute_base_directory, "or_label.mat"))
        import h5py
        labels = h5py.File(os.path.join(self.absolute_base_directory, "or_label_full.mat"))
        ind = -1
        for dtype in ["train", "test"] :
            labeltype = labels["or_label_" + dtype]
            name = labeltype["name"]
            roll = labeltype["roll"]
            pan = labeltype["pan"]
            tilt = labeltype["tilt"]
            ind += 1
            print dtype
            self.numTrain.append(name.shape[0])
            for i in range(self.numTrain[ind]):
                nStr = ''.join(chr(t) for t in labels[name[i,0]].value)
                self.imageIndex[nStr] = i
                if dtype == "test":
                    self.imageIndex[nStr] = i + self.numTrain[0]
                self.images.append(nStr)
                self.pan.append(labels[pan[i,0]].value[0,0])
                self.tilt.append(labels[tilt[i,0]].value[0,0])
                self.roll.append(labels[roll[i,0]].value[0,0])
                print nStr

        
    def __len__(self):
        return len(self.images)

    def get_standard_train_test_splits(self):
        return (range(self.numTrain[0]), range(self.numTrain[0], len(self.images)))

    def get_pan_tilt_and_roll(self, i):
        return (self.pan[i], self.tilt[i], self.roll[i])

    def get_original_image_path_relative_to_base_directory(self, i):
        if i>= 0 and i< len(self.images) :
            if i < self.numTrain:
                return os.path.join(self.absolute_base_directory,'train', self.images[i])
            else:
                return os.path.join(self.absolute_base_directory,'test', self.images[i])
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
    for index in range(1):    
        print ncku.get_original_image_path(index)
        print ncku.get_head_pose(index)
        print ncku.get_subject_id_of_ith_face(index)
        print ncku.get_index_from_image_filename(ncku.images[index])
        print ncku.get_pan_tilt_and_roll(index)
        #print ncku.get_standard_train_test_splits()


if __name__ == '__main__':
    testWorks() 
