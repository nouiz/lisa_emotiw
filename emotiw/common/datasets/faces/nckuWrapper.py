# Wrapper to access headpose dataset given at
# http://robotics.csie.ncku.edu.tw/Databases/FaceDetect_PoseEstimate.htm#Our_Database_)
# coded by - abhi (abhiggarwal@gmail.com)

import os
import numpy as np
import math
from NckuBasedDataset import NckuBasedDataset
import pickle

class NCKUHeadPose(NckuBasedDataset):
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
        self.relPaths = []
        self.out = 0
        idx = 0
        for root, subdirs, files in os.walk(self.absolute_base_directory):
            for file in files:
                if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg'):
                    subjectID = int(file[2:4])
                    angle = file[5:-4]
                    relPath = os.path.join(root.split('/')[-1], file)
                    self.relPaths.append(relPath)
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
        self.read_json_keypoints()

def testWorks():
    save = 0
    if (save):
        obj = NCKUHeadPose()
        output = open('ncku.pkl', 'wb')
        data = obj
        pickle.dump(data, output)
        output.close()
    else:
        pkl_file = open('ncku.pkl', 'rb')
        obj = pickle.load(pkl_file)
        pkl_file.close()

    obj.verify_samples()

if __name__ == '__main__':
    testWorks()
