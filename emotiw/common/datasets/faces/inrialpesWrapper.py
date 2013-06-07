# Wrapper to access headpose dataset given at
# http://www-prima.inrialpes.fr/perso/Gourier/Faces/HPDatabase.html
# coded by - abhi (abhiggarwal@gmail.com)

import os
import numpy as np
from scipy import io as sio
import math
import pickle

from NckuBasedDataset import NckuBasedDataset

# (685) 24.55 percent data with keypoints out of 2790 SAMPLES

class InrialpesHeadPose(NckuBasedDataset):
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
        self.relPaths = []
        self.out = 0
        idx = 0
        for root, subdirs, files in os.walk(self.absolute_base_directory):
            for file in files:
                if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg'):
                    subjectID = int(file[6:8])
                    series = int(file[8])
                    num = int(file[9:11]) 
                    nextPos = 14
                    relPath = os.path.join(root.split('/')[-1], file)
                    self.relPaths.append(relPath)
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
        self.read_json_keypoints()

def testWorks():
    save = 0
    if (save):
        obj = InrialpesHeadPose()
        output = open('inrialpes.pkl', 'wb')
        data = obj
        pickle.dump(data, output)
        output.close()
    else:
        pkl_file = open('inrialpes.pkl', 'rb')
        obj = pickle.load(pkl_file)
        pkl_file.close()

    obj.verify_samples()

if __name__ == '__main__':
    testWorks()
