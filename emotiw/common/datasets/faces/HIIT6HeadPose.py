from NckuBasedDataset import NckuBasedDataset
import os
import math
import pickle

class HIIT6HeadPose(NckuBasedDataset):
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

def testWorks():
    save = 0
    if (save):
        HIIT6 = HIIT6HeadPose()
        output = open('HIIT6HeadPose.pkl', 'wb')
        data = HIIT6
        pickle.dump(data, output)
        output.close()
    else:
        pkl_file = open('HIIT6HeadPose.pkl', 'rb')
        HIIT6 = pickle.load(pkl_file)
        pkl_file.close()

    HIIT6.verify_samples()

if __name__ == '__main__':
    testWorks()


    
