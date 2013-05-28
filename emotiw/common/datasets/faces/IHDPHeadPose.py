# Wrapper to access headpose dataset given at
# http://robotics.csie.ncku.edu.tw/Databases/FaceDetect_PoseEstimate.htm#Our_Database_)
# coded by - abhi (abhiggarwal@gmail.com)

from NckuBasedDataset import NckuBasedDataset
import os

class IHDPHeadPose(NckuBasedDataset):
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
        self.relPaths = []
        self.numTrain = 0
        self.numTest = 0
        self.out = 0
        #labels =sio.loadmat(os.path.join(self.absolute_base_directory, "or_label.mat"))
        import h5py
        labels = h5py.File(os.path.join(self.absolute_base_directory, "or_label_full.mat"), 'r')
        for dtype in ["train", "test"] :
            labeltype = labels["or_label_" + dtype]
            name = labeltype["name"]
            roll = labeltype["roll"]
            pan = labeltype["pan"]
            tilt = labeltype["tilt"]
            num = 0
            #print dtype
            for i in range(name.shape[0]):
                nStr = ''.join(chr(t) for t in labels[name[i,0]].value)
                self.imageIndex[nStr] = i
                self.relPaths.append(os.path.join(dtype, nStr))
                if dtype == "test":
                    self.imageIndex[nStr] = i + self.numTrain
                self.images.append(os.path.join(dtype, nStr))
                self.pan.append(labels[pan[i,0]].value[0,0])
                self.tilt.append(labels[tilt[i,0]].value[0,0])
                self.roll.append(labels[roll[i,0]].value[0,0])
                #print nStr
                num += 1

            if dtype in ["train"] :
                self.numTrain = num
            else:
                self.numTest = num

        self.read_json_keypoints()


    def get_standard_train_test_splits(self):
        return (range(self.numTrain), range(self.numTrain, len(self.images)))

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
