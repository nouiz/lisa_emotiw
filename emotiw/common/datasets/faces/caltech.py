from helper import within_bounds
from faceimages import FaceImagesDataset
import os

class Caltech(FaceImagesDataset):
    def __init__(self):
        super(Caltech, self).__init__('Caltech 10000', 'faces/caltech/')
        
        self.lstImages = []
        self.keyPointsDict = []
        point_names = ['right_eye_pupil', 'left_eye_pupil', 'nose_tip', 'mouth_center']

        points_file = os.path.join(self.absolute_base_directory, 'WebFaces_GroundThruth.txt')
        f = open(points_file)

        for l in f.readlines():
            words = l.strip().split(' ')
            self.lstImages.append(os.path.join('Caltech_WebFaces', words[0]))
            #Caltech_WebFaces.tar hasn't been untar'd yet, so the path is
            #technically incorrect.

            prev_coord = None
            self.keyPointsDict.append({})
            for idx, w in enumerate(words[1:]):
                if idx % 2 == 0:
                    prev_coord = float(w)
                else:
                    self.keyPointsDict[-1][point_names[idx//2]] = (prev_coord, float(w))

        f.close()

    def get_original_image_path_relative_to_base_directory(self, i):
        if within_bounds(i, len(self.lstImages)):
            return self.lstImages[i]
        return None

    def get_keypoints_location(self, i):
        if within_bounds(i, len(self.keyPointsDict)):
            return self.keyPointsDict[i]
        return None

    def get_eyes_location(self, i):
        if within_bounds(i, len(self.keyPointsDict)):
            points = self.keyPointsDict[i]
            
            try:
                return (points['right_eye_pupil'], points['left_eye_pupil'])
            except KeyError:
                return None
        return None

    def __len__(self):
        return len(self.lstImages)
