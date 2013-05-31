from helper import read_points, within_bounds
from faceimages import FaceImagesDataset
import os

class BioID(FaceImagesDataset):
    def __init__(self):
        super(BioID, self).__init__('BioID', 'faces/BioID/')
        point_names = ['right_eye_pupil', 'left_eye_pupil', 'mouth_right_corner', 'mouth_left_corner',
                        'right_eyebrow_outer_end', 'right_eyebrow_inner_end', 'left_eyebrow_inner_end',
                        'left_eyebrow_outer_end', 'face_left', 'right_eye_outer_corner', 'right_eye_inner_corner',
                        'left_eye_inner_corner', 'left_eye_outer_corner', 'face_right', 'nose_tip', 'right_nostril', 
                        'left_nostril', 'mouth_top_lip', 'mouth_bottom_lip', 'chin_center']

        self.lstImages = []
        self.keyPointsDict = []

        img_dir = 'BioID-FaceDatabase-V1.2/'
        self.lstImages = [os.path.join(img_dir, 'BioID_' + str(i).zfill(4) + '.pgm')  for i in range(1520)]

        points_dir = 'points_20/'
        points_path = os.path.join(self.absolute_base_directory, points_dir)
        
        points = read_points(self.lstImages, 
                            lambda x: x[-14:-3].lower() + 'pts', 
                            directory=points_path)

        #Note: all points are present for all images in this dataset.
        #If this assumption is incorrect, point names to coordinates will
        #be incorrect.
        for img_points in points:
            self.keyPointsDict.append({})
            prev_coord = None

            for idx, point in enumerate(img_points):
                if idx % 2 == 0:
                    prev_coord = point
                else:
                    self.keyPointsDict[-1][point_names[idx//2]] = (prev_coord, point)
                    prev_coord = None

    def __len__(self):
        return len(self.lstImages)

    def get_original_image_path_relative_to_base_directory(self, i):
        return self.lstImages[i]

    def get_eyes_location(self, i):
        if within_bounds(i, len(self.keyPointsDict)):
            points = self.keyPointsDict[i]
            try:
                try:
                    #the interface calls 'left' the right of the subject and vice versa for this function.
                    left = [x0 + (x1 - x0)/2 for x0, x1 in zip(points['left_eye_outer_corner'], points['left_eye_inner_corner'])]
                except KeyError:
                    left = [None, None]
            finally:
                try:
                    try:
                        right = [x0 + (x1 - x0)/2 for x0, x1 in zip(points['right_eye_inner_corner'], points['right_eye_outer_corner'])]
                    except KeyError:
                        right = [None, None]
                finally:
                    right.extend(left) 
                    return right
        else:
            return [None, None, None, None]

    def get_keypoints_location(self, i):
        if within_bounds(i, len(self.keyPointsDict)):
            return self.keyPointsDict[i]
        else:
            return None


def testWorks():
    save = 0
    import pickle
    if (save):
        obj = BioID()
        output = open('BioID.pkl', 'wb')
        data = obj
        pickle.dump(data, output)
        output.close()
    else:
        pkl_file = open('BioID.pkl', 'rb')
        obj = pickle.load(pkl_file)
        pkl_file.close()

    obj.verify_samples()

if __name__ == '__main__':
    testWorks()
