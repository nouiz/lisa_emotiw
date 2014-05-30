import os
from helper import within_bounds, flatten
from faceimages import FaceImagesDataset
import h5py
import math

class AFW(FaceImagesDataset):
    def __init__(self):
        super(AFW, self).__init__('AFW', 'faces/AFW/testimages/')
        self.lstImages = []
        self.keyPointsDict = []
        self.yawPitchRoll = []
        self.bbox = []
        point_names = ['right_eye_center', 'left_eye_center', 'nose_tip', 'mouth_right_corner', 
                        'mouth_center', 'mouth_left_corner']

        f = h5py.File(os.path.join(self.absolute_base_directory, 'anno.mat'), 'r')
        self.lstImages = ["".join(map(lambda x: chr(x), f[i].value)) for i in f['anno'].value[0]]

        #magic.
        #see show_keypoints.py
        #NOTE: might not take into account multiple pointsets in the same image
        #corresponding to different subjects.
        points = [flatten(zipped) for zipped in
                        [zip(coords[0][0], coords[0][1]) for coords in
                            [map(lambda a: f[a].value, coord_ref) for coord_ref in
                                [flatten(f[coord_col]) for coord_col in f['anno'].value[3]]]]]
        points = [[point for point in keypoints if not math.isnan(point)] for keypoints in points]

        bbox = [flatten(zipped) for zipped in
                [zip(coords[0][0], coords[0][1]) for coords in
                    [map(lambda a: f[a].value, coord_ref) for coord_ref in
                        [flatten(f[coord_col]) for coord_col in f['anno'].value[1]]]]]

        yaw_pitch_roll = [flatten(zipped) for zipped in
                            [zip(coords[0][0], coords[0][1], coords[0][2]) for coords in
                                [map(lambda a: f[a].value, coord_ref) for coord_ref in
                                    [flatten(f[coord_col]) for coord_col in f['anno'].value[2]]]]]

        f.close() 

        for img_ypr in yaw_pitch_roll:
            self.yawPitchRoll.append(img_ypr)

        for img_bounds in bbox:
            self.bbox.append(img_bounds)

        for img_points in points:
            self.keyPointsDict.append({})
            prev_point = None

            for i, point in enumerate(img_points):
                if i % 2 == 0:
                    prev_point = float(point)
                else:
                    self.keyPointsDict[-1][point_names[i//2]] = (prev_point, float(point))
                    prev_point = None
    
    def get_original_bbox(self, i):
        #Note: no standard is suggested, let alone prescribed, for
        #the return values of bounds. We have chosen to return the
        #[x0, y0, x1, y1] list of coordinates.
        if within_bounds(i, len(self.bbox)):
            return self.bbox[i]
        return None

    def get_eyes_location(self, i):
        if within_bounds(i, len(self.keyPointsDict)):
            points = self.keyPointsDict[i]
            
            try:
                try:
                    left = list(points['right_eye_center'])
                    if math.isnan(left[0]) or math.isnan(left[1]):
                        raise KeyError()
                except KeyError:
                    left = [None, None]
            finally:
                try:
                    try:
                        right = list(points['left_eye_center'])
                        if math.isnan(right[0]) or math.isnan(right[1]):
                            raise KeyError()
                    except KeyError:
                        right = [None, None]
                finally:
                    left.extend(right)
                    return left
        else:    
            return None

    def __len__(self):
        return len(self.lstImages)

    def get_keypoints_location(self, i):
        if within_bounds(i, len(self.keyPointsDict)):
            return self.keyPointsDict[i]

    def get_pan_tilt_and_roll(self, i):
        if within_bounds(i, len(self.yawPitchRoll)):
            return self.yawPitchRoll[i]

    def get_original_image_path_relative_to_base_directory(self, i):
        if within_bounds(i, len(self.lstImages)):
            return self.lstImages[i]
        return None

def testWorks():
    save = 1
    import pickle
    if (save):
        obj = AFW()
        output = open('afw.pkl', 'wb')
        data = obj
        pickle.dump(data, output)
        output.close()
    else:
        pkl_file = open('afw.pkl', 'rb')
        obj = pickle.load(pkl_file)
        pkl_file.close()

    obj.verify_samples()

if __name__ == '__main__':
    testWorks()
         
