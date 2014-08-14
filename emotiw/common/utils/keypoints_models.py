import apikey
import cv2
import numpy
import Pyro4
from facepp import API, File


facepp_keypoints = ['contour_chin', 'contour_left1', 'contour_left2',
                    'contour_left3', 'contour_left4', 'contour_left5',
                    'contour_left6', 'contour_left7', 'contour_left8',
                    'contour_left9', 'contour_right1', 'contour_right2',
                    'contour_right3', 'contour_right4', 'contour_right5',
                    'contour_right6', 'contour_right7', 'contour_right8',
                    'contour_right9',
                    'left_eye_bottom', 'left_eye_center',
                    'left_eye_left_corner', 'left_eye_lower_left_quarter',
                    'left_eye_lower_right_quarter', 'left_eye_pupil',
                    'left_eye_right_corner', 'left_eye_top',
                    'left_eye_upper_left_quarter',
                    'left_eye_upper_right_quarter',
                    'left_eyebrow_left_corner',
                    'left_eyebrow_lower_left_quarter',
                    'left_eyebrow_lower_middle',
                    'left_eyebrow_lower_right_quarter',
                    'left_eyebrow_right_corner',
                    'left_eyebrow_upper_left_quarter',
                    'left_eyebrow_upper_middle',
                    'left_eyebrow_upper_right_quarter',
                    'mouth_left_corner', 'mouth_lower_lip_bottom',
                    'mouth_lower_lip_left_contour1',
                    'mouth_lower_lip_left_contour2',
                    'mouth_lower_lip_left_contour3',
                    'mouth_lower_lip_right_contour1',
                    'mouth_lower_lip_right_contour2',
                    'mouth_lower_lip_right_contour3',
                    'mouth_lower_lip_top', 'mouth_right_corner',
                    'mouth_upper_lip_bottom', 'mouth_upper_lip_left_contour1',
                    'mouth_upper_lip_left_contour2',
                    'mouth_upper_lip_left_contour3',
                    'mouth_upper_lip_right_contour1',
                    'mouth_upper_lip_right_contour2',
                    'mouth_upper_lip_right_contour3', 'mouth_upper_lip_top',
                    'nose_contour_left1', 'nose_contour_left2',
                    'nose_contour_left3', 'nose_contour_lower_middle',
                    'nose_contour_right1', 'nose_contour_right2',
                    'nose_contour_right3', 'nose_left', 'nose_right',
                    'nose_tip',
                    'right_eye_bottom', 'right_eye_center',
                    'right_eye_left_corner', 'right_eye_lower_left_quarter',
                    'right_eye_lower_right_quarter', 'right_eye_pupil',
                    'right_eye_right_corner', 'right_eye_top',
                    'right_eye_upper_left_quarter',
                    'right_eye_upper_right_quarter',
                    'right_eyebrow_left_corner',
                    'right_eyebrow_lower_left_quarter',
                    'right_eyebrow_lower_middle',
                    'right_eyebrow_lower_right_quarter',
                    'right_eyebrow_right_corner',
                    'right_eyebrow_upper_left_quarter',
                    'right_eyebrow_upper_middle',
                    'right_eyebrow_upper_right_quarter']

deepconvcascade_keypoints = ['left_eye_center', 'mouth_left_corner',
                             'mouth_right_corner', 'nose_tip',
                             'right_eye_center']


def deep_conv_cascade_keypoints_dictlist_to_mat(kpts_dicts):
    return keypoints_dictlist_to_mat(kpts_dicts, deepconvcascade_keypoints)


def deep_conv_cascade_keypoints_mat_to_dictlist(kpts_mat):
    return keypoints_mat_to_dictlist(kpts_mat, deepconvcascade_keypoints)


def facepp_keypoints_dictlist_to_mat(kpts_dicts):
    return keypoints_dictlist_to_mat(kpts_dicts, facepp_keypoints)


def facepp_keypoints_mat_to_dictlist(kpts_mat):
    return keypoints_mat_to_dictlist(kpts_mat, facepp_keypoints)


def get_deep_cascade_keypoints(imagepath):
    try:
        convCascade = Pyro4.Proxy("PYRONAME:deepConvCascade")
        keypoints = convCascade.get_keypoints(imagepath)
    except Exception, e:
        print e
        keypoints = None

    return keypoints


def get_faceplusplus_keypoints(imagepath):
    # face++ API access
    api = API(apikey.API_KEY, apikey.API_SECRET, apikey.SERVER)

    img = cv2.imread(imagepath)
    kpts_list = []

    try:
        faces = api.detection.detect(img=File(imagepath))

        for face in faces['face']:
            result = api.detection.landmark(face_id=face['face_id'])
            keypoints = result['result'][0]['landmark']
            for kpt in keypoints:
                # x = int(img.shape[1] * keypoints[kpt]['x']/100)
                # y = int(img.shape[0] * keypoints[kpt]['y']/100)
                x = img.shape[1] * keypoints[kpt]['x']/100
                y = img.shape[0] * keypoints[kpt]['y']/100
                keypoints[kpt] = (x, y)
            kpts_list.append(keypoints)

    except Exception, e:
        print e

    return kpts_list


def keypoints_dictlist_to_mat(kpts_dicts, keypoints_names):
    l = []

    for d in kpts_dicts:
        vec = [d[key] for key in keypoints_names]
        l.append(vec)

    return numpy.array(l)  # nbfaces x nbkpts x 2 matrix


def keypoints_mat_to_dictlist(kpts_mat, keypoints_names):
    keypoints = []

    for row in kpts_mat:
        kpts_dict = dict((keypoints_names[j], tuple(coord))
                         for j, coord in enumerate(row))
        keypoints.append(kpts_dict)

    return keypoints
