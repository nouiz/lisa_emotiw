import cv, cv2
import glob
import numpy
import os
from scipy import io as sio
import subprocess
import sys
import warnings


debug = False

def get_output_size(path, width = 1024):
    """
    Read the aspect ration information and return
    an output size string based on aspect ratio
    and given width

    Params
    -----
    path: avi file path
    widht: output image width
    """

    command = ["ffprobe", path]
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    res = p.communicate()[0]
    asr = res[res.find("DAR "):].split(']')[0][4:].split(':')
    orig_dims = res[res.find("Stream #0.0"):].split('[')[0].split(',')[-1]
    if 'x' not in orig_dims:
        #orig_dims = [720, 576]
        print 'The original dimensions for the images in the avi file are not found for clip %s!'%path
        print 'Script will exit and we will process the next clip if necessary.'
        sys.exit(0)
    else:
        orig_dims = orig_dims.split('x')
        if len(orig_dims) != 2:
            #orig_dims = [720, 576]
            print 'The original dimensions for the images in the avi file are not found for clip %s!'%path
            print 'Script will exit and we will process the next clip if necessary.'
            sys.exit(0)
        else:
            orig_dims = [int(orig_dims[0]), int(orig_dims[1])]
    try:
        asr = map(float, asr)
    except ValueError:
        asr = res[res.find("DAR "):].split(' ')[1].split(':')
        asr[1] = asr[1].split(",")[0]
        asr = map(float, asr)

    height = int(width / (asr[0]/asr[1]))
    return asr, orig_dims, [width, height]


def extract_frames(src, dest, asr):
    """
    Extract frame images from avi

    Params
    -----
    src: src file
    dest: dest file pattern
    asr: aspect ration string e.g. 1024x576
    """

    command = ["ffmpeg", "-i", src, "-s", asr, "-qscale", "1", dest]
    subprocess.check_call(command)


def get_bbox(avi_path, extracted_frames_path, ramanan_keypts_path, bbox_path, bbox_on_img_path = None):
    # TODO: pas de '/' a la fin des arguments (chemins complets).
    if extracted_frames_path[-1] == '/':
        extracted_frames_path = extracted_frames_path[:-1]
    clip = extracted_frames_path.split('/')[-1]
    print 'clip id: ', clip

    asr, orig_dims, dims = get_output_size(avi_path)
    scale_x = dims[0] / float(orig_dims[0])
    scale_y = dims[1] / float(orig_dims[1])

    # TODO: jpg ou png?
    if debug:
        frames_paths = glob.glob("{}/*jpg".format(extracted_frames_path))
    else:
        frames_paths = glob.glob("{}/*png".format(extracted_frames_path))
    frames_paths.sort()

    if not os.path.exists(ramanan_keypts_path):
        # TODO: si pas de ramanan donc pas de bounding box pour la frame.
        print 'Did not find any ramanan mat file for clip ', clip
        return 0
    matfile = sio.loadmat(ramanan_keypts_path)
    all_xy = matfile['points']['xy']

    frame_i = 0
    for i, xy in enumerate(all_xy):
        if len(xy[0]) == 1:
            continue
        #if len(xy[0]) != 68 and len(xy[0]) != 39:
        #    import pdb; pdb.set_trace()
        #if len(xy) != 1:
        #    import pdb; pdb.set_trace()
        if frame_i == len(frames_paths):
            print 'There are more ramanan keypoints (%s) than frames (%s)'%(len(all_xy), len(frames_paths))
            return 0
        frame_path = frames_paths[frame_i]
        frame_i += 1
        basename, ext = os.path.splitext(frame_path)
        name = basename.split('/')[-1]
        img = cv2.imread(frame_path)

        xs = (xy[0][:, 0] + xy[0][:,2])*0.5
        ys = (xy[0][:, 1] + xy[0][:,3])*0.5
        xs *= scale_x
        ys *= scale_y

        pts_idx_dict_68 = {0: 'nostrils_center', 1: 'right_nostril_inner_end', 2: 'right_nostril', 3: 'left_nostril_inner_end', 4: 'left_nostril', 5: 'nose_tip', 6: 'nose_ridge_bottom', 7: 'nose_ridge_top', 8: 'nose_center_top', 9: 'right_eye_inner_corner', 10: 'right_eye_bottom_inner_midpoint', 11: 'right_eye_bottom_outer_midpoint', 12: 'right_eye_top_inner_midpoint', 13: 'right_eye_top_outer_midpoint', 14: 'right_eye_outer_corner', 15: 'right_eyebrow_outer_end', 16: 'right_eyebrow_outer_midpoint', 17: 'right_eyebrow_center', 18: 'right_eyebrow_inner_midpoint', 19: 'right_eyebrow_inner_end', 20: 'left_eye_inner_corner', 21: 'left_eye_bottom_inner_midpoint', 22: 'left_eye_bottom_outer_midpoint', 23: 'left_eye_top_inner_midpoint', 24: 'left_eye_top_outer_midpoint', 25: 'left_eye_outer_corner', 26: 'left_eyebrow_outer_end', 27: 'left_eyebrow_outer_midpoint', 28: 'left_eyebrow_center', 29: 'left_eyebrow_inner_midpoint', 30: 'left_eyebrow_inner_end', 31: 'mouth_top_lip', 32: 'top_lip_top_right_center', 33: 'top_lip_top_right_midpoint', 34: 'mouth_right_corner', 35: 'top_lip_bottom_right_midpoint', 36: 'top_lip_bottom_right_center', 37: 'top_lip_bottom_center', 38: 'top_lip_top_left_center', 39: 'top_lip_top_left_midpoint', 40: 'mouth_left_corner', 41: 'top_lip_bottom_left_midpoint', 42: 'top_lip_bottom_left_center', 43: 'bottom_lip_bottom_left_midpoint', 44: 'bottom_lip_top_left_midpoint' , 45: 'bottom_lip_bottom_left_center', 46: 'bottom_lip_top_left_center', 47: 'bottom_lip_bottom_right_center', 48: 'bottom_lip_top_right_center', 49: 'bottom_lip_bottom_left_midpoint', 50: 'mouth_bottom_lip', 51: 'chin_center', 52: 'chin_right', 53: 'right_jaw_1', 54: 'right_jaw_0', 55: 'right_cheek_1', 56: 'right_cheek_0', 57: 'right_ear_bottom', 58: 'right_ear_center', 59: 'right_ear_top', 60: 'chin_left', 61: 'left_jaw_1', 62: 'left_jaw_0', 63: 'left_cheek_1', 64: 'left_cheek_0', 65: 'left_ear_bottom', 66: 'left_ear_center', 67: 'left_ear_top'}
        pts_idx_dict_39 = {0: 'left_nostril', 1: 'nostrils_center', 2: 'nose_tip', 3: 'nose_ridge_bottom', 4: 'nose_ridge_top', 5: 'nose_center_top', 6: 'left_eye_bottom_inner_midpoint', 7: 'left_eye_bottom_outer_midpoint', 8: 'left_eye_outer_corner', 9: 'left_eye_top_inner_midpoint', 10: 'left_eye_top_outer_midpoint', 11: 'left_eyebrow_inner_midpoint', 12: 'left_eyebrow_center', 13: 'left_eyebrow_outer_midpoint', 14: 'left_eyebrow_outer_end', 15: 'mouth_top_lip', 16: 'top_lip_top_left_center', 17: 'top_lip_top_left_midpoint', 18: 'mouth_left_corner', 19: 'bottom_lip_bottom_left_midpoint', 20: 'bottom_lip_bottom_left_center', 21: 'mouth_bottom_lip', 22: 'top_lip_bottom_left_midpoint', 23: 'top_lip_bottom_left_center', 24: 'top_lip_bottom_left_center', 25: 'bottom_lip_top_left_center', 26: 'bottom_lip_top_center', 27: 'chin_center_top', 28: 'chin_center', 29: 'chin_left', 30: 'left_jaw_2', 31: 'left_jaw_1', 32: 'left_jaw_0', 33: 'left_cheek_2', 34: 'left_cheek_1', 35: 'left_cheek_0', 36: 'left_ear_bottom', 37:'left_ear_center', 38: 'left_ear_top'}
        keypoint_dicts = []

        translation_dict = pts_idx_dict_68
        if len(xs) == 39:
            translation_dict = pts_idx_dict_39
        keypoint_dict = dict([ (translation_dict[pos], coord) for pos,coord in enumerate(zip(xs,ys)) ])
        keypoint_dicts.append(keypoint_dict)

        if debug:
            for keypt_i, keypt_x in enumerate(xs):
                keypt_y = ys[keypt_i]
                cv2.circle(img, (numpy.int(keypt_x), numpy.int(keypt_y)), 3, (0, 0, 255))

        # TODO: il ne devrait qu'il n'y avoir qu'un set de keypoints par video clip.
        for face_keypts in keypoint_dicts:
            d1 = face_keypts['nose_ridge_bottom']
            d2 = face_keypts['nose_ridge_top']
            d3 = face_keypts['nose_center_top']
            up_nose = d3
            bottom_nose = face_keypts['nostrils_center']
            nose_heigth = bottom_nose[1] - up_nose[1]
            bottomMouth_to_chin = face_keypts['chin_center'][1] - face_keypts['mouth_bottom_lip'][1]

            face_keypts = numpy.array(face_keypts.values(), dtype=numpy.int32)

            x_max, y_max = face_keypts.max(axis=0)
            x_min, y_min = face_keypts.min(axis=0)

            param1 = 1.1
            param2 = 1.1
            if param1 * nose_heigth < bottomMouth_to_chin:
                y_max = bottom_nose[1] +  param2 * nose_heigth

            y_min = y_min - nose_heigth
            heigth = y_max - y_min
            diff_heigth = 0.1 * heigth
            y_max += diff_heigth
            heigth = y_max - y_min
            width = heigth * 0.95
            x_min_before = numpy.copy(x_min)
            x_min = x_max - width
            if x_min_before > x_min:
                x_diff = x_min_before - x_min
                x_max += 0.6 * x_diff
                y_min -= 0.6 * x_diff
            y_val = numpy.array([y_min, y_max])
            y_min, y_max = numpy.clip(y_val, 0, img.shape[0])
            x_val = numpy.array([x_min, x_max])
            x_min, x_max = numpy.clip(x_val, 0, img.shape[1])

            if bbox_on_img_path is not None:
                cv2.rectangle(img, (numpy.int(x_min), numpy.int(y_min)), (numpy.int(x_max), numpy.int(y_max)), (0, 255, 0), 2)
                if not os.path.isdir(bbox_on_img_path):
                    os.makedirs(bbox_on_img_path)
                cv2.imwrite(os.path.join(bbox_on_img_path, '%s.png'%name), img)
            if not os.path.isdir(bbox_path):
                os.makedirs(bbox_path)
            f = open(os.path.join(bbox_path, '%s.txt'%name), 'a')
            f.write('%s, %s, %s, %s\n'%(x_min, y_min, x_max, y_max))
            f.close()


if __name__ == '__main__':
    # python get_bbox_optB.py arg1 arg2 arg3 arg4 [arg5]
    # python get_bbox_optB.py /u/ebrahims/emotiw_pipeline/validation_set/Valid_Vid_Distr/Data/000147200.avi
    #                    /u/ebrahims/emotiw_pipeline/validation_set/ramanan_keypoints_picassa_backup/000147200
    #                    /data/lisa/data/faces/EmotiW/Points/Happy/000147200.mat
    #                    /data/lisa/exp/chandiar/Challenge/DATA/bbox_coords/000147200
    #                    /data/lisa/exp/chandiar/Challenge/DATA/bbox_on_data/000147200

    #   arg1 is the path to the folder containing the audio-video file for the given clip (.avi)
    #   arg2 is the path to the folder containing the extracted frames for the given video clip (.png)
    #   arg3 is the path to the ramanan mat file for the given video clip (.mat)
    #   arg4 is the path to the folder where we will save the bounding boxes coordinates (.txt)
    #   arg5 is the path to the folder where we will save the images with the bounding boxes drawn (.png), this is an optional argument
    if debug:
        #missing_clips = ['002350040', '003044960', '005242000', '010924040', '013736360', '014429160', '003258400']
        missing_clips = [   'Happy/000147200',
                            'Surprise/000256440',
                            'Fear/000606007',
                            'Neutral/000831400',
                            'Surprise/002916678',
                            'Sad/003024767',
                            'Neutral/004008240',
                            'Sad/005700400',
                            'Sad/005711000',
                            'Angry/011018760',
                            'Sad/011616534']
        for clip in missing_clips:
            clip_id = clip.split('/')[-1]
            avi_path = '/u/ebrahims/emotiw_pipeline/validation_set/Valid_Vid_Distr/Data/%s.avi'%clip_id
            extracted_frames_path = '/u/ebrahims/emotiw_pipeline/validation_set/ramanan_keypoints_picassa_backup/%s'%clip_id
            ramanan_keypts_path = '/data/lisa/data/faces/EmotiW/Points/%s.mat'%clip
            bbox_path = '/data/lisa/exp/chandiar/Challenge/DATA/bbox_coords/%s'%clip_id
            if True:
                bbox_on_img_path = '/data/lisa/exp/chandiar/Challenge/DATA/bbox_on_data/%s'%clip_id
            else:
                bbox_on_img_path = None
            get_bbox(   avi_path,
                        extracted_frames_path,
                        ramanan_keypts_path,
                        bbox_path,
                        bbox_on_img_path)
    else:
        avi_path = sys.argv[1]
        extracted_frames_path = sys.argv[2]
        ramanan_keypts_path = sys.argv[3]
        bbox_path = sys.argv[4]
        if len(sys.argv) == 6:
            bbox_on_img_path = sys.argv[5]
        else:
            bbox_on_img_path = None
        get_bbox(   avi_path,
                    extracted_frames_path,
                    ramanan_keypts_path,
                    bbox_path,
                    bbox_on_img_path)
