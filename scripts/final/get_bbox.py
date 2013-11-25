import cv, cv2
import glob
import numpy
import os
from scipy import io as sio
import sys
import warnings


def get_bbox(extracted_frames_path, ramanan_keypts_path, bbox_path, bbox_on_img_path = None):
    #no_keypoints = []
    #info = []
    #keypts ={}

    # TODO: pas de '/' a la fin des arguments (chemins complets).
    if extracted_frames_path[-1] == '/':
	extracted_frames_path = extracted_frames_path[:-1] 
    clip = extracted_frames_path.split('/')[-1]
    print 'clip id: ', clip
    frames_paths = glob.glob("{}/*png".format(extracted_frames_path))
    #ramanan_paths = glob.glob("{}/*mat".format(ramanan_keypts_path))
    #frames_paths.sort()
    #ramanan_paths.sort()
    # TODO: peut-on supposer qu'on a le meme nombre de mat et png? une seule face, la preponderante?
    # NON! voir 003258400
    # TODO: est-ce que c'est la meme convention de noms de fichiers pour mat et png?
    #assert len(frames_paths) == len(ramanan_paths)
    #keypts[clip] = []
    for i, frame_path in enumerate(frames_paths):
	basename, ext = os.path.splitext(frame_path)
	name = basename.split('/')[-1]
	img = cv2.imread(frame_path)
	# TODO: c'est quoi la convention de nommer les fichiers mat?
	ramanan_path = ramanan_keypts_path + name + '__ramanan.mat'#+ '_ramanan.mat'
	if not os.path.exists(ramanan_path):
	    # TODO: si pas de ramanan donc pas de bounding box pour la frame.
	    continue
	matfile = sio.loadmat(ramanan_path)

	xs_and_ys = []
	xoffset = 0
	yoffset = 0
	# Look for offsets to apply to coordinates
	# TODO: pas d'offset puisqu'on n'a pas de bounding box coords.
	'''
	basename,ext = os.path.splitext(ramanan_paths[i])
	bbox_filepath = basename+"_bbox.txt"
	if os.path.exists(bbox_filepath):
	    with open(bbox_filepath, 'r') as infile:
		x1,y1,x2,y2 = infile.readline().split()
		xoffset = float(x1)
		yoffset = float(y1)            
	'''
			    
	first_xs = matfile['xs'][0]
	first_ys = matfile['ys'][0]
	xs_and_ys.append((first_xs+xoffset, first_ys+yoffset))

	# TODO: est-ce que ca nous concerne? on a tu des fichiers mat qui terminent par 'ramanan_face*'
	if ramanan_path[-18:-6]!='ramanan_face':
	    bs = matfile['bs']
	    bs_n, bs_m = bs.shape
	    for i in range(bs_m):
		bs_xy = bs[0,i]['xy']
		# print "Ramanan bs_xy",bs_xy
		xs = 0.5*(bs_xy[:,0]+bs_xy[:,2])
		ys = 0.5*(bs_xy[:,1]+bs_xy[:,3])

		if len(xs)!=len(first_xs) or (numpy.abs(first_xs-xs)).max()>0.1: # don't append if it's same as first
		    xs_and_ys.append((xs+xoffset, ys+yoffset))
		    
	pts_idx_dict_68 = {0: 'nostrils_center', 1: 'right_nostril_inner_end', 2: 'right_nostril', 3: 'left_nostril_inner_end', 4: 'left_nostril', 5: 'nose_tip', 6: 'nose_ridge_bottom', 7: 'nose_ridge_top', 8: 'nose_center_top', 9: 'right_eye_inner_corner', 10: 'right_eye_bottom_inner_midpoint', 11: 'right_eye_bottom_outer_midpoint', 12: 'right_eye_top_inner_midpoint', 13: 'right_eye_top_outer_midpoint', 14: 'right_eye_outer_corner', 15: 'right_eyebrow_outer_end', 16: 'right_eyebrow_outer_midpoint', 17: 'right_eyebrow_center', 18: 'right_eyebrow_inner_midpoint', 19: 'right_eyebrow_inner_end', 20: 'left_eye_inner_corner', 21: 'left_eye_bottom_inner_midpoint', 22: 'left_eye_bottom_outer_midpoint', 23: 'left_eye_top_inner_midpoint', 24: 'left_eye_top_outer_midpoint', 25: 'left_eye_outer_corner', 26: 'left_eyebrow_outer_end', 27: 'left_eyebrow_outer_midpoint', 28: 'left_eyebrow_center', 29: 'left_eyebrow_inner_midpoint', 30: 'left_eyebrow_inner_end', 31: 'mouth_top_lip', 32: 'top_lip_top_right_center', 33: 'top_lip_top_right_midpoint', 34: 'mouth_right_corner', 35: 'top_lip_bottom_right_midpoint', 36: 'top_lip_bottom_right_center', 37: 'top_lip_bottom_center', 38: 'top_lip_top_left_center', 39: 'top_lip_top_left_midpoint', 40: 'mouth_left_corner', 41: 'top_lip_bottom_left_midpoint', 42: 'top_lip_bottom_left_center', 43: 'bottom_lip_bottom_left_midpoint', 44: 'bottom_lip_top_left_midpoint' , 45: 'bottom_lip_bottom_left_center', 46: 'bottom_lip_top_left_center', 47: 'bottom_lip_bottom_right_center', 48: 'bottom_lip_top_right_center', 49: 'bottom_lip_bottom_left_midpoint', 50: 'mouth_bottom_lip', 51: 'chin_center', 52: 'chin_right', 53: 'right_jaw_1', 54: 'right_jaw_0', 55: 'right_cheek_1', 56: 'right_cheek_0', 57: 'right_ear_bottom', 58: 'right_ear_center', 59: 'right_ear_top', 60: 'chin_left', 61: 'left_jaw_1', 62: 'left_jaw_0', 63: 'left_cheek_1', 64: 'left_cheek_0', 65: 'left_ear_bottom', 66: 'left_ear_center', 67: 'left_ear_top'}
	pts_idx_dict_39 = {0: 'left_nostril', 1: 'nostrils_center', 2: 'nose_tip', 3: 'nose_ridge_bottom', 4: 'nose_ridge_top', 5: 'nose_center_top', 6: 'left_eye_bottom_inner_midpoint', 7: 'left_eye_bottom_outer_midpoint', 8: 'left_eye_outer_corner', 9: 'left_eye_top_inner_midpoint', 10: 'left_eye_top_outer_midpoint', 11: 'left_eyebrow_inner_midpoint', 12: 'left_eyebrow_center', 13: 'left_eyebrow_outer_midpoint', 14: 'left_eyebrow_outer_end', 15: 'mouth_top_lip', 16: 'top_lip_top_left_center', 17: 'top_lip_top_left_midpoint', 18: 'mouth_left_corner', 19: 'bottom_lip_bottom_left_midpoint', 20: 'bottom_lip_bottom_left_center', 21: 'mouth_bottom_lip', 22: 'top_lip_bottom_left_midpoint', 23: 'top_lip_bottom_left_center', 24: 'top_lip_bottom_left_center', 25: 'bottom_lip_top_left_center', 26: 'bottom_lip_top_center', 27: 'chin_center_top', 28: 'chin_center', 29: 'chin_left', 30: 'left_jaw_2', 31: 'left_jaw_1', 32: 'left_jaw_0', 33: 'left_cheek_2', 34: 'left_cheek_1', 35: 'left_cheek_0', 36: 'left_ear_bottom', 37:'left_ear_center', 38: 'left_ear_top'}
	keypoint_dicts = []

	for xs,ys in xs_and_ys:
	    translation_dict = pts_idx_dict_68
	    if len(xs) == 39:
		translation_dict = pts_idx_dict_39
	    keypoint_dict = dict([ (translation_dict[pos], coord) for pos,coord in enumerate(zip(xs,ys)) ]) 
	    keypoint_dicts.append(keypoint_dict)

	# TODO: il ne devrait qu'il n'y avoir qu'un set de keypoints par video clip.
	for face_keypts in keypoint_dicts:
	    #keypts[clip].append((frame_path, face_keypts))
	    #if len(all_keypoints) > 1:
	    #    import pdb; pdb.set_trace()
	    d1 = face_keypts['nose_ridge_bottom']
	    d2 = face_keypts['nose_ridge_top']
	    d3 = face_keypts['nose_center_top']
	    up_nose = d3
	    bottom_nose = face_keypts['nostrils_center']
	    nose_heigth = bottom_nose[1] - up_nose[1]
	    bottomMouth_to_chin = face_keypts['chin_center'][1] - face_keypts['mouth_bottom_lip'][1]

	    #assert numpy.argmin([d1[1], d2[1], d3[1]]) == 2

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

	    #info.append((x_max - x_min, y_max - y_min))
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
    #'013547320'
    #python get_bbox.py /home/raul/Desktop/ExtractedFrame/003044960/ /home/raul/Desktop/ExtractedFrame/003044960/ /home/raul/Desktop/BoundBoxData/003044960/ /home/raul/Desktop/bbox_on_image/003044960
    debug = False
    if debug:
	missing_clips = ['002350040', '003044960', '005242000', '010924040', '013736360', '014429160', '003258400']
	for clip in missing_clips:
	    extracted_frames_path = '/home/raul/Desktop/ExtractedFrame/%s/'%clip#sys.argv[1]
	    ramanan_keypts_path = '/home/raul/Desktop/ExtractedFrame/%s/'%clip#sys.argv[2]
	    bbox_path = '/home/raul/Desktop/BoundBoxData/%s'%clip#sys.argv[3]	
	    if True:
		bbox_on_img_path = '/home/raul/Desktop/bbox_on_image/%s'%clip#sys.argv[4]
	    else:
		bbox_on_img_path = None
	    get_bbox( extracted_frames_path,
		      ramanan_keypts_path,
		      bbox_path,
		      bbox_on_img_path)
    else:
	extracted_frames_path = sys.argv[1]
	ramanan_keypts_path = sys.argv[2]
	bbox_path = sys.argv[3]	
	if len(sys.argv) == 5:
	    bbox_on_img_path = sys.argv[4]
	else:
	    bbox_on_img_path = None
	get_bbox( extracted_frames_path,
		  ramanan_keypts_path,
		  bbox_path,
		  bbox_on_img_path)
