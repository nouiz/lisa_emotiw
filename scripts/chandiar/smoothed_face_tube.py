import cPickle
import os
import sys

from util import *
from conf import config


def main():
    extracted_frames_path = sys.argv[1]
    picasa_bbox_path = sys.argv[2]
    save_path = sys.argv[3]

    to_be_saved = []
    missing_bbx = []
    clip_id = extracted_frames_path.split('/')[-1]
    if True:
        print 'clip id: ', clip_id
        frames = get_frames("{}/".format(extracted_frames_path), [clip_id])
        bounding_boxes = get_bounding_boxes("{}/".format(picasa_bbox_path), [clip_id])
        failed = []
        face_tubes = get_face_tubes(frames, bounding_boxes, failed, 
                                    "{}/".format(extracted_frames_path), 
                                    "{}/".format(save_path), 
                                    config['distance_thr'], 
                                    config['size_thre'], 
                                    config['overlap_thr'], 
                                    config['similar_thr'])

        if config['what_to_save'] in ['img', 'mat']:
            save_tube(face_tubes, 
                      "{}/".format(extracted_frames_path), 
                      "{}/".format(save_path), 
                      config['size'],
                      config['what_to_save'])
        else:
            raise NotImplementedError('Option %s not supported'%config['what_to_save'])


if __name__ == "__main__":
    main()
