import cPickle
import os
import sys

from util import *
from conf import config


def main():
    import pdb; pdb.set_trace()
    to_be_saved = []
    missing_bbx = []
    clip_id = config['extracted_frames_path'].split('/')[-1]
    if True:
        print 'clip id: ', clip_id
        frames = get_frames("{}/".format(config['extracted_frames_path']), [clip_id])
        bounding_boxes = get_bounding_boxes("{}/".format(config['picasa_bbox_path']), [clip_id])
        failed = []
        face_tubes = get_face_tubes(frames, bounding_boxes, failed, 
                                    "{}/".format(config['extracted_frames_path']), 
                                    "{}/".format(config['save_path']), 
                                    config['distance_thr'], 
                                    config['size_thre'], 
                                    config['overlap_thr'], 
                                    config['similar_thr'])

        if config['what_to_save'] in ['img', 'mat']:
            save_tube(face_tubes, 
                      "{}/".format(config['extracted_frames_path']), 
                      "{}/".format(config['save_path']), 
                      config['size'],
                      config['what_to_save'])
        else:
            raise NotImplementedError('Option %s not supported'%config['what_to_save'])


if __name__ == "__main__":
    main()
