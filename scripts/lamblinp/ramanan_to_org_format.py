"""
For testing, take our Ramanan keypoints and put them in the same format as the organizers.

This is the format we expect them to extract Ramanan keypoints for frames with correct aspect ratio.
"""
import glob
import os
import sys

import numpy
import scipy.io


def poly_to_org_format(input_base_dir, extracted_frames_base_dir, output_dir, clip_id):
    input_dir = os.path.join(input_base_dir, clip_id)
    frames_dir = os.path.join(extracted_frames_base_dir, clip_id)

    # Get the number of frames from extracted_frames/clip_id/
    nb_in_frames = len(glob.glob(os.path.join(frames_dir, '*.png')))
    in_files = [os.path.join(input_dir, '%s-%03d.mat' % (clip_id, i + 1))
                for i in range(nb_in_frames)]

    out_points = numpy.zeros((nb_in_frames, 1), dtype=[('s', '|O8'), ('c', '|O8'), ('xy', '|O8'), ('level', '|O8')])
    for i, f in enumerate(in_files):
        if os.path.isfile(f):
            pts = scipy.io.loadmat(f)['bs'][0][0]
            out_points[i] = pts
        else:
            for k in ('s', 'c', 'xy', 'level'):
                out_points[i][0][k] = numpy.zeros((1, 1), dtype='uint8')

    out_file = os.path.join(output_dir, '%s.mat' % clip_id)
    scipy.io.savemat(out_file, dict(points=out_points))


if __name__ == '__main__':
    input_base_dir = sys.argv[1]
    extracted_frames_base_dir = sys.argv[2]
    output_dir = sys.argv[3]
    clip_ids = sys.argv[4:]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for clip_id in clip_ids:
        poly_to_org_format(input_base_dir, extracted_frames_base_dir, output_dir, clip_id)
