#!/usr/bin/env python

import os
import shutil
import subprocess
import sys
import time

from mlabwrap import mlab

### Define environment variables for configuration

# For libsvm
os.environ['PYTHONPATH'] = (
    '/data/lisa/exp/faces/emotiw_final/Kishore/inference_line/'
    'libsvm-3.13/python:') + os.environ['PYTHONPATH']


# Root directory for the data read and generated
DATA_ROOT_DIR = '/u/ebrahims/emotiw_pipeline/test1'

# Initial directory containing the *.avi files,
# relative to DATA_ROOT_DIR
AVI_DIR = 'Test_Vid_Distr/Data'
AVI_DIR = os.path.join(DATA_ROOT_DIR, AVI_DIR)

# Data where we put the individual predictions
PREDICTION_DIR = 'module_predictions'
PREDICTION_DIR = os.path.join(DATA_ROOT_DIR, PREDICTION_DIR)
if not os.path.exists(PREDICTION_DIR):
    os.mkdir(PREDICTION_DIR)

# Names of clips to process
CLIP_IDS = [
    '000143240',
#    '000152960',
#    '000157760',
#    '000201320',
#    '000231280',
#    '000236840',
#    '000240440',
#    '000247920',
#    '000257240',
#    '000311160',
    ]

# TODO: Jean-Philippe, which directories to use?
# Picasa incoming directory
PICASA_PROCESSING_DIR = '/data/lisatmp/faces/picasa_process'

## Export these variables as environment variables
# TODO: Pascal L., see if actually needed
#os.environ['DATA_ROOT_DIR'] = DATA_ROOT_DIR
#os.environ['AVI_DIR'] = AVI_DIR
#os.environ['PICASA_INCOMING_DIR'] = PICASA_INCOMING_DIR
#os.environ['PICASA_FACES_DIR'] = PICASA_FACES_DIR


## Path of current script and "scripts" directory
SELF_PATH = __file__
print 'SELF_PATH:', SELF_PATH

SCRIPTS_PATH = os.path.abspath(os.path.join(
    os.path.dirname(SELF_PATH), os.pardir))


# Names of clip

### Phase 1:  Extract frames from clips as individual files
script_name = os.path.join(SCRIPTS_PATH, 'mirzamom', 'test_frame_extractor.py')
frame_dir = os.path.join(DATA_ROOT_DIR, 'extracted_frames')
if not os.path.exists(frame_dir):
    os.mkdir(frame_dir)

for clip_id in CLIP_IDS:
    clip_dir = AVI_DIR
    clip_frame_dir = os.path.join(frame_dir, clip_id)
    if os.path.exists(clip_frame_dir):
        raise Exception(
            "Directory for extracting frames of clip id %s already exists: "
            "%s" % (clip_id, clip_frame_dir))
    os.mkdir(clip_frame_dir)

    # sys.executable is the full path to python
    cmd_line = '%s %s %s %s %s' % (
        sys.executable, script_name, clip_dir, clip_frame_dir,
        '%s.avi' % clip_id)

    subprocess.check_call(cmd_line, shell=True)

### Phase 2: Extract Picasa faces
faces_dir = os.path.join(DATA_ROOT_DIR, 'picasa_faces')
if not os.path.exists(faces_dir):
    os.mkdir(faces_dir)
bbox_dir = os.path.join(DATA_ROOT_DIR, 'picasa_bbox')
if not os.path.exists(bbox_dir):
    os.mkdir(bbox_dir)

## Phase 2.1: put the files at the right place, and wait for Picasa to
#  complete
for clip_id in CLIP_IDS:
    # Copy the directory containing frames to Picasa's incoming directory
    clip_frame_dir = os.path.join(frame_dir, clip_id)
    clip_picasa_incoming = os.path.join(PICASA_PROCESSING_DIR, clip_id)
    shutil.copytree(clip_frame_dir, clip_picasa_incoming)

    # Then, append '.process_me' to its name, to signal the Windows script
    # that it can proceed
    os.rename(clip_picasa_incoming, '%s.process_me' % clip_picasa_incoming)

    # Wait for Picasa to return
    # This one contains the full frames, after processing
    clip_picasa_processed_dir = os.path.join(
        PICASA_PROCESSING_DIR, '%s.processed' % clip_id)
    # This one contains the extracted faces
    clip_picasa_faces_dir = os.path.join(
        PICASA_PROCESSING_DIR, '%s.faces' % clip_id)

    for i in xrange(300):
        time.sleep(1)
        if os.path.exists(clip_picasa_processed_dir):
            break

    else:
        # This is executed if the "break" was never executed
        raise Exception("Picasa script timed out")

    assert os.path.exists(clip_picasa_faces_dir)
    clip_faces_dir = os.path.join(faces_dir, clip_id)
    shutil.copytree(clip_picasa_faces_dir, clip_faces_dir)


## Phase 2.2: get bounding boxes

## Phase 2.3:


### Phase 2b: Fallback if Picasa did not find anything

# Use Ramanan keypoints algorithm to find and save keypoint detections.
# How to deal with multiple detections?
backup_faces_dir = os.path.join(DATA_ROOT_DIR, 'ramanan_keypoints_picassa_backup')
if not os.path.exists(backup_faces_dir):
    os.mkdir(backup_faces_dir)

for clip_id in CLIP_IDS:
    clip_faces_dir = os.path.join(faces_dir, clip_id)
    nb_faces = len([f for f in os.listdir(clip_faces_dir) if f.endswith('.png')])
    if nb_faces > 0:
        # Picasa detected faces, no need for the backup plan
        continue

    clip_frame_dir = os.path.join(frame_dir, clip_id)
    clips = []
    for i, j, c in os.walk(clip_frame_dir):
        clips = c
    for clip in clips:
        print '\n', clip, clip_frame_dir, backup_faces_dir
        dest = os.path.join(backup_faces_dir, clip[:-4] + '__ramanan.mat')
        clip = os.path.join(clip_frame_dir, clip)
        print clip, dest
        # model can be 0 (used for challenge), 1, or 2.
        # Lower numbered models are better and slower.
        mlab.ramanan1(clip, dest, 0)




###
### Kishore's module (activity recognition)

kishore_model_root = '/data/lisa/exp/faces/emotiw_final/Kishore/inference_line'

# Convert from mpeg2 to mjpeg, otherwise opencv cannot read the video
mjpeg_dir = os.path.join(DATA_ROOT_DIR, 'mjpeg_avi')
if not os.path.exists(mjpeg_dir):
    os.mkdir(mjpeg_dir)

convert_line_template = 'mencoder %(inp)s -ovc lavc -lavcopts vcodec=mjpeg -oac copy -o %(out)s'
cmd_line_template = '%(python)s %(inference_line)s %(centroids_file)s %(model_file)s %(videos_path)s %(train_data_file)s %(clip_ids)s'

for clip_id in CLIP_IDS:
    convert_line = convert_line_template % dict(
        inp=os.path.join(AVI_DIR, '%s.avi' % clip_id),
        out=os.path.join(mjpeg_dir, '%s.avi' % clip_id))
    subprocess.check_call(convert_line, shell=True)

    cmd_line = cmd_line_template % dict(
        python=sys.executable,
        inference_line=os.path.join(SCRIPTS_PATH, 'kishore', 'inference_line', 'inference_line.py'),
        centroids_file=os.path.join(kishore_model_root, 'kmeans_centroids.npy'),
        model_file=os.path.join(kishore_model_root, 'model_params.npz'),
        videos_path=mjpeg_dir,
        train_data_file=os.path.join(kishore_model_root, 'chal_train_data.npz'),
        clip_ids=clip_id)

    subprocess.check_call(cmd_line, shell=True)

    # The output will be a one-liner file in the current directory
    # TODO: check if it makes a difference to run all the clips at once,
    #       Kishore did that, and the results could be different
    shutil.move(os.path.join('activity_recognition_test_results.txt'),
                os.path.join(PREDICTION_DIR, 'kishore_pred_%s.txt' % clip_id))
