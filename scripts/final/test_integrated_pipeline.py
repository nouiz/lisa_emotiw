#!/usr/bin/env python

import os
import shutil
import subprocess
import sys
import time

from mlabwrap import mlab

### Define environment variables for configuration

# Root directory for the data read and generated
DATA_ROOT_DIR = '/u/lamblinp/emotiw_pipeline/test1'

# Initial directory containing the *.avi files,
# relative to DATA_ROOT_DIR
AVI_DIR = 'Test_Vid_Distr/Data'

# Names of clips to process
CLIP_IDS = [
    '000143240',
    ]

# TODO: Jean-Philippe, which directories to use?
# Picasa incoming directory
PICASA_PROCESSING_DIR = '???'

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
    clip_dir = os.path.join(DATA_ROOT_DIR, AVI_DIR)
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

## Phase 2.2: get bounding boxes

## Phase 2.3:


### Phase 2b: Fallback if Picasa did not find anything

# Use Ramanan keypoints algorithm... finds and saves detections. 
# Needs to check it PICASSA failed!
# How to deal with multiple detections? 
backup_faces_dir = os.path.join(DATA_ROOT_DIR, 'ramanan_keypoints_picassa_backup')
if not os.path.exists(backup_faces_dir):
    os.mkdir(backup_faces_dir)

for clip_id in CLIP_IDS:
    clip_frame_dir = os.path.join(frame_dir, clip_id)
    ramanan1(clip_frame_dir, backup_faces_dir, model = 0)
    








