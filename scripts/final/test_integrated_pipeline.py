#!/usr/bin/env python

import os
import shutil
import subprocess
import sys
import time

from mlabwrap import mlab

### Define environment variables for configuration

# Root directory for the data read and generated
DATA_ROOT_DIR = '/u/kruegerd/Desktop/test1'

# Initial directory containing the *.avi files,
# relative to DATA_ROOT_DIR
AVI_DIR = '/u/kruegerd/Desktop/test1/Test_Vid_Distr/Data'

# Names of clips to process


CLIP_IDS = [
    '000143240',
    ]

print CLIP_IDS

pCLIP_IDS = ['002350040', '003044960', '003258400', '005242000',
                    '010924040', '013736360', '014429160']

CLIP_IDS = [pCLIP_IDS[0]]

print CLIP_IDS


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
    clips = []
    for i,j,c in os.walk(clip_frame_dir):
        clips = c
    for clip in clips:
        print '\n', clip, clip_frame_dir, backup_faces_dir, '\n'
        dest = backup_faces_dir + '/' + clip[:-4] + '__ramanan.mat'
        clip = clip_frame_dir + '/' + clip
        print clip, dest, '\n'
        #model can be 0 (used for challenge), 1, or 2.  
        #Lower numbered models are better and slower.
        mlab.ramanan1(clip, dest, 0)
    








