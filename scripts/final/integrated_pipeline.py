#!/usr/bin/env python

import os
import shutil
import subprocess
import sys
import time


### Define environment variables for configuration

# Root directory for the data read and generated
DATA_ROOT_DIR = '/u/ebrahims/emotiw_pipeline/test1'

# Initial directory containing the *.avi files,
# relative to DATA_ROOT_DIR
AVI_DIR = 'Test_Vid_Distr/Data'

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
