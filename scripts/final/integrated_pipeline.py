#!/usr/bin/env python

import os
import shutil
import subprocess
import sys
import time

from mlabwrap import mlab


###
### DEBUG THING
extract_frames = 0
run_picasa = 0
extract_bbox = 0
smooth_facetubes = 0

run_audio = 0

run_kishore = 0
run_bomf = 0


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

# Initial directory containing faces aligned by the organizers
ALIGNED_DIR = 'Faces_Aligned_Test'
ALIGNED_DIR = os.path.join(DATA_ROOT_DIR, ALIGNED_DIR)

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

if extract_frames:
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
if run_picasa:
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
if extract_bbox:
    find_match_script = os.path.join(SCRIPTS_PATH, 'mirzamom', 'find_match.py')
    cmd_line_template = '%(python)s %(script)s %(orig_path)s %(cropped_path)s %(save_path)s'
    for clip_id in CLIP_IDS:
        save_path = os.path.join(bbox_dir, clip_id)
        if os.path.exists(save_path):
            raise Exception(
                "Directory for extracting bbox coordinates for clip id %s "
                "already exists: %s" % (clip_id, save_path))
        else:
            os.mkdir(save_path)
    
        cmd_line = cmd_line_template % dict(
            python=sys.executable,
            script=find_match_script,
            orig_path=os.path.join(frame_dir, clip_id),
            cropped_path=os.path.join(faces_dir, clip_id),
            save_path=save_path)
        subprocess.check_call(cmd_line, shell=True)


### Phase 2.1b: Fallback if Picasa did not find anything

# Use Ramanan keypoints algorithm to find and save keypoint detections.
# How to deal with multiple detections?
backup_faces_dir = os.path.join(DATA_ROOT_DIR, 'ramanan_keypoints_picassa_backup')
if not os.path.exists(backup_faces_dir):
    os.mkdir(backup_faces_dir)

for clip_id in CLIP_IDS:
    clip_faces_dir = os.path.join(faces_dir, clip_id)
    nb_faces = len([f for f in os.listdir(clip_faces_dir) if f.endswith('.jpg')])
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

### Phase 2.2b: bbox coordinates if Picasa did not find anything
# TODO: Raul, finish

## Phase 2.3:
facetubes_dir = os.path.join(DATA_ROOT_DIR, 'facetubes_96x96')
if not os.path.exists(facetubes_dir):
    os.mkdir(facetubes_dir)

if smooth_facetubes:
    cmd_template = "%(python)s %(script)s %(orig_path)s %(bbox_path)s %(save_path)s"
    for clip_id in CLIP_IDS:
        save_path = os.path.join(facetubes_dir, clip_id)
        if os.path.exists(save_path):
            raise Exception(
                "Directory for exporting smooth facetube images "
                "for clip id %s already exists: %s" % (clip_id, save_path))
        else:
            os.mkdir(save_path)

        cmd_line = cmd_template % dict(
            python=sys.executable,
            script=os.path.join(SCRIPTS_PATH, 'chandiar', 'smoothed_face_tube.py'),
            orig_path=os.path.join(frame_dir, clip_id),
            bbox_path=os.path.join(bbox_dir, clip_id),
            save_path=save_path)
        print 'executing:'
        print cmd_line
        subprocess.check_call(cmd_line, shell=True)        


###
### Kishore's module (activity recognition)
if run_kishore:
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

###
### Sebastien Jean's module (bag of mouth features)
if run_bomf:
    #ex_cmd_line = 'python ../jeasebas/BoMF_cmdline.py /u/ebrahims/emotiw_pipeline/test1/Faces_Aligned_Test /u/ebrahims/emotiw_pipeline/test1/Faces_Aligned_Test_Small /data/lisa/exp/faces/emotiw_final/jeasebas /u/ebrahims/emotiw_pipeline/test1/bomf_pred 50 000143240'
    small_faces_outdir = ALIGNED_DIR + '_Small'
    if not os.path.exists(small_faces_outdir):
        os.mkdir(small_faces_outdir)
    bomf_model_dir = '/data/lisa/exp/faces/emotiw_final/jeasebas'
    
    cmd_line_template = '%(python)s %(bomf_cmdline)s %(aligned_faces_dir)s %(small_faces_outdir)s %(model_dir)s %(pred_dir)s %(batch_size)i %(clip_ids)s'

    for clip_id in CLIP_IDS:
        cmd_line = cmd_line_template % dict(
            python=sys.executable,
            bomf_cmdline=os.path.join(SCRIPTS_PATH, 'jeasebas', 'BoMF_cmdline.py'),
            aligned_faces_dir=ALIGNED_DIR,
            small_faces_outdir=small_faces_outdir,
            model_dir=bomf_model_dir,
            pred_dir=PREDICTION_DIR,
            batch_size=50,
            clip_ids=clip_id)
        subprocess.check_call(cmd_line, shell=True)
        shutil.move(os.path.join(PREDICTION_DIR, 'BoMF_test_probabilities.npy'),
                    os.path.join(PREDICTION_DIR, 'bomf_pred_%s.npy' % clip_id))
