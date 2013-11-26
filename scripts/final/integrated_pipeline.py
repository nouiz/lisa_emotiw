#!/usr/bin/env python

import os
import sys
import shutil
import subprocess
import logging
import time

from get_bbox import get_bbox

#LOG_LEVEL = logging.DEBUG
LOG_LEVEL = logging.INFO

logging.basicConfig(format="%(asctime)s: %(message)s", level=LOG_LEVEL)

### PIPELINE STAGES
extract_frames = 0
run_picasa = 0
extract_bbox = 0
smooth_facetubes = 0
run_svm_convnet = 0

alt_path1 = 1       # run Ramanan on full frames if picasa did not find anything
alt_path2 = 1       # postprocess alt_path1 output

run_audio = 0
run_svm_convnet_audio = 0

run_kishore = 0
run_bomf = 0

run_xavier = 0

### Configureation ###

REMOTE_RAMANAN = 1                                    # use remote ramanan computation
REMOTE_USER_HOST='bornj@cudahead.rdgi.polymtl.ca'     # cluster account to use
REMOTE_DATA_PATH='tmp/jobdata/'                       # directory there the workpackages will be copied
REMOTE_NO_WORKER='24'                                 # number of parallel workers on the cluster
REMOTE_POLLING_TIMEOUT=10                             # polling interval in seconds
REMOTE_SUBMIT_SCIPT='lisa_emotiw/scripts/jorg/remote/submit-worker.py' 

# Picasa incoming directory
PICASA_PROCESSING_DIR = '/data/lisatmp/faces/picasa_process'

# More imports

sys.path.append('../')
sys.path.append('libsvm-3.13/python')
sys.path.append('/data/lisa/exp/faces/emotiw_final/Kishore/inference_line/')

import jorg.remote as remote
remote.REMOTE_USER_HOST = REMOTE_USER_HOST

# Root directory for the data read and generated
#DATA_ROOT_DIR = '/u/ebrahims/emotiw_pipeline/test1'
DATA_ROOT_DIR = '/u/bornj/emotiw_pipeline/test1'

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
#    '000143240',
#    '000152960',
#    '000157760',
#    '000201320',
#    '000231280',
#    '000236840',
#    '000240440',
#    '000247920',
#    '000257240',
#    '000311160',
#    '002350040',   ## tested for convnet
     'our-little-one'
    ]


## Path of current script and "scripts" directory
SELF_PATH = __file__
logging.debug('SELF_PATH:'+SELF_PATH)

SCRIPTS_PATH = os.path.abspath(os.path.join(
    os.path.dirname(SELF_PATH), os.pardir))


### Phase 1:  Extract frames from clips as individual files
script_name = os.path.join(SCRIPTS_PATH, 'mirzamom', 'test_frame_extractor.py')
frame_dir = os.path.join(DATA_ROOT_DIR, 'extracted_frames')

if extract_frames:
    logging.info("Phase 1 -- Extract frames from clips")
    if not os.path.exists(frame_dir):
        os.mkdir(frame_dir)

    for clip_id in CLIP_IDS:
        clip_dir = AVI_DIR
        clip_frame_dir = os.path.join(frame_dir, clip_id)
        if os.path.exists(clip_frame_dir):
            #raise Exception(
            #    "Directory for extracting frames of clip id %s already exists: "
            #    "%s" % (clip_id, clip_frame_dir))
            pass
        else:
            os.mkdir(clip_frame_dir)

        # sys.executable is the full path to python
        cmd_line = '%s %s %s %s %s' % (
            sys.executable, script_name, clip_dir, clip_frame_dir,
            '%s.avi' % clip_id)

        print 'executing cmd:'
        print cmd_line
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
    logging.info("Phase 2.1 -- Run Picasa on clips")
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
        if os.path.exists(clip_faces_dir):
            # It may or may not be correct, remove it
            shutil.rmtree(clip_faces_dir)
        shutil.copytree(clip_picasa_faces_dir, clip_faces_dir)

        # Clean up, so the Windows script does not crash if the same
        # clip gets extracted again
        shutil.rmtree(clip_picasa_processed_dir)
        shutil.rmtree(clip_picasa_faces_dir)


## Phase 2.2: get bounding boxes
if extract_bbox:
    logging.info("Phase 2.2 -- Compute bounding boxes from Picasa output")
    find_match_script = os.path.join(SCRIPTS_PATH, 'mirzamom', 'find_match.py')
    cmd_line_template = '%(python)s %(script)s %(orig_path)s %(cropped_path)s %(save_path)s'
    for clip_id in CLIP_IDS:
        save_path = os.path.join(bbox_dir, clip_id)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        cmd_line = cmd_line_template % dict(
            python=sys.executable,
            script=find_match_script,
            orig_path=os.path.join(frame_dir, clip_id),
            cropped_path=os.path.join(faces_dir, clip_id),
            save_path=save_path)
        print 'executing cmd:'
        print cmd_line
        subprocess.check_call(cmd_line, shell=True)


### Phase 2.1b: Fallback if Picasa did not find anything

# Use Ramanan keypoints algorithm to find and save keypoint detections.
# How to deal with multiple detections?
backup_faces_dir = os.path.join(DATA_ROOT_DIR, 'ramanan_keypoints_picassa_backup')
if not os.path.exists(backup_faces_dir):
    os.mkdir(backup_faces_dir)

script_dir = os.path.join(SCRIPTS_PATH, 'samira/RamananCodes')

if alt_path1:
    logging.info("Phase 2.1b -- Run Ramanan on full frames if Picasa did not find anythin")
    for clip_id in CLIP_IDS:
        logging.debug("2.1b: processing clip_id %s" % clip_id)
        this_clip_frame_dir = os.path.join(frame_dir, clip_id)
        this_clip_backup_faces_dir = os.path.join(backup_faces_dir, clip_id)

        if not os.path.exists(this_clip_frame_dir):
             os.mkdir(this_clip_frame_dir)
        if not os.path.exists(this_clip_backup_faces_dir):
             os.mkdir(this_clip_backup_faces_dir)
 
        # XXX Somebody check if picasa found a face in this clip skip it completely XXX

        if REMOTE_RAMANAN:
            # ensure remote directory exists
            remote.run_remote(['mkdir', '-p', REMOTE_DATA_PATH])

            logging.info("rsync '%s' to cluster... " % this_clip_frame_dir)
            remote.rsync_local_remote(this_clip_frame_dir, REMOTE_DATA_PATH )

            logging.info("Submitting jobs to queuing system on cluster...")
            remote.run_remote([REMOTE_SUBMIT_SCIPT, REMOTE_NO_WORKER, REMOTE_DATA_PATH, clip_id])
            
            # Check for all jobs finished
            test_cmd = ['test', '-e', REMOTE_DATA_PATH+clip_id+'/DONE.txt' ]
            while remote.run_remote(test_cmd, except_on_error=False) == 1:
                logging.info("... waiting for workpackage ...")
                time.sleep(REMOTE_POLLING_TIMEOUT)

            logging.info("Copy results back to local machine and deleting remote dir...")
            remote.rsync_remote_local(REMOTE_DATA_PATH+clip_id, this_clip_backup_faces_dir)
            remote.run_remote(['rm', '-Rf', REMOTE_DATA_PATH+clip_id])
        else:
            logging.info("2.1b: demoneim filepaths = ", this_clip_frame_dir, this_clip_backup_faces_dir, '\n')

            current_dir = os.getcwd()
            cmd_line_template = 'bash %(script)s %(frame_dir)s %(backup_faces_dir)s %(scriptdir)s %(currentdir)s'
            print "script_dir = ", script_dir
            cmd_line = cmd_line_template % dict(
                script=os.path.join(SCRIPTS_PATH, 'samira/RamananCodes', 'demoneimagewhole_alt.bash'),
                frame_dir=this_clip_frame_dir,
                backup_faces_dir=this_clip_backup_faces_dir,
                scriptdir = script_dir,
                currentdir = current_dir)
            print '\n', 'executing cmd:'
            print cmd_line, '\n', '\n', '\n', '\n'
            subprocess.check_call(cmd_line, shell=True)

### Phase 2.2b: bbox coordinates if Picasa did not find anything
# TODO: Raul, finish
if alt_path2:
    logging.info("Phase 2.2 -- Compute bounding boxes from Ramanan runs on full frames")
    #script_name = os.path.join(SCRIPTS_PATH, 'chandiar/missing_clips', 'get_bbox.py')
    #print script_name, type(script_name), str(script_name)

    backup_bboxes_dir = os.path.join(backup_faces_dir, 'bboxes')
    if not os.path.exists(backup_bboxes_dir):
        os.mkdir(backup_bboxes_dir)

    for clip_id in CLIP_IDS:
        this_clip_frame_dir = os.path.join(frame_dir, clip_id)
        if not os.path.exists(this_clip_frame_dir):
            os.mkdir(this_clip_frame_dir)
        this_clip_backup_faces_dir = os.path.join(backup_faces_dir, clip_id)
        if not os.path.exists(this_clip_backup_faces_dir):
            os.mkdir(this_clip_backup_faces_dir)
        this_clip_backup_bboxes_dir = os.path.join(backup_bboxes_dir, clip_id)
        if not os.path.exists(this_clip_backup_bboxes_dir):
           os.mkdir(this_clip_backup_bboxes_dir)
        print 'getbbox inputs = ', this_clip_frame_dir, this_clip_backup_faces_dir, this_clip_backup_bboxes_dir
        get_bbox(this_clip_frame_dir, this_clip_backup_faces_dir, this_clip_backup_bboxes_dir, this_clip_backup_bboxes_dir)



## Phase 2.3:
facetubes_dir = os.path.join(DATA_ROOT_DIR, 'facetubes_96x96')
if not os.path.exists(facetubes_dir):
    os.mkdir(facetubes_dir)

if smooth_facetubes:
    logging.info("Phase 2.3 -- Smooth facetubes")
    cmd_template = "%(python)s %(script)s %(orig_path)s %(bbox_path)s %(save_path)s"
    for clip_id in CLIP_IDS:
        save_path = os.path.join(facetubes_dir, clip_id)
        if os.path.exists(save_path):
            #raise Exception(
            #    "Directory for exporting smooth facetube images "
            #    "for clip id %s already exists: %s" % (clip_id, save_path))
            pass
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


### Phase 3: Poly's part of the pipeline, from smoothed facetubes to SVM prediction
#  from convnet's output
if run_svm_convnet:
    logging.info("Phase 3")
    samira_model_dir = '/data/lisa/exp/faces/emotiw_final/Samira'
    cmd_line_template = 'bash %(script)s %(clip_id)s %(model_dir)s %(data_root_dir)s'
    for clip_id in CLIP_IDS:
        cmd_line = cmd_line_template % dict(
            script=os.path.join(SCRIPTS_PATH, 'samira', 'Dmodel1.bash'),
            clip_id=clip_id,
            model_dir=samira_model_dir,
            data_root_dir=DATA_ROOT_DIR)
        print 'executing cmd:'
        print cmd_line
        subprocess.check_call(cmd_line, shell=True)

###
### audio module
if run_audio:
    logging.info("Phase 4 -- Audio")
    caglar_audio_model_dir = '/data/lisa/exp/faces/emotiw_final/caglar_audio'
    cmd_line_template = "%(python)s %(audio_script)s %(data)s %(feats)s %(output)s %(model_dir)s %(clip_id)s"
    for clip_id in CLIP_IDS:
        cmd_line = cmd_line_template % dict(
            python=sys.executable,
            audio_script=os.path.join(SCRIPTS_PATH, 'caglar', 'save_features_pascal_pipeline.py'),
            data=os.path.join(DATA_ROOT_DIR, 'Test_Vid_Distr', 'Data'),
            feats=os.path.join(DATA_ROOT_DIR, 'audio_feats'),
            output=PREDICTION_DIR,
            model_dir=caglar_audio_model_dir,
            clip_id=clip_id)
        print 'executing cmd:'
        print cmd_line
        subprocess.check_call(cmd_line, shell=True)

        os.rename(os.path.join(PREDICTION_DIR, 'audio_mlp_learned_on_train_predict_on_test_scores.npy'),
                  os.path.join(PREDICTION_DIR, 'audio_pred_%s.npy' % clip_id))
        os.rename(os.path.join(PREDICTION_DIR, 'audio_mlp_learned_on_train_predict_on_test_scores.txt'),
                  os.path.join(PREDICTION_DIR, 'audio_pred_%s.txt' % clip_id))


###
### Second SVM script from Poly, works on top of first SVM prediction and audio
# NB: The script is "Smodel1.bash", the first one was "Dmodel1.bash".
if run_svm_convnet_audio:
    logging.info("Phase 4 -- Second SVM")
    samira_model_dir = '/data/lisa/exp/faces/emotiw_final/Samira'
    cmd_line_template = 'bash %(script)s %(clip_id)s %(model_dir)s %(data_root_dir)s'
    for clip_id in CLIP_IDS:
        cmd_line = cmd_line_template % dict(
            script=os.path.join(SCRIPTS_PATH, 'samira', 'Smodel1.bash'),
            clip_id=clip_id,
            model_dir=samira_model_dir,
            data_root_dir=DATA_ROOT_DIR)
        print 'executing cmd:'
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
        print 'executing cmd:'
        print cmd_line
        subprocess.check_call(convert_line, shell=True)

        cmd_line = cmd_line_template % dict(
            python=sys.executable,
            inference_line=os.path.join(SCRIPTS_PATH, 'kishore', 'inference_line', 'inference_line.py'),
            centroids_file=os.path.join(kishore_model_root, 'kmeans_centroids.npy'),
            model_file=os.path.join(kishore_model_root, 'model_params.npz'),
            videos_path=mjpeg_dir,
            train_data_file=os.path.join(kishore_model_root, 'chal_train_data.npz'),
            clip_ids=clip_id)

        print 'executing cmd:'
        print cmd_line
        subprocess.check_call(cmd_line, shell=True)

        # The output will be a one-liner file in the current directory
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
        print 'executing cmd:'
        print cmd_line
        subprocess.check_call(cmd_line, shell=True)
        shutil.move(os.path.join(PREDICTION_DIR, 'BoMF_test_probabilities.npy'),
                    os.path.join(PREDICTION_DIR, 'bomf_pred_%s.npy' % clip_id))


# Xavier's weighted prediction
if run_xavier:

    weights_file = "/data/lisa/exp/faces/emotiw_final/bouthilx/weights_in_paper.npy"
    cmd_line_template = "%(python)s %(xavier_cmdline)s %(weights)s %(activity)s %(audio)s %(bagofmouth)s %(convnet)s %(convnet_audio)s %(output)s"
    for clip_id in CLIP_IDS:
        cmd_line = cmd_line_template % dict(
                python = sys.executable,
                xavier_cmdline = os.path.join(SCRIPTS_PATH, 'bouthilx', 'weighted_average.py'),
                weights = weights_file,
                activity = os.path.join(PREDICTION_DIR, 'kishore_pred_%s.txt' % clip_id),
                audio = os.path.join(PREDICTION_DIR, 'audio_pred_%s.npy' % clip_id),
                bagofmouth = os.path.join(PREDICTION_DIR, 'bomf_pred_%s.npy' % clip_id),
                convnet = os.path.join(PREDICTION_DIR, 'svm_convnet_pred_%s.mat' % clip_id),
                convnet_audio = os.path.join(PREDICTION_DIR, 'svm_convnet_audio_pred_%s.mat' % clip_id),
                output = os.path.join(PREDICTION_DIR, 'xavier_output_%s.npy' % clip_id))
        print 'executing cmd:'
        print cmd_line
        subprocess.check_call(cmd_line, shell = True)
