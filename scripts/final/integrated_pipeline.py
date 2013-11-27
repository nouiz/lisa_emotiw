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
extract_frames = 1
run_picasa = 1
extract_bbox = 1
smooth_facetubes = 1
run_svm_convnet = 1

alt_path1 = 1       # run Ramanan on full frames if picasa did not find anything
alt_path2 = 1       # postprocess alt_path1 output

run_audio = 1
run_svm_convnet_audio = 1

run_kishore = 1
run_bomf = 1

run_xavier = 1
export_final = 1

### Configureation ###

REMOTE_RAMANAN = 1                                    # use remote ramanan computation
REMOTE_USER_HOST='bornj@cudahead.rdgi.polymtl.ca'     # cluster account to use
REMOTE_DATA_PATH='tmp/jobdata/'                       # directory there the workpackages will be copied
REMOTE_NO_WORKER='24'                                 # number of parallel workers on the cluster
REMOTE_POLLING_TIMEOUT=10                             # polling interval in seconds
REMOTE_SUBMIT_SCIPT='lisa_emotiw/scripts/jorg/remote/submit-worker.py' 

# Picasa incoming directory
PICASA_PROCESSING_DIR = '/data/lisatmp/faces/picasa_process'


# Default prediction if a module fails
DEFAULT_PREDICITION_DIR = '/data/lisa/exp/faces/emotiw_final/default_pred/uniform/'

# More imports
sys.path.append('../')
sys.path.append('/data/lisa/exp/faces/emotiw_final/Kishore/inference_line/')

import jorg.remote as remote
remote.REMOTE_USER_HOST = REMOTE_USER_HOST

# Root directory for the data read and generated
DATA_ROOT_DIR = '/u/ebrahims/emotiw_pipeline/workshop_demo'

# Initial directory containing the *.avi files,
# relative to DATA_ROOT_DIR
AVI_DIR = 'Vid_Distr/Data'
AVI_DIR = os.path.join(DATA_ROOT_DIR, AVI_DIR)

# Initial directory containing faces aligned by the organizers
ALIGNED_DIR = 'Faces_Aligned'
ALIGNED_DIR = os.path.join(DATA_ROOT_DIR, ALIGNED_DIR)

# Data where we put the individual predictions
PREDICTION_DIR = 'module_predictions'
PREDICTION_DIR = os.path.join(DATA_ROOT_DIR, PREDICTION_DIR)
if not os.path.exists(PREDICTION_DIR):
    os.mkdir(PREDICTION_DIR)

# Directory where we export the final prediction and probabilities for each clip
FINAL_PREDICTION_DIR = 'final_prediction'
FINAL_PREDICTION_DIR = os.path.join(DATA_ROOT_DIR, FINAL_PREDICTION_DIR)
if not os.path.exists(FINAL_PREDICTION_DIR):
    os.mkdir(FINAL_PREDICTION_DIR)

# Names of clips to process
#CLIP_IDS = [
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
#    '002350040',
#    'our-little-one'
#    ]
CLIP_IDS = sys.argv[1:]
if not CLIP_IDS:
    logging.error("No clip IDs provided on the command line.")
    sys.exit(-1)

#### Hack to process all the clips at the same time
#if not CLIP_IDS:
#    # The whole content of AVI_DIR
#    import glob
#    CLIP_IDS = [os.path.basename(f).rsplit('.', 1)[0]
#                for f in sorted(glob.glob(os.path.join(AVI_DIR, '*.avi')))]

### Print general information ###
logging.info("Processing %d clips: %s", len(CLIP_IDS), CLIP_IDS)

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
        if not os.path.exists(clip_frame_dir):
            os.mkdir(clip_frame_dir)

        # sys.executable is the full path to python
        cmd_line = '%s %s %s %s %s' % (
            sys.executable, script_name, clip_dir, clip_frame_dir,
            '%s.avi' % clip_id)

        try:
            logging.debug("executing cmd: %s", cmd_line)
            subprocess.check_call(cmd_line, shell=True)
        except subprocess.CalledProcessError:
            logging.warn('WARNING: Frame extraction crashed on %s', clip_id)


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
        clip_frame_dir = os.path.join(frame_dir, clip_id)
        clip_picasa_incoming = os.path.join(PICASA_PROCESSING_DIR, clip_id)
        # This one contains the full frames, after processing
        clip_picasa_processed_dir = os.path.join(
            PICASA_PROCESSING_DIR, '%s.processed' % clip_id)
        # This one contains the extracted faces
        clip_picasa_faces_dir = os.path.join(
            PICASA_PROCESSING_DIR, '%s.faces' % clip_id)
        # This is where we put the faces afterwards
        clip_faces_dir = os.path.join(faces_dir, clip_id)

        if os.path.exists(clip_faces_dir):
            print '%s skipping picasa processing' % clip_id
            continue

        n_attempts = 3
        restart_file = os.path.join(PICASA_PROCESSING_DIR, 'RESTART')
        picasa_succeeded = False
        for attempt in xrange(n_attempts):
            # Copy the directory containing frames to Picasa's incoming directory
            shutil.copytree(clip_frame_dir, clip_picasa_incoming)

            # Make sure there is always a face in there
            shutil.copy('/data/lisa/exp/faces/emotiw_final/lamblinp/yoshua.jpg',
                        clip_picasa_incoming)

            # Then, append '.process_me' to its name, to signal the Windows script
            # that it can proceed
            print '%s enters picasa processing' % clip_id
            os.rename(clip_picasa_incoming, '%s.process_me' % clip_picasa_incoming)

            # Wait for Picasa to return
            for i in xrange(300):
                time.sleep(1)
                if os.path.exists(clip_picasa_processed_dir):
                    print '  done.'
                    picasa_succeeded = True
                    break

            if picasa_succeeded:
                break
            else:
                # Cleanup:
                # Remove all directories we created or expected with
                # the clip_id in its name
                print "Picasa timed out %i times, cleaning up" % (attempt + 1)
                shutil.rmtree(clip_picasa_incoming, ignore_errors=1)
                shutil.rmtree('%s.process_me' % clip_picasa_incoming, ignore_errors=1)
                shutil.rmtree(clip_picasa_processed_dir, ignore_errors=1)
                shutil.rmtree(clip_picasa_faces_dir, ignore_errors=1)
                if attempt == n_attempts - 1:
                    raise Exception("Picasa script timed out too many times (%i), aborting" % n_attempts)
                else:
                    # Create an empty file named 'RESTART', to signal the script to start again
                    open(restart_file, 'a').close()
                    for i in xrange(10):
                        time.sleep(5)
                        if not os.path.exists(restart_file):
                            break
                    else:
                        # executed if the "break" was not triggered
                        raise Exception("Picasa script did not respond to RESTART request, aborting")


        assert os.path.exists(clip_picasa_faces_dir)
        if os.path.exists(clip_faces_dir):
            # It may or may not be correct, remove it
            shutil.rmtree(clip_faces_dir)
        shutil.copytree(clip_picasa_faces_dir, clip_faces_dir)

        # Clean up, so the Windows script does not crash if the same
        # clip gets extracted again
        shutil.rmtree(clip_picasa_processed_dir)
        shutil.rmtree(clip_picasa_faces_dir)

        # Remove Yoshua's face
        os.remove(os.path.join(clip_faces_dir, 'yoshua_picassa.jpg'))


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
        try:
            logging.debug("executing cmd: %s", cmd_line)
            subprocess.check_call(cmd_line, shell=True)
        except subprocess.CalledProcessError:
            logging.warn('WARNING: module bbox crashed on %s', clip_id)

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
        # If picasa found a face in this clip, skip it completely
        this_clip_picasa_faces_dir = os.path.join(faces_dir, clip_id)
        nb_picasa_faces = len([f
                               for f in os.listdir(this_clip_picasa_faces_dir)
                               if f.endswith('.jpg')])
        if nb_picasa_faces > 0:
            logging.debug("2.1b: skipping clip_id %s" % clip_id)
            continue

        logging.debug("2.1b: processing clip_id %s" % clip_id)
        this_clip_frame_dir = os.path.join(frame_dir, clip_id)
        this_clip_backup_faces_dir = os.path.join(backup_faces_dir, clip_id)

        if not os.path.exists(this_clip_frame_dir):
             os.mkdir(this_clip_frame_dir)
        if not os.path.exists(this_clip_backup_faces_dir):
             os.mkdir(this_clip_backup_faces_dir)
 
        if REMOTE_RAMANAN:
            try:
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
            except remote.RemoteExecutionError:
                logging.warn('WARNING: cluster ramanan crashed on %s -- skipping clip', clip_id)
                input("Press Enter to continue...")
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
            try:
                logging.debug("executing cmd: %s", cmd_line)
                subprocess.check_call(cmd_line, shell=True)
            except subprocess.CalledProcessError:
                logging.warn('WARNING: module ramanan crashed on %s -- skipping clip', clip_id)

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
        # If picasa found a face in this clip, skip it completely
        this_clip_picasa_faces_dir = os.path.join(faces_dir, clip_id)
        nb_picasa_faces = len([f
                               for f in os.listdir(this_clip_picasa_faces_dir)
                               if f.endswith('.jpg')])
        if nb_picasa_faces > 0:
            continue

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
    logging.info("="*77)
    logging.info("Phase 3 -- SVM ConvNet")
    samira_model_dir = '/data/lisa/exp/faces/emotiw_final/Samira'
    cmd_line_template = 'bash %(script)s %(clip_id)s %(model_dir)s %(data_root_dir)s'
    for clip_id in CLIP_IDS:
        cmd_line = cmd_line_template % dict(
            script=os.path.join(SCRIPTS_PATH, 'samira', 'Dmodel1.bash'),
            clip_id=clip_id,
            model_dir=samira_model_dir,
            data_root_dir=DATA_ROOT_DIR)
        try:
            logging.debug('executing cmd: %s', cmd_line)
            subprocess.check_call(cmd_line, shell=True)
        except subprocess.CalledProcessError:
            logging.warn('WARNING: module crashed on %s -- USING DEFAULT PREDICTIONS', clip_id)
            shutil.copyfile(os.path.join(DEFAULT_PREDICITION_DIR, 'svm_convnet_pred.mat'),
                            os.path.join(PREDICTION_DIR, 'svm_convnet_pred_%s.npy' % clip_id))

###
### audio module
if run_audio:
    logging.info("="*77)
    logging.info("Phase 4 -- Audio")
    caglar_audio_model_dir = '/data/lisa/exp/faces/emotiw_final/caglar_audio'
    cmd_line_template = "%(python)s %(audio_script)s %(data)s %(feats)s %(output)s %(model_dir)s %(clip_id)s"
    for clip_id in CLIP_IDS:
        cmd_line = cmd_line_template % dict(
            python=sys.executable,
            audio_script=os.path.join(SCRIPTS_PATH, 'caglar', 'save_features_pascal_pipeline.py'),
            data=AVI_DIR,
            feats=os.path.join(DATA_ROOT_DIR, 'audio_feats'),
            output=PREDICTION_DIR,
            model_dir=caglar_audio_model_dir,
            clip_id=clip_id)
        try:
            logging.debug('executing cmd: %s', cmd_line)
            subprocess.check_call(cmd_line, shell=True)

            os.rename(os.path.join(PREDICTION_DIR, 'audio_mlp_learned_on_train_predict_on_test_scores.npy'),
                      os.path.join(PREDICTION_DIR, 'audio_pred_%s.npy' % clip_id))
            os.rename(os.path.join(PREDICTION_DIR, 'audio_mlp_learned_on_train_predict_on_test_scores.txt'),
                      os.path.join(PREDICTION_DIR, 'audio_pred_%s.txt' % clip_id))
        except subprocess.CalledProcessError:
            logging.warn('WARNING: Audio module crashed on %s -- using default predictions', clip_id)
            shutil.copyfile(os.path.join(DEFAULT_PREDICITION_DIR, 'audio_pred.npy'),
                            os.path.join(PREDICTION_DIR, 'audio_pred_%s.npy' % clip_id))
            shutil.copyfile(os.path.join(DEFAULT_PREDICITION_DIR, 'audio_pred.txt'),
                            os.path.join(PREDICTION_DIR, 'audio_pred_%s.txt' % clip_id))

### Second SVM script from Poly, works on top of first SVM prediction and audio
# NB: The script is "Smodel1.bash", the first one was "Dmodel1.bash".
if run_svm_convnet_audio:
    logging.info("="*77)
    logging.info("Phase 4 -- SVM ConvNet Audio")
    samira_model_dir = '/data/lisa/exp/faces/emotiw_final/Samira'
    cmd_line_template = 'bash %(script)s %(clip_id)s %(model_dir)s %(data_root_dir)s'
    for clip_id in CLIP_IDS:
        cmd_line = cmd_line_template % dict(
            script=os.path.join(SCRIPTS_PATH, 'samira', 'Smodel1.bash'),
            clip_id=clip_id,
            model_dir=samira_model_dir,
            data_root_dir=DATA_ROOT_DIR)
        try:
            logging.debug('executing cmd: %s', cmd_line)
            subprocess.check_call(cmd_line, shell=True)
        except subprocess.CalledProcessError:
            logging.warn('WARNING: module crashed on %s -- USING DEFAULT PREDICTIONS', clip_id)
            shutil.copyfile(os.path.join(DEFAULT_PREDICITION_DIR, 'svm_convnet_audio_pred.mat'),
                            os.path.join(PREDICTION_DIR, 'svm_convnet_audio_pred_%s.mat' % clip_id))

###
### Kishore's module (activity recognition)
if run_kishore:
    logging.info("="*77)
    logging.info("Phase 5 -- Kishore's activity recognition")
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
        cmd_line = cmd_line_template % dict(
            python=sys.executable,
            inference_line=os.path.join(SCRIPTS_PATH, 'kishore', 'inference_line', 'inference_line.py'),
            centroids_file=os.path.join(kishore_model_root, 'kmeans_centroids.npy'),
            model_file=os.path.join(kishore_model_root, 'model_params.npz'),
            videos_path=mjpeg_dir,
            train_data_file=os.path.join(kishore_model_root, 'chal_train_data.npz'),
            clip_ids=clip_id)
        try:
            logging.debug('executing cmd: %s', cmd_line)
            subprocess.check_call(convert_line, shell=True)
            logging.debug('executing cmd: %s', cmd_line)
            subprocess.check_call(cmd_line, shell=True)

            # The output will be a one-liner file in the current directory
            shutil.move(os.path.join('activity_recognition_test_results.txt'),
                        os.path.join(PREDICTION_DIR, 'kishore_pred_%s.txt' % clip_id))
        except subprocess.CalledProcessError:
            logging.warn('WARNING: Kishores module crashed on %s -- USING DEFAULT PREFICTIONS', clip_id)
            shutil.copy(os.path.join(DEFAULT_PREDICITION_DIR, 'kishore_pred.txt'),
                        os.path.join(PREDICTION_DIR, 'kishore_pred_%s.txt' % clip_id))


###
### Sebastien Jean's module (bag of mouth features)
if run_bomf:
    logging.info("="*77)
    logging.info("Running bag of mouth module")
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
        try:
            logging.debug('executing cmd: %s', cmd_line)
            subprocess.check_call(cmd_line, shell=True)
            shutil.move(os.path.join(PREDICTION_DIR, 'BoMF_test_probabilities.npy'),
                        os.path.join(PREDICTION_DIR, 'bomf_pred_%s.npy' % clip_id))
        except subprocess.CalledProcessError:
            logging.warn('WARNING: Bag-of-mouth module crashed on %s with batch_size 50 -- retrying with 51', clip_id)
            cmd_line = cmd_line_template % dict(
                python=sys.executable,
                bomf_cmdline=os.path.join(SCRIPTS_PATH, 'jeasebas', 'BoMF_cmdline.py'),
                aligned_faces_dir=ALIGNED_DIR,
                small_faces_outdir=small_faces_outdir,
                model_dir=bomf_model_dir,
                pred_dir=PREDICTION_DIR,
                batch_size=51,
                clip_ids=clip_id)
            try:
                logging.debug('executing cmd: %s', cmd_line)
                subprocess.check_call(cmd_line, shell=True)
                shutil.move(os.path.join(PREDICTION_DIR, 'BoMF_test_probabilities.npy'),
                            os.path.join(PREDICTION_DIR, 'bomf_pred_%s.npy' % clip_id))
            except subprocess.CalledProcessError:
                logging.warn('WARNING: Bag-of-mouth module crashed again on %s -- USING DEFAULT PREDICTIONS', clip_id)
                shutil.copy(os.path.join(DEFAULT_PREDICITION_DIR, 'bomf_pred.npy'),
                            os.path.join(PREDICTION_DIR, 'bomf_pred_%s.txt' % clip_id))

# Xavier's weighted prediction
if run_xavier:
    logging.info("="*77)
    logging.info("Running final weighted prediction")
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
        try:
            logging.debug('executing cmd: %s', cmd_line)
            subprocess.check_call(cmd_line, shell = True)
        except subprocess.CalledProcessError:
            logging.warn('WARNING: Weighted prediction crashed on %s -- FINAL OUTPUT NOT GENERATED', clip_id)

### Final label prediction and probabilities in the format the organizers expected
if export_final:
    logging.info("Exporting final label and probabilities")
    cmd_line_template = "%(python)s %(export_script)s %(pred_dir)s %(final_pred_dir)s %(clip_id)s"
    for clip_id in CLIP_IDS:
        cmd_line = cmd_line_template % dict(
            python=sys.executable,
            export_script=os.path.join(SCRIPTS_PATH, 'final', 'export_labels_and_probs.py'),
            pred_dir=PREDICTION_DIR,
            final_pred_dir=FINAL_PREDICTION_DIR,
            clip_id=clip_id)
        try:
            logging.debug('executing cmd: %s', cmd_line)
            subprocess.check_call(cmd_line, shell=True)
        except subprocess.CalledProcessError:
            logging.warn('WARNING: Export crashed on %s -- FINAL PREDICTION NOT EXPORTED', clip_id)
