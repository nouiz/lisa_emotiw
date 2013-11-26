import sys
import cPickle
import os
import re
import time
import glob

import PIL.Image
from sklearn import preprocessing
import numpy
import theano
import ml
from jobman import DD
import jobman, jobman.sql

from subprocess import call
from audio_features import export_features

def run_pipeline(file_dir,
                 clip_ids,
                 features_dir,
                 scores_out_dir,
                 params_dir):

    audiofiles = []

    for clip_id in clip_ids:
        video_file_path = os.path.join(file_dir, clip_id + '.avi')
        audio_file = clip_id + ".mp3"
        audio_file_path = os.path.join(file_dir, audio_file)
        audiofiles.append(audio_file)

        try:
            retcode = call("mplayer -dumpaudio -dumpfile %s %s" %(audio_file_path,
                video_file_path), shell=True)
            if retcode < 0:
                print >>sys.stderr, "Child was terminated by signal", -retcode
            else:
                print >>sys.stderr, "Child returned", retcode
        except OSError as e:
            print >>sys.stderr, "Execution failed:", e
    export_features(file_dir, audiofiles=audiofiles, out=features_dir,
                    train_file_path="/data/lisa/data/faces/EmotiW/audios/Train/")

    run_nnet(clip_ids, features_dir, scores_out_dir, params_dir)

def run_nnet(clip_ids, features_dir, scores_out_dir, params_dir, **kwargs):

    file_dir=None
    learning_rate=0.001
    no_final_dropout=False
    hidden_dropout=.5
    center_grads=False
    layer_dropout=False
    rmsprop = True
    ratio=1.0
    train_epochs=240
    K=1
    state=None
    channel=None
    n_hiddens=310
    use_nesterov=1
    learning_rate=0.000502311626412
    momentum=0.459258503786
    features="full.pca"
    example_dropout=98
    rbm_epochs=15
    topN_pooling=1
    mean_pooling=0
    rbm_batch_size=60
    normalize_acts=False
    enable_standardization=0
    loss_based_pooling=False
    response_normalize=False
    rmsprop=1
    layer_dropout=True
    no_final_dropout=0
    l2=1e-05
    hidden_dropout=0.121193556495
    max_col_norm=1.2875
    n_layers=2
    rho=0.92

    print "Loading dataset..."
    LABELS = ["Disgust",  "Fear",  "Happy",  "Neutral",  "Sad",  "Surprise", "Angry"]
    nclasses = len(LABELS)

    numpy.random.seed(0x7265257d5f)
    final_audio_path = "/data/lisa/exp/faces/emotiw_final/caglar_audio/"

    test_files = glob.glob("%s/*.pkl" % features_dir)
    test_x = []
    test_y = []
    test_filenames = []

    print "Loading test set..."
    for clip_id in clip_ids:
        test_file = "%s/%s.%s.pkl" % (features_dir, clip_id, features)
        feat = numpy.load(test_file)
        test_filenames.append(test_file.replace("pkl", "npy"))
        test_x.append(numpy.asarray(feat, theano.config.floatX))

    test_file_h = open("test_file.txt", "w")
    test_file_h.writelines(["%s\n" % item for item in test_filenames])

    test_means = numpy.load(os.path.join(final_audio_path, "test_means.npy"))
    test_means = numpy.cast[theano.config.floatX](test_means)

    print "Building model..."

    layers = (n_layers - 1) * [('R', n_hiddens)] + [('L', n_hiddens)]  +  [('S', nclasses)]

    model = ml.MLP(n_in=test_x[0].shape[1],
                   layers=layers,
                   learning_rate=learning_rate,
                   l2=l2,
                   rho=rho,
                   rmsprop=rmsprop,
                   response_normalize=response_normalize,
                   center_grads=center_grads,
                   max_col_norm=max_col_norm,
                   loss_based_pooling=loss_based_pooling,
                   hidden_dropout=hidden_dropout,
                   layer_dropout=layer_dropout,
                   use_nesterov=use_nesterov,
                   normalize_acts=normalize_acts,
                   topN_pooling=topN_pooling,
                   mean_pooling=mean_pooling,
                   no_final_dropout=no_final_dropout,
                   enable_standardization=enable_standardization,
                   momentum=momentum,
                   base_path=params_dir)

    model.load()
    train_error = 0.

    test_feats = []
    test_preds = []

    #Testing the means
    for minibatch in range(len(test_x)):
        x = test_x[minibatch] - test_means
        test_feats.append(model.pooled_output_features(x))
        test_preds.append(model.compute_preds(x))

    test_feats = numpy.asarray(test_feats, dtype="float32")
    test_feats_dir = os.path.join(scores_out_dir, "audio_mlp_test_feats.npy")
    test_preds_dir = os.path.join(scores_out_dir, "audio_mlp_learned_on_train_predict_on_test_scores.npy")
    test_preds = numpy.asarray(test_preds).flatten().reshape(-1, 7)[:, (6,0,1,2,4,5,3)]

    numpy.save(test_feats_dir, test_feats)
    numpy.save(test_preds_dir, test_preds)
    # Also save as .txt for the SVM prediction
    test_preds_txt = open(os.path.join(scores_out_dir, "audio_mlp_learned_on_train_predict_on_test_scores.txt"), 'w')
    for item, pred in zip(clip_ids, test_preds):
        label = LABELS[pred.argmax()]
        print >>test_preds_txt, item, label, ('%f ' * 7) % tuple([p for p in pred])
    test_preds_txt.close()



if __name__ == "__main__" :
    #run_pipeline(file_dir="../Test_Vid_Distr/Data/",
    #             clip_ids=["000152960"],
    #             features_dir="./audio_feats2/",
    #             scores_out_dir="./scores/")

    file_dir = sys.argv[1]
    features_dir = sys.argv[2]
    scores_out_dir = sys.argv[3]
    params_dir = sys.argv[4]
    clip_ids = sys.argv[5:]
    run_pipeline(
        file_dir=file_dir,
        features_dir=features_dir,
        scores_out_dir=scores_out_dir,
        clip_ids=clip_ids,
        params_dir=params_dir)
