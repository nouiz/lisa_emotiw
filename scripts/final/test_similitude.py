import numpy as np
import scipy.io as sio

mat_orders = [range(7),
              range(7),
              range(7),
              [5,6,0,4,1,3,2],
              #[2,4,6,5,3,0,1],
              [0,1,2,3,5,6,4],
              #[0,1,2,3,6,4,5]
              range(7)]

s = 'valid'

if s=='test':
    templates = [("/u/ebrahims/emotiw_pipeline/test_set/module_predictions/kishore_pred_%s.txt",
                  "/data/lisa/data/faces/EmotiWTest/ModelPredictionsToCombine/ActivityRecognition/activity_learned_on_train_predict_on_%s_scores.npy"),
                 ("/u/ebrahims/emotiw_pipeline/test_set/module_predictions/audio_pred_%s.npy",
                  "/data/lisa/data/faces/EmotiWTest/ModelPredictionsToCombine/Audio/model2/audio_mlp_learned_on_train_predict_on_%s_scores.npy"),
                 ("/u/ebrahims/emotiw_pipeline/test_set/module_predictions/bomf_pred_%s.npy",
                  "/data/lisa/data/faces/EmotiWTest/ModelPredictionsToCombine/BagOfMouthFeatures/mouthBoF8_learned_on_train_predict_on_%s_scores.npy"),
                 ("/u/ebrahims/emotiw_pipeline/test_set/module_predictions/svm_convnet_pred_%s.mat",
                  "/data/lisa/data/faces/EmotiWTest/ModelPredictionsToCombine/ConvNetPlusSVM_PierreSamiraChris/learned_on_train_predict_on_%s_scores.npy"),
                 ("/u/ebrahims/emotiw_pipeline/test_set/module_predictions/svm_convnet_audio_pred_%s.mat",
                  "/data/lisa/data/faces/EmotiWTest/ModelPredictionsToCombine/ConvNetConcat/learned_on_train_predict_on_%s_scores.npy"),
                 ("/u/ebrahims/emotiw_pipeline/test_set/module_predictions/xavier_output_%s.npy",
                  "/data/lisa/data/faces/EmotiWTest/ModelPredictionsToCombine/Final/learned_on_train_predict_on_%s_scores.npy")]
elif s=='valid':
    templates = [("/u/ebrahims/emotiw_pipeline/validation_set/module_predictions/kishore_pred_%s.txt",
                  "/data/lisa/data/faces/EmotiWTest/ModelPredictionsToCombine/ActivityRecognition/activity_learned_on_train_predict_on_%s_scores.npy"),
                 ("/u/ebrahims/emotiw_pipeline/validation_set/module_predictions/audio_pred_%s.npy",
                  "/data/lisa/data/faces/EmotiWTest/ModelPredictionsToCombine/Audio/model2/audio_mlp_learned_on_train_predict_on_%s_scores.npy"),
                 ("/u/ebrahims/emotiw_pipeline/validation_set/module_predictions/bomf_pred_%s.npy",
                  "/data/lisa/data/faces/EmotiWTest/ModelPredictionsToCombine/BagOfMouthFeatures/mouthBoF8_learned_on_train_predict_on_%s_scores.npy"),
                 ("/u/ebrahims/emotiw_pipeline/validation_set/module_predictions/svm_convnet_pred_%s.mat",
                  "/data/lisa/data/faces/EmotiWTest/ModelPredictionsToCombine/ConvNetPlusSVM_PierreSamiraChris/learned_on_train_predict_on_%s_scores.npy"),
                 ("/u/ebrahims/emotiw_pipeline/validation_set/module_predictions/svm_convnet_audio_pred_%s.mat",
                  "/data/lisa/data/faces/EmotiWTest/ModelPredictionsToCombine/ConvNetConcat/learned_on_train_predict_on_%s_scores.npy"),
                 ("/u/ebrahims/emotiw_pipeline/validation_set/module_predictions/xavier_output_%s.npy",
                  "/data/lisa/data/faces/EmotiWTest/ModelPredictionsToCombine/Final/learned_on_train_predict_on_%s_scores.npy")]


ids = "/data/lisa/data/faces/EmotiWTest/ModelPredictionsToCombine/afew2_%s_filelist.txt"

CLIPS = []
# automatically load ids from file
if s=='test':
    CLIPS = [line.strip("\n").split(" ")[-1] for line in open(ids % 'test').readlines()]
elif s=='valid':
    CLIPS = [line.strip("\n").split("/")[-1] for line in open(ids % 'valid').readlines()]
#CLIPS = ['000147200', '000256440', '000606007', '000831400', '002916678', '003024767', '004008240', '005700400', '005711000', '011018760', '011616534' ]

predictions = [[] for i in templates]
oldpredictions = [[] for i in templates]

skipped_clips = [[] for i in templates]

def load_preds(path):
    if path.split(".")[-1]=="npy":
        preds = np.load(path)
        if np.prod(preds.shape) == 7:
            preds = preds.reshape(1,7)
        return preds
    elif path.split(".")[-1]=="mat":
        preds = sio.loadmat(path)['prob_values_test']
        if np.prod(preds.shape) == 7:
            preds = preds.reshape(1,7)
        return preds
    elif path.split(".")[-1]=="txt":
        txt_file = open(path,'r')
        preds = []
        for line in txt_file.readlines():
            # The 7 probabilities should be at the end of the line
            preds.append([float(prob) for prob in line.strip().split()[-7:]])
        return np.array(preds)
    else:
        raise ValueError("The file must be .npy (numpy), .txt or .mat (matlab)")

for i, [newmodel, oldmodel] in enumerate(templates):
    for clip in CLIPS:
        try:
            predictions[i].append(load_preds(newmodel % clip)[:,mat_orders[i]].argmax())
        except IOError:
            skipped_clips[i].append(clip)
            continue
        # find position of clip id and type of set (valid or train)
        position = -1
        order_file = open(ids % s,'r')
        for line in order_file.readlines():
            line = line.strip("\n")
            if (s=='test' and line.split(" ")[-1]==clip) or line.split("/")[-1]==clip:
                position = int(line.split(" ")[0])
                break

        if position == -1:
            raise BaseException("clip %s not found in templates" % clip)

        # load the old predictions for the model i
        oldpredictions[i].append(np.load(oldmodel % s)[position].argmax())#[position,mat_orders[i]].argmax())

    newp = np.array(predictions[i])
    oldp = np.array(oldpredictions[i])

    print newmodel
    print oldmodel
    print "skipped clips"
    print skipped_clips[i]
    print "mean over %d clips" % newp.shape[0]
    print np.mean(newp==oldp)*100
