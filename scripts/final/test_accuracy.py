import numpy as np
import scipy.io as sio

#predictions_template = "/u/ebrahims/emotiw_pipeline/validation_set/module_predictions/xavier_output_optional_%s.npy"

predictions_template = "/u/ebrahims/emotiw_pipeline/validation_set/module_predictions/svm_convnet_pred_%s.mat"
old_template = "/data/lisa/data/faces/EmotiWTest/ModelPredictionsToCombine/ConvNetPlusSVM_PierreSamiraChris/learned_on_train_predict_on_%s_scores.npy"

#predictions_template = "/u/ebrahims/emotiw_pipeline/validation_set/module_predictions/svm_convnet_audio_pred_%s.mat"

#predictions_template = "/u/ebrahims/emotiw_pipeline/validation_set/module_predictions/kishore_pred_%s.txt"

#predictions_template = "/u/ebrahims/emotiw_pipeline/validation_set/module_predictions/audio_pred_%s.npy"

#predictions_template = "/u/ebrahims/emotiw_pipeline/validation_set/module_predictions/bomf_pred_%s.npy"

ids = "/data/lisa/data/faces/EmotiWTest/ModelPredictionsToCombine/afew2_%s_filelist.txt"
labels_template = "/data/lisa/data/faces/EmotiWTest/ModelPredictionsToCombine/afew2_%s_targets.npy"

sets = ['valid','train'] # not test because we have no labels!

CLIPS = []
# automatically load ids from file
CLIPS = [line.strip("\n").split("/")[-1] for line in open(ids % 'valid').readlines()]

# Where the alt path was taken
CLIPS = ['000147200', '000256440', '000606007', '000831400', '002916678', '003024767', '004008240', '005700400', '005711000', '011018760', '011616534' ]

predictions = []
oldpredictions = []

labels = []

skipped_clips = []

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

for clip in CLIPS:
    try:
        predictions.append(load_preds(predictions_template % clip).argmax())
        # for ConvNet
#        predictions.append(load_preds(predictions_template % clip)[:,[5,6,0,4,1,3,2]].argmax())
        # for ConvNet-audio
#        predictions.append(load_preds(predictions_template % clip)[:,[0,1,2,3,5,6,4]].argmax())
    except IOError:
        skipped_clips.append(clip)
        continue
    # find position of clip id and type of set (valid or train)
    position = -1
    for s in sets:
        order_file = open(ids % s,'r')
        for line in order_file.readlines():
            line = line.strip("\n")
            if line.split("/")[-1]==clip:
                position = int(line.split(" ")[0])
                break

        if position != -1:
            break

    # load the valid or train file and select label at good position
    labels.append(np.load(labels_template % s)[position])
    oldpredictions.append(np.load(old_template % s)[position].argmax())

predictions = np.array(predictions)
labels = np.array(labels)

print "skipped clips"
print skipped_clips
print "mean over %d clips" % predictions.shape[0]
print np.mean(predictions==labels)*100
print np.mean(oldpredictions==labels)*100
