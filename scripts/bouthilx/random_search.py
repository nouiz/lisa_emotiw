import numpy as np
import scipy.io as sio

mat_orders = [range(7),
              range(7),
              range(7),
              [5,6,0,4,1,3,2],
              #[2,4,6,5,3,0,1],
              [0,1,2,3,5,6,4]]
              #[0,1,2,3,6,4,5]]

templates = ["/u/ebrahims/emotiw_pipeline/validation_set/module_predictions/kishore_pred_%s.txt", 
             "/u/ebrahims/emotiw_pipeline/validation_set/module_predictions/audio_pred_%s.npy", 
             "/u/ebrahims/emotiw_pipeline/validation_set/module_predictions/bomf_pred_%s.npy",  
             "/u/ebrahims/emotiw_pipeline/validation_set/module_predictions/svm_convnet_pred_%s.mat", 
             "/u/ebrahims/emotiw_pipeline/validation_set/module_predictions/svm_convnet_audio_pred_%s.mat"]

ids = "/data/lisa/data/faces/EmotiWTest/ModelPredictionsToCombine/afew2_%s_filelist.txt"
labels_template = "/data/lisa/data/faces/EmotiWTest/ModelPredictionsToCombine/afew2_%s_targets.npy"

save_path = "/data/lisa/exp/faces/emotiw_final/bouthilx/best_weights.npy"

# we search on validation set
s = 'valid'
# automatically load ids from file
CLIPS = [line.strip("\n").split("/")[-1] for line in open(ids % s).readlines()]

def load_preds(path):
    if path.split(".")[-1]=="npy":
        preds = np.load(path)
        preds = preds
        return preds.reshape(1,7)
    elif path.split(".")[-1]=="mat":
        preds = sio.loadmat(path)['prob_values_test']
        return preds.reshape(1,7)
    elif path.split(".")[-1]=="txt":
        txt_file = open(path,'r')
        preds = []
        for line in txt_file.readlines():
            # The 7 probabilities should be at the end of the line
            preds.append([float(prob) for prob in line.strip().split()[-7:]])
        return np.array(preds).reshape(1,7)
    else:
        raise ValueError("The file must be .npy (numpy), .txt or .mat (matlab)")


def extract_predictions():
    predictions = []
    labels = []
    skipped_clips = []

    for clip in CLIPS:
        clip_predictions = []
        for template, mat_order in zip(templates, mat_orders):
            try:
                clip_predictions.append(load_preds(template % clip)[:,mat_order])
            except IOError:
                skipped_clips.append((clip,template))
                clip_predictions.append(np.ones((1,7))*1/7.0)
        predictions.append(np.concatenate(clip_predictions,axis=0))
        # find position of clip id and type of set (valid or train)
        position = -1
        order_file = open(ids % s,'r')
        for line in order_file.readlines():
            line = line.strip("\n")
            if line.split("/")[-1]==clip:
                position = int(line.split(" ")[0])
                break

        if position == -1:
            raise BaseException("clip %s not found in templates" % clip)

        # load the valid or train file and select label at good position
        labels.append(np.load(labels_template % s)[position])

    predictions = np.array(predictions)
    labels = np.array(labels)

    print "skipped clips"
    print skipped_clips

    print predictions.shape
    print labels.shape
    return predictions, labels

predictions, labels = extract_predictions()

best = (None,0.0)
r = 1.0

K_u = 4000*100
# K_u trials uniformly distributed
for k in range(K_u/4000):
    bunch_of_weights = np.random.random((4000,5,7))
    bunch_of_weights = bunch_of_weights/bunch_of_weights.sum(1)[:,None,:]
    for weights in bunch_of_weights:
        mean_pred = (predictions*weights).sum(1)
        rval = np.mean(mean_pred.argmax(1)==labels)
        if rval > best[1]:
            best = (weights,rval)#(base,rval)

    print "Uniform search, round %d/%s" % (k*4000,K_u)
#    print best[0]
    print best[1]
    np.save(save_path,best[0])

# Infinite trials around last best weights
K_g = 0
while True:
    tests = np.random.normal(scale=0.10,size=(4000,5,7))
    tests = (tests * (tests < r)) * (tests > -r)

    for test in tests:
        new = test+best[0]
        new = new * (new > 0)
        new = np.round(new/np.sum(new,axis=0)[None,:],decimals=2)
        mean_pred = predictions*new
        if s!='test':
            rval = np.mean(mean_pred.argmax(1)==labels)
            if rval > best[1]:
                best = (new,rval)

    print "Local search, round %d" % K_g
#    print best[0]
    print best[1]
    np.save(save_path,best[0])

    K_g += 1
