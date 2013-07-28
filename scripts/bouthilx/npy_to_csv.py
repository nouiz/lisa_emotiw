import numpy as np
import os
import argparse
import scipy 
from extract_predictions import base_path, base_name

from mean_predictions import class_precision

classnames = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

shortcut = {
    "activity": "ActivityRecognition",
    "samira":"ConvNetPlusSVM_PierreSamiraChris",
    "bag":"BagOfMouthFeatures",
    "mehdi":"MehdisModels",
    "pascal":"PascalAndXavier"
}

sets = ['train','valid','test']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",default=".")
    parser.add_argument("--normalize",default=True)
    parser.add_argument("--features",action='append')
    parser.add_argument("--predictions",action='append')

    options = parser.parse_args()

    from emotiw.bouthilx.datasets import FeaturesDataset


    out = options.out
    normalize = "True"==options.normalize
    feats = options.features
    preds = options.predictions

    if feats is None:
        feats = []
    if preds is None:
        preds = []


    print normalize

#    if feats:
#        base_name = "%s_features.npy"
#    else:
#        base_name = "learned_on_train_predict_on_%s_scores.npy"

    if len(feats+preds)>1:
        feats_names = "_".join(feats) + "_features" if len(feats) else ""
        preds_names = "_".join(preds) + "_predicts" if len(preds) else ""
        name = feats_names + "_" + preds_names
        if not normalize:
            name = "not_norm_"+name
#        name = "weighted"+name
        matfile = os.path.join(base_path,name)
        csvfile = os.path.join(base_path,name+'_%s.csv')
    elif len(feats):
        name = os.path.splitext(base_name)[0]+"features.mat"
        if not normalize:
            name = "not_norm_"+name

        matfile = os.path.join(base_path,shortcut.get(feats[0],feats[0]),name).replace('%s','')
        csvfile = os.path.join(base_path,name+'_features_%s.csv')
    elif len(preds):
        name = os.path.splitext(base_name)[0]+"predictions.mat"
        if not normalize:
            name = "not_norm_"+name

        matfile = os.path.join(base_path,shortcut.get(preds[0],preds[0]),name).replace('%s','')
        csvfile = os.path.join(base_path,name+'_predictions_%s.csv')

    print matfile
    d = {}

    for s in sets:
        print s

        filelist = open(os.path.join(base_path,"afew2_%s_filelist.txt") % s)
        ids = [int(i.strip().split(" ")[-1].split("/")[-1]) for i in filelist.readlines()]
        targets = os.path.join(base_path,"afew2_%s_targets.npy") % s
        if s=="test":
            targets = targets.replace("test","train")

        tmpcsvfile = open(csvfile % s,'w')

        features = [os.path.join(base_path,shortcut.get(mode,mode),"%s_features.npy" % s) for mode in feats]
        predictions = [os.path.join(base_path,shortcut.get(mode,mode),"learned_on_train_predict_on_%s_scores.npy" % s) for mode in preds]
        fd = FeaturesDataset(features+predictions,targets,"",normalize,shuffle=False)
        #precisions = class_precision(predictions)
        data = fd.get_design_matrix()#*precisions.flatten()
        save(ids,data,tmpcsvfile,fd.get_targets() if s!='test' else None)
        if s!='_test':
            d[s+'_labels'] = fd.get_targets().argmax(1)
        d[s+'_ids'] = ids
        d[s+'_features'] = data
#        np.save(os.path.join(base_path,"take_best_%s.npy" %s), data)

    scipy.io.savemat(matfile,mdict=d)

def save(ids,npy,csvfile,targets=None):
    if targets is not None:
        for clip_id, target, row in zip(ids,targets,npy):
            csvfile.write("%s %s %s\n" % (clip_id,classnames[target.argmax()]," ".join([str(i) for i in row])))

    else:
        for clip_id, row in zip(ids,npy):
            csvfile.write("%s %s\n" % (clip_id," ".join([str(i) for i in row])))

    csvfile.close()

if __name__=="__main__":
    main()
