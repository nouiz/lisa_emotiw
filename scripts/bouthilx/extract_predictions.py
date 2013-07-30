import numpy as np
import argparse
import os
import emotiw.bouthilx.utils as utils
base_path = "/data/lisatmp2/bouthilx/EmotiW/ModelPredictionsToCombine"
base_name = "learned_on_train_predict_on_%s_scores.npy"

modes = {
    "activity": "ActivityRecognition",
    "audio":"Audio",
    "samira":"ConvNetPlusSVM_PierreSamiraChris",
    "bagofmouth":"BagOfMouthFeatures",
    "mehdi":"MehdisModels",
    "pascal":"PascalAndXavier",
    "samira-audio":"ConvNetConcat"
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",default=".")
    parser.add_argument("--modes",action='append',required=True)
    parser.add_argument("--sets",action='append',required=True)
    parser.add_argument("--normalize",default=True)
    parser.add_argument("model_path",help="Pylearn2 model")

    options = parser.parse_args()

    from extract_features import get_features
    from emotiw.bouthilx.datasets import FeaturesDataset

    out = options.out
    d_modes = options.modes
    sets = options.sets
    model_path = options.model_path
    normalize = options.normalize

    targets = os.path.join(base_path,"afew2_train_targets.npy")
    
    from theano import config
    from theano import function
    for s in sets:
        features = [os.path.join(base_path,modes[mode],base_name % s) for mode in d_modes]
        fd = FeaturesDataset(features,targets,"",normalize,shuffle=False)
        data = np.cast[config.floatX](fd.get_design_matrix())
        preds = get_features(model_path,data,layer_idx=None)

        np.save(os.path.join(out,"_".join(d_modes)+"_"+s),preds)

if __name__=="__main__":
    main()
