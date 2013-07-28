import traceback
import sys
import os
import argparse
import numpy as np

from emotiw.bouthilx.datasets import FeaturesDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",required=True)
    parser.add_argument("--features",required=True,action='append')
    parser.add_argument("--targets")
#    parser.add_argument("--base_path",default="")
    parser.add_argument("--normalize",default=True)
    options = parser.parse_args()

    out = options.out
    features = options.features
    targets = options.targets
#    base_path = options.base_path
    normalize = "True"==options.normalize
    print normalize,type(normalize)

    if targets is None:
        targets = "/data/afew/ModelPredictionsToCombine/afew2_train_targets.npy"

    fd = FeaturesDataset(features,targets,"",
                    normalize,shuffle=False)

    data = fd.get_design_matrix()
    print data.shape
    np.save(out,data)

if __name__=="__main__":
    main()
