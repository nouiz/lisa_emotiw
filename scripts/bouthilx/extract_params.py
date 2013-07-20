#yaml/test.yaml
#yaml2/test6.yaml

import numpy as np
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",default=".",help="directory where to save npy files")
    parser.add_argument("models_paths",nargs="+",help="models paths")
    options = parser.parse_args()

    models_paths = options.models_paths
    out = os.path.splitext(options.out)[0]

    from pylearn2.utils import serial
    for model_path in models_paths:
        model = serial.load(model_path)
        params_out = os.path.splitext(model_path.split("/")[-1])[0]
        print params_out

        params = model.get_params()

        for param in params:
            path = out+"/"+params_out+"_"+param.name+".npy"
            np.save(path,param.get_value())

if __name__=="__main__":
    main()
