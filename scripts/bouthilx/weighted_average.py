import numpy as np
import argparse
from scipy import io as sio

# Order of the modelFiles
# Activity      - numpy or txt 
# Audio         - numpy
# Bag of mouth  - numpy
# ConvNet       - matlab probs must be under ['prob_values']
# ConvNet+Audio - matlab probs must be under ['prob_values']
# The function automatically adapts to numpy or matlab,
# files names must have .npy, .txt or .mat extension
def make_weighted_prediction(weightFile, *modelFiles):
    weights = np.load(weightFile)
    model_preds = []
    for i, path in enumerate(modelFiles):
        if path.split(".")[-1]=="npy":
            preds = np.load(path)
            if np.prod(preds.shape) == 7:
                preds = preds.reshape(1,7)
            model_preds.append(preds)
        elif path.split(".")[-1]=="mat":
            preds = sio.loadmat(path)['prob_values_test']
            if np.prod(preds.shape) == 7:
                preds = preds.reshape(1,7)
            model_preds.append(preds)
        elif path.split(".")[-1]=="txt":
            txt_file = open(path,'r')
            preds = []
            for line in txt_file.readlines():
                preds.append([float(prob) for prob in line.strip().split(" ")])
            model_preds.append(np.array(preds))
        else:
            raise ValueError("The file must be .npy (numpy) or .mat (matlab)")

    print model_preds
    preds = np.zeros(model_preds[i].shape)
    for i in xrange(len(model_preds)):
        preds += weights[i]*model_preds[i]

    print preds
    return preds

def main():
    parser = argparse.ArgumentParser(prog="WEIGHTED_AVERAGE",usage="%(prog)s Weights Activity Audio BagOfMouth ConvNet ConNet+Audio OUTPUT_FILE")
    parser.add_argument("files",nargs=7,help="There should be Weights Activity Audio BagOfMouth ConvNet ConNet+Audio Output files in this exact order")
    options = parser.parse_args()

    files = options.files
    output = make_weighted_prediction(*files[:-1])
    np.save(files[-1],output)

if __name__=="__main__":
    main()
