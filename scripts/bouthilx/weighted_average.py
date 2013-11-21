import numpy as np
import argparse
from scipy import io as sio

# Order of the modelFiles
# Activity      - numpy
# Audio         - numpy
# Bag of mouth  - numpy
# ConvNet       - matlab probs must be under ['prob_values']
# ConvNet+Audio - matlab probs must be under ['prob_values']
# The function automatically adapts to numpy or matlab,
# files names must have .npy or .mat extension
def make_weighted_prediction(weightFile, *modelFiles):
    weights = np.load(weightFile)
    preds = np.zeros(np.load(modelFiles[0]).shape)
    for i, f in enumerate(modelFiles):
        if f.split(".")[-1]=="npy":
            x = np.load(f)
        elif f.split(".")[-1]=="mat":
            x  = sio.loadmat(f)['prob_values']
        else:
            raise ValueError("The file must be .npy (numpy) or .mat (matlab)")
        preds += weights[i]*x

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
