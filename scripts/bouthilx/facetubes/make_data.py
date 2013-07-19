import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",nargs=1)
    parser.add_argument("model_path", nargs=1)
    options = parser.parse_args()

    out = options.out[0]

    d = afew2.AFEW2ImageSequenceDataset()#preload_facetubes=True)
    for i, [clip, info, target] in enumerate(zip(d.imagesequences,d.seq_info,d.labels)):
        
        for facetube in d.get_facetubes(i):
            onehot = np.zeros(len(d.emotionNames.keys()))
            onehot[d.emotionNames.values().index(target)] = 1.0
            targets[info[0]] += [onehot]*facetube.shape[0]
            faces[info[0]] += [face for face in facetube]
 
    np.save(out+"_X.npy",X)
    np.save(out+"_y.npy",y)

if __name__ == "__main__":
    main()
