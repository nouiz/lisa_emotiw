import os
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",nargs=1)
    parser.add_argument("-y",nargs=1)
    parser.add_argument("data_paths", nargs='+')
    options = parser.parse_args()

    out = options.out[0]
    out = os.path.splitext(out)[0]
     
    Xs = []
    for data_path in options.data_paths:
        data = np.load(data_path)
        Xs.append(data)
        print Xs[-1].shape

    Xs = np.concatenate(Xs,axis=1)
    print Xs.shape

    # min becomes 0
    Xs -= Xs.min(0)
    # range is [0,1]
    Xs /= Xs.max(0)
    # range is [-1,1]
    Xs = Xs*2.0-1.0

    print Xs.min(0)
    print Xs.mean(0)
    print Xs.max(0)

    y = np.load(options.y[0])

    print options

    np.save(out+"_x.npy",Xs)
    np.save(out+"_y.npy",y)

if __name__ == "__main__":
    main()
