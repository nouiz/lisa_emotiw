import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",nargs=1)
    parser.add_argument("-y",nargs=1)
    parser.add_argument("data_paths", nargs='+')
    options = parser.parse_args()

    out = options.out[0]
    out = out.strip('.npy')
     
    Xs = []
    for data_path in options.data_paths:
        Xs.append(np.load(data_path))
        print Xs[-1].shape

    Xs = np.concatenate(Xs,axis=1)
    print Xs.shape

    y = np.load(options.y[0])

    print options

    np.save(out+"_X.npy",Xs)
    np.save(out+"_y.npy",y)

if __name__ == "__main__":
    main()
