<<<<<<< HEAD
import os
=======
>>>>>>> 37bef88e556862d180581c89fe7b7b5017036991
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",nargs=1)
    parser.add_argument("-y",nargs=1)
    parser.add_argument("data_paths", nargs='+')
    options = parser.parse_args()

    out = options.out[0]
<<<<<<< HEAD
    out = os.path.splitext(out)[0]
     
    Xs = []
    for data_path in options.data_paths:
        data = np.load(data_path)
        Xs.append(data)
=======
    out = out.strip('.npy')
     
    Xs = []
    for data_path in options.data_paths:
        Xs.append(np.load(data_path))
>>>>>>> 37bef88e556862d180581c89fe7b7b5017036991
        print Xs[-1].shape

    Xs = np.concatenate(Xs,axis=1)
    print Xs.shape

<<<<<<< HEAD
    # min becomes 0
    Xs -= Xs.min(0)
    # range is [0,1]
    Xs /= Xs.max(0)
    # range is [-1,1]
    Xs = Xs*2.0-1.0

    print Xs.min(0)
    print Xs.mean(0)
    print Xs.max(0)

=======
>>>>>>> 37bef88e556862d180581c89fe7b7b5017036991
    y = np.load(options.y[0])

    print options

<<<<<<< HEAD
    np.save(out+"_x.npy",Xs)
=======
    np.save(out+"_X.npy",Xs)
>>>>>>> 37bef88e556862d180581c89fe7b7b5017036991
    np.save(out+"_y.npy",y)

if __name__ == "__main__":
    main()
