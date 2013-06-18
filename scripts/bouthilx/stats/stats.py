from emotiw.bouthilx import timer

import os
import glob
import numpy as np
import PIL.Image as Image

DIR = "/data/lisa/data/faces/EmotiW/images/"
sets = ["Train","Val"]
labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

# load one sequence of .jpg and save as .npy
def load_set(s,t=None,process=None):
    """
        TODO? Load face tubes with afew2 rather than directly loading .jpg files
    """

    sequences = []
    targets = []
    if process is None:
        process = lambda a: np.array(a)
    for label in labels:
        paths = glob.glob(os.path.join(DIR,s,label,"*.png"))
        paths = sorted(paths)
        
        sequence = None
        j = 0
        for i, path in enumerate(paths):
            if path[-7:-4] == "001" and sequence is not None:
                sequences.append( process(sequence/(0.+j)) )
                target = np.zeros(len(labels))
                target[labels.index(label)] = 1.0
                targets.append(target)
                sequence = None

            if sequence is None:
                sequence = np.asarray(Image.open(path).convert('RGB'),int)
            else:
                sequence += np.asarray(Image.open(path).convert('RGB'),int)

            if t is not None:
                t.print_update(1)

            j = int(path[-7:-4])

    return np.array(sequences), np.array(targets)

def process(mean):
    """
        mean saturation 
        mean illumination
        mean red
        mean green
        mean blue
    """

    # RGB
    m = np.min(mean,2).T
    M = np.max(mean,2).T
    C = M-m
    Cmsk = C!=0
    I = M 
    S = np.zeros(mean.T[0].shape)
    S[Cmsk] = ((255*C)/I)[Cmsk]
    R = mean[:,:,0].mean()
    G = mean[:,:,1].mean()
    B = mean[:,:,2].mean()

    return np.array([S.mean(),I.mean(),R,G,B])

if __name__ == "__main__":

    t = timer.Timer(len(sets)*len(labels)*90*762/len(sets)/len(labels),min_time=1)
    t.start()
    for set_path in sets:
        x, y = load_set(set_path,t=t,process=process)
        np.save(set_path+"_X.npy",x)
        np.save(set_path+"_y.npy",y)
        print x.shape
        print y.shape
        print x
        print y
    t.over()
