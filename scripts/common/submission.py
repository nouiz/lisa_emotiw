import argparse
import numpy as np
import os
import zipfile

# take a prediction : id winning_class scores

# create dir
# create .txt files for each row
# zip the dir

classes = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",required=True,help="Name of zipped file produced")
    parser.add_argument("--force",action='store_true',help="Force to overwrite the zipped file")
    parser.add_argument("prediction_file",help="Prediction file with rows defined as : id_of_clip  winning_class  Angry_score Disgust_score Fear_score Happy_score Sad_score Surprise_score Neutral_score")
    options = parser.parse_args()
    
    predictions = open(options.prediction_file,'r')

    d = create_tmp_folder()

    try:
        for prediction in predictions:
            p_id, p_win = prediction.split(' ')[:2]
            f = open(os.path.join(d,p_id+'.txt'),'w')
            f.write(p_win)
            f.close()

        zipf = zipfile.ZipFile(options.out,'w')

        for root, dirs, files in os.walk(d):
            cwd = os.getcwd()
            os.chdir(root)
            for f in files:
                zipf.write(f)
            os.chdir(cwd)

        zipf.close()
    finally:
        for root, dirs, files in os.walk(d):
            for f in files:
                os.remove(os.path.join(root,f))

        os.rmdir(d)

def create_tmp_folder():
    d = "tmp_"+str(np.random.random_integers(0, 100000))

    while os.path.isdir(d):
        d = "tmp_"+str(np.random.random_integers(0, 100000))
    
    os.mkdir(d)

    return d

def create_test():
    s = np.random.multinomial(20,[1/(0.+len(classes))]*len(classes),size=50)/20.
    test = open('test.txt','w')
    for i, row in enumerate(s):
        test.write("%(id)08d %(win)s %(scores)s\n" % {'id':i, 'win': classes[row.argmax()], 
            'scores':" ".join([str(item) for item in row])})

    test.close()

if __name__=="__main__":
    main()
    #create_test()
