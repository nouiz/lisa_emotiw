#!/usr/bin/env python

import numpy as np
import sys
import os
from submission import create_submission_file

classnames = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

usage_msg = """

** emotiw.py usage **

This script handles operations on an .npy matrix that contains the predicted scores for the 7 classes from either
all train, all valid, or all test clips.
The rows of the .npy matrix must correspond to all clips in the same order as specified in one of the
   /data/lisatmp2/EmotiWTest/ModelPredictionsToCombine/afew2_*_filelist.txt
It must contain 7 columns giving scores for the classes in that order:
   Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

The script can be evoked using one of the following 3 subcommands:

emotiw.py show score_matrix.npy
   Will print the true label, winning class and scores of the clips
   For the test set it does not print the true label (we don't have it), only the winning class

emotiw.py eval score_matrix.npy  
   Will compute and output the accuracy for the given 7 class score matrix (and associated labels)
   If instead of a score_matrix.npy you specify a directory, it will walk through the directory
   in searh for model prediciton files of the form and report all associated accuracies

emotiw.py submit score_matrix.npy 
   will output a file named score_matrix.txt that will contain the same content as shown by the show command
   as well as a score_matrix_submission.zip which corresponds to the organiser's submission format for those same results

All 3 subcommands can take an optional start_column last argument (that defaults to 0) which indicates the
starting column of the 7 class socres.
"""


def load_file_list(txt_file_path):
    with open(txt_file_path, 'r') as infile:
        return [ line.strip().split()[1] for line in infile if line.strip()!="" ]

train_labels = np.load('/data/lisatmp2/EmotiWTest/ModelPredictionsToCombine/afew2_train_targets.npy').astype(int)
valid_labels = np.load('/data/lisatmp2/EmotiWTest/ModelPredictionsToCombine/afew2_valid_targets.npy').astype(int)
train_filelist = load_file_list('/data/lisatmp2/EmotiWTest/ModelPredictionsToCombine/afew2_train_filelist.txt')
valid_filelist = load_file_list('/data/lisatmp2/EmotiWTest/ModelPredictionsToCombine/afew2_valid_filelist.txt')
test_filelist = load_file_list('/data/lisatmp2/EmotiWTest/ModelPredictionsToCombine/afew2_test_filelist.txt')
ntrain = len(train_filelist)
nvalid = len(valid_filelist)
ntest = len(test_filelist)

# print "# training clips:",ntrain
# print "# valid clips:",nvalid
# print "# test clips:",ntest

assert len(train_labels)==ntrain
assert len(valid_labels)==nvalid

filelist_and_labelvec = { "train": (train_filelist, train_labels),
                       "valid": (valid_filelist, valid_labels),
                       "test": (test_filelist, None) }

def get_associated_dataset_name(scoremat):
    """Takes a sinput either a numpy ndarray, or a filename.npy,
    and returns the name of the dataset (train, valid or test) associated to the number of rows of that matrix"""
    if isinstance(scoremat,basestring):
        n = len(np.load(scoremat))
    else:
        n = len(scoremat)

    if n==ntrain:
        return "train"
    elif n==nvalid:
        return "valid"
    elif n==ntest:
        return "test"
    else:
        raise ValueError("Score matrix length (%d) corresponds to neither length of train (%d) valid (%d) or test (%d) sets" % (n,ntrain,nvalid,ntest))

    
def print_usage_and_exit(msg=""):
    print
    print "***** ",msg
    print usage_msg
    sys.exit()


def print_scores(scorefile, score_start_column=0, out=sys.stdout):
    scoremat = np.load(scorefile)
    scoremat = scoremat[:,score_start_column:(score_start_column+7)]
    dataset_name = get_associated_dataset_name(scoremat)
    filelist, labelvec = filelist_and_labelvec[dataset_name]
    for pos,scores in enumerate(scoremat):
        winner = classnames[scores.argmax()]
        filename = filelist[pos]
        s0,s1,s2,s3,s4,s5,s6 = scores
        if labelvec is not None:
            out.write("%s \t %10s %10s \t" %(filename, classnames[labelvec[pos]], winner) + "%1.4f %1.4f %1.4f %1.4f %1.4f %1.4f %1.4f" % tuple(scores) + "\n")
        else:
            out.write("%s \t %10s \t" %(filename, winner) + "%1.4f %1.4f %1.4f %1.4f %1.4f %1.4f %1.4f" % tuple(scores) + "\n")

def compute_accuracy(scorefile, score_start_column=0):
    scoremat = np.load(scorefile)
    scoremat = scoremat[:,score_start_column:(score_start_column+7)]
    dataset_name = get_associated_dataset_name(scoremat)
    filelist, labelvec = filelist_and_labelvec[dataset_name]
    if labelvec is None:
        print """You provided a score matrix %s that corresponds to the test set, for which
        we don't have labels, so I can't compute accuracy. Maybe you want to try the show subcommand (instead of eval).""" % scorefile
        sys.exit()
    winners = scoremat.argmax(axis=1)
    accuracy = (winners==labelvec).mean()
    return accuracy

def submit(scorefile, score_start_column=0):
    scoremat = np.load(scorefile)
    scoremat = scoremat[:,score_start_column:(score_start_column+7)]
    dataset_name = get_associated_dataset_name(scoremat)
    if dataset_name!="test":
        print "ABORTING: The score file you want to submit does not seem to correspond to a performance on the test set " \
              "since it contains %d rows, whereas the test set has %d clips" % (len(scoremat),len(test_filelist)) 
        return
    
    basename,ext = os.path.splitext(scorefile)
    txtfile = basename+".txt"
    zipfile = basename+"_submit.zip"

    print "Writing file", txtfile
    with open(txtfile, 'w') as out:
        print_scores(scorefile, score_start_column, out)

    print "Packaging its content into submission format: ", zipfile
    create_submission_file(txtfile, zipfile)
    
def main(argv):
    # print argv
    if len(argv)==3:
        score_start_column = 0
    elif len(argv)==4:
        score_start_column = eval(argv[-1])
    else:
        print_usage_and_exit("Wrong number of arguments")

    command = argv[1]    
    matrix_filepath = argv[2]
    if command=='show':
        print_scores(matrix_filepath,score_start_column)
    elif command=='eval':
        accu = compute_accuracy(matrix_filepath,score_start_column)
        print "%2.2f%% accuracy on %s" % (accu*100., get_associated_dataset_name(matrix_filepath))
    elif command=='submit':
        submit(matrix_filepath,score_start_column)
    else:
        print_usage_and_exit("Invalid first argument")
        

if __name__ == '__main__':    
    main(sys.argv)
