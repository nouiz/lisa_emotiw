'''
Created on 2013-07-09

@author: "JP Raymond (raymonjp@iro)

Processes images using Ramanan's code ("Face Detection, Pose Estimation, and
Landmark Localization in the Wild, Zhu X. and Ramanan D., 2012", available at
http://www.ics.uci.edu/~xzhu/paper/face-cvpr12.pdf).

Requires the mlabwrap library (http://mlabwrap.sourceforge.net/). Also,
Ramanan's MATLAB code (in folder face-release1.0-basic) must be included in
MATLAB search path.
'''

from mlabwrap import mlab

def process_first_file(folder_todo, folder_results, folder_done, model_no=3):
    '''
    Processes the first file from folder_todo, saves the results in
    folder_results and moves the file in folder_done. model_no refers to one of
    the three models available in Ramanan's code :
        "
        1 : pre-trained model with 146 parts. Works best for faces larger than
            80*80.            
        2 : pre-trained model with 99 parts. Works best for faces larger than
            150*150.
        3 : pre-trained model with 1050 parts. Gives best performance on
            localization, but very slow.
        "
    '''
    mlab.demoneimagewhole(folder_todo, folder_results, folder_done, model_no)
