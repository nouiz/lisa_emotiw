# Copyright (c) 2013 University of Montreal, Hani Almousli, Pascal Vincent
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The names of the authors and contributors to this software may not be
#       used to endorse or promote products derived from this software without
#       specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os.path
import numpy as np
from faceimages import FaceImagesDataset

class AFLW(FaceImagesDataset):
    """
    Annotated Facial Landmarks in the Wild dataset
    Note: subjects id starts from 39341 to 65384
    """
    
    def __init__(self):
        super(AFLW,self).__init__("AFLW", "faces/AFLW/")
        import sqlite3
        self.conn = sqlite3.connect(self.absolute_base_directory+'aflw/data/aflw.sqlite')
        self.valid_id = self.conn.execute('select distinct face_id from FeatureCoords').fetchall()
        
    def __len__(self):
        
        return self.conn.execute('select count(face_id) from Faces').fetchall()[0][0]
    
    def get_eyes_location(self, i):
        
        res = self.conn.execute('select x,y,feature_id from FeatureCoords where face_id ='+str(i)+' and( feature_id = 11 or feature_id =8) order by feature_id').fetchall()
        if len(res) ==2 :       
            return  [res[0][0], res[0][1],res[1][0], res[1][1]]
        elif len(res)==1:
            if res[0][2]==8:
                eye = [res[0][0], res[0][1]]
                eye.extend([None,None])
                return eye
            else:
                w= [None,None]
                w.extend([res[1][0], res[1][1]])
                print 'w: ', w
                return w
        else:
            return[None,None,None,None]
            
    def get_id_of_kth_subject(self,i):
        return self.valid_id[i][0]
        

    def get_keypoints_location(self,i):
        translation_dict = {'LeftEyeRightCorner': 'right_eye_inner_corner', 'RightBrowCenter': 'left_eyebrow_center', 
                            'LeftEyeLeftCorner': 'right_eye_outer_corner', 
                            'RightBrowLeftCorner': 'left_eyebrow_inner_end', 'NoseLeft': 'right_nostril', 
                            'NoseCenter': 'nose_tip', 'MouthCenter': 'mouth_center', 
                            'LeftEyeCenter': 'right_eye_pupil', 'RightEyeCenter': 'left_eye_pupil', 
                            'LeftEar': 'right_ear_bottom', 'RightEyeRightCorner': 'left_eye_outer_corner', 
                            'RightEyeLeftCorner': 'left_eye_inner_corner', 'RightEar': 'left_ear_bottom', 
                            'NoseRight': 'left_nostril', 'MouthLeftCorner': 'mouth_right_corner', 
                            'ChinCenter': 'chin_center', 'RightBrowRightCorner': 'left_eyebrow_outer_end', 
                            'MouthRightCorner': 'mouth_left_corner', 'LeftBrowLeftCorner': 'right_eyebrow_outer_end', 
                            'LeftBrowCenter': 'right_eyebrow_center', 'LeftBrowRightCorner': 'right_eyebrow_inner_end'}
        
        dic = {}
        res = self.conn.execute('select FeatureCoords.x ,FeatureCoords.y,descr  from FeatureCoords,FeatureCoordTypes where face_id ='+str(self.valid_id[i][0])+' and FeatureCoordTypes.feature_id =FeatureCoords.feature_id order by FeatureCoords.feature_id').fetchall()
        for t in res:
            dic[translation_dict[str(t[2])]] = (t[0], t[1])
        return [dic]
    
    def get_n_subjects(self):
        return self.__len__()
                
    def get_original_image_path_relative_to_base_directory(self, i):    
        return self.absolute_base_directory +"aflw/Images/aflw/data/flickr/"+ self.conn.execute('select file_id from Faces where face_id ='+str(self.valid_id[i][0])).fetchall()[0][0]
        

def testWorks():
    save = 1
    import pickle
    if (save):
        obj = AFLW()
    #    output = open('aflw.pkl', 'wb')
    #    data = obj
    #    pickle.dump(data, output)
    #    output.close()
    else:
        pkl_file = open('aflw.pkl', 'rb')
        obj = pickle.load(pkl_file)
        pkl_file.close()

    obj.verify_samples()

if __name__ == '__main__':
    testWorks()
        
        
