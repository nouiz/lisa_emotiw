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
    Note: sibjects id starts from 39341 to 65384
    """
    
    def __init__(self):
        super(AFLW,self).__init__("AFLW", "faces/AFLW/")
        import sqlite3
        self.conn = sqlite3.connect(self.absolute_base_directory+'aflw/data/aflw.sqlite')
        
    def __len__(self):
        
        return self.conn.execute('select count(face_id) from Faces').fetchall()[0][0]
    
    def get_eyes_location(self, i):
        
        res = self.conn.execute('select x,y,feature_id from FeatureCoords where face_id ='+str(i)+' and( feature_id = 11 or feature_id =8) order by feature_id').fetchall()
        if len(res) ==2 :       
            return  [res[0][0:2],res[1][0:2]]
        elif len(res)==1:
            if res[0][2]==8:
                res.append((None,None))
                return res
            else:
                w= [(None,None)]
                w.append(res)
                return w
        else:
            return[(None,None),(None,None)]
            
    def get_id_of_kth_subject(self,i):
        return 39341+i
        

    def get_keypoints_location(self,i):
    
        res = self.conn.execute('select FeatureCoords.x ,FeatureCoords.y,descr  from FeatureCoords,FeatureCoordTypes where face_id ='+str(i)+' and FeatureCoordTypes.feature_id =FeatureCoords.feature_id').fetchall()
        return np.array(res)
    
    def get_n_subjects(self):
        return self.__len__()
                
    def get_original_image_path_relative_to_base_directory(self, i):    
        return self.absolute_base_directory +"aflw/Images/aflw/data/flickr/"+ self.conn.execute('select file_id from Faces where face_id ='+str(i)).fetchall()[0][0]
        
        
        
