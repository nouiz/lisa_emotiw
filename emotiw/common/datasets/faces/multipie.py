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

import os
import scipy.io

from faceimages import FaceImagesDataset

class MultiPie(FaceImagesDataset):
    
    def __init__(self):
        super(MultiPie,self).__init__("MultiPie", "faces/Multi-Pie/")
        self.lstImages= []
        self.lstgender=[]
                             
        label = self.absolute_base_directory+'MPie_Labels/labels/'

        #read male and female
        metaPath = self.absolute_base_directory+'meta/subject_list.txt'
        f = open(metaPath)
        for line in f:
            if line!='':
                self.lstgender.append(line.split(' ')[2])
        
        cam = os.listdir(label)
        for c in cam:
            files = os.listdir(label+c)
            for f in files:
                imrelpath=self.absolute_base_directory+'data/'
                parts = f.split('.')[0].split('_')
                subject = int(parts[0])
                session= parts[1]
                imrelpath += 'session'+session+'/multiview/'+parts[0]+'/'+parts[2]+'/'+parts[3][0]+parts[3][1]+"_"+parts[3][2]+'/'
                imrelpath += "_".join(parts[0:5])+'.png'
                points = scipy.io.loadmat(label+c+'/'+f)['pts']
                #There are some subjects with out male or female information                
                if subject<len(self.lstgender):
                    self.lstImages.append([imrelpath,subject,points,self.lstgender[subject]])
                else:
                    self.lstImages.append([imrelpath,subject,points,""])
                
    
    
    def __len__(self):
        return len(self.lstImages)        
        
    def get_original_image_path_relative_to_base_directory(self, i):       
        return self.lstImages[i][0]    
        
    
    def get_eyes_location(self, i):
        """
        returns left eye (left corner, right corner), right eye (left corner, right corner)
        """
        return [self.lstImages[i][2][37], self.lstImages[i][2][40],self.lstImages[i][2][43],self.lstImages[i][2][46]]
    
    def get_keypoints_location(self,i):
        """
        Check MPie_Labels/examples/ to see which corresponds to what
        """        
        return self.lstImages[i][2]    
        
    def get_subject_id_of_ith_face(self, i):
        return str(self.lstImages[i][1])
        
    def get_id_of_kth_subject(self, k):    
        return str(self.lstImages[k][1])
        
    def get_gender(self,i):        
        return str(self.lstImages[i][3])
            
