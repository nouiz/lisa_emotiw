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

from faceimages import FaceImagesDataset
import numpy as np

class AFW(FaceImagesDataset):
    """Annotated Face in-the-Wild (from Ramanan, images obtained from flickr)"""

    def __init__(self):
        super(AFW,self).__init__("AFW", "faces/AFW/testimages/")
        self.lstImages= []             
        self.lstkeypoints = []
        import h5py
        f = h5py.File(self.absolute_base_directory+'anno.mat', 'r')
        entryCount = f["anno"].shape[1]

        lstImages= []
        for imIndex in xrange(entryCount):
            
            lstfacebb= [];lstPose=[];lstkeypoints=[]
            
            ref = f["anno"].value[0].item(imIndex)
            res =[ chr(c) for c in f[ref].value]
            name = "".join(res)
            
            ref2 = f["anno"].value[1].item(imIndex)
            count2= f[ref2].value.shape[0]
            for i in xrange(count2):
                lstfacebb.append( np.reshape(f[f[ref2].value[i][0]].value,[1,4]))   #(x1,y1,x2,y2)  

            ref3 = f["anno"].value[2].item(imIndex)
            count3= f[ref3].value.shape[0]
            for j in xrange(count3):
                poseArray = f[f[ref3].value[j][0]].value
                lstPose.append([p[0] for p in poseArray]) #(_,_,_)
                    
            ref4 = f["anno"].value[3].item(imIndex)
            count4= f[ref4].value.shape[0]
            for k in xrange(count4):
                keypoints = f[f[ref4].value[k][0]].value
                valid_kp = [kp[~np.isnan(kp)] for kp in keypoints]
                # [[x1 x2 ..],[y1 y2 ...]]
                self.lstkeypoints.append(np.asarray(valid_kp))

            for c in xrange(len(lstfacebb)):
                self.lstImages.append([name,lstfacebb[c],lstPose[c],self.lstkeypoints[c]])
        
            
    def __len__(self):
        return len(self.lstImages)        
        
    def get_original_image_path_relative_to_base_directory(self, i):       
        return self.lstImages[i][0]    
        
    def get_bbox(self, i):
        return self.lstImages[i][1]

    def get_eyes_location(self, i):
        print '...'
        print self.lstkeypoints[i]
        print 'OK'
        return [self.lstkeypoints[i][0][0],self.lstkeypoints[i][1][0],self.lstkeypoints[i][0][1],self.lstkeypoints[i][1][1]]

    def get_keypoints_location(self,i):
        """
        contains 6 landmarks. [(the center of eyes, tip of nose, the two corners and center of mouth)]
        """
        landmarks = ['left_eye_center', 'right_eye_center', 'nose_tip', 'mouth_left_corner', 'mouse_right_corner', 'mouth_center']
        keypoints = dict(enumerate(zip(*self.lstkeypoints[i])))
        return dict((landmarks[k], v) for (k, v) in keypoints.items())
