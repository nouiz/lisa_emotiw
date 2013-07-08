#class to do basic face alignment based on keypoints
#by Yoshua
#coded by Abhi (abhiggarwal@gmail.com)

import os
import theano.tensor as T
import theano
import numpy
import gzip
import scipy
from scipy import io as sio
import Image
import math


class faceAlign(object):
   def __init__(self, datasetObjects, keypoint_dictionary, face_size = (256, 256), margin = 0.3):
   
      self.margin = margin
      self.face_size = face_size
      self.numKeypoints = len(keypoint_dictionary)
      self.keypoint_dictionary = keypoint_dictionary
      
      #calculate mean for all datasets
      meanX = numpy.zeros((len(datasetObjects),len(keypoint_dictionary)))
      meanY = numpy.zeros((len(datasetObjects),len(keypoint_dictionary)))
      num_each_keypoint = numpy.zeros((len(datasetObjects),len(keypoint_dictionary)))
      for i in range(len(datasetObjects)):
          dataset = datasetObjects[i]
          indexLocs = range(len(dataset))
          (mX, mY, num) = self.calculate_mean(dataset, indexLocs, keypoint_dictionary, margin)
          print 'Mean for dataset:',i,'calculated'
          meanX[i,:] = mX
          meanY[i,:] = mY
          num_each_keypoint[i, :] = num

      self.meanX = numpy.sum(meanX * num_each_keypoint, axis = 0)/numpy.sum(num_each_keypoint, axis = 0)
      self.meanY = numpy.sum(meanY * num_each_keypoint, axis = 0)/numpy.sum(num_each_keypoint, axis = 0)
      self.meanX[numpy.sum(num_each_keypoint, axis =0) == 0] = 0
      self.meanY[numpy.sum(num_each_keypoint, axis =0) == 0] = 0
      
      
      self.mu = theano.shared(numpy.zeros((2*self.numKeypoints, 1)))
      mu = numpy.zeros((2*self.numKeypoints, 1))
      mu[0::2,0] = self.meanX[:]
      mu[1::2,0] = self.meanY[:]
      
      self.mu.set_value(mu)
      self.setup_theano()




   def setup_theano(self):
      #for numpy optimization
      oneCol = T.col('oneCol')
      pi_t = T.col('pi_t')
      z_t = T.col('z_t')
      z_t1 = z_t.reshape((self.numKeypoints, 2))
      pts = T.concatenate((z_t1, oneCol), axis=1)
      A_t_ = T.matrix('A_t_')
      r_t_ = T.dot(A_t_, pts.transpose()).transpose()
      r_t1_ = r_t_[:,0:2].reshape((2*self.numKeypoints,1))

      diff_ = pi_t * (r_t1_ - self.mu)
      difft_ = diff_.reshape((1, 2 * self.numKeypoints))
      
      cost_1 = T.dot(difft_,diff_)
      #cost_1 = theano.printing.Print('cost is:')(cost_1)
      cost_ = T.max(cost_1)
   
      A_t_grad_ = T.grad(cost=cost_, wrt=A_t_)
      A_t_grad_ = T.basic.set_subtensor(A_t_grad_[2,:],0)
      self.cost = theano.function(inputs=[A_t_, pi_t, z_t, oneCol],
                                  outputs=[cost_, A_t_grad_])

   def apply_sequence(self, dataset, window = 2, returnA_t = False, perturb = False, flip=False):
      numFrame = len(dataset)
      mats = []
      mean = 0
      meanNum = 0
      #calculate the mean A_t for the clip
      for i in range(numFrame):
         mats.append(self.apply(dataset, i, returnA_t=True))
         mean += mats[i]
         meanNum += 1
      mean = mean/meanNum


      #calculate the mean A_t per frame given the window size and magic number
      transMats = []
      magicNum = []
      for frame in range(numFrame):
         transMat = 0
         num = 0
         for i in range(max(0, frame - window), min(frame+window+1, numFrame)):
               transMat += mats[i]
               num += 1
         transMats.append(transMat/num)
         magicNum.append(numpy.sqrt(numpy.sum(numpy.square(transMats[frame] - mean))/9.0))
      
      #calculate weighted mean A_t given the window size
      transMats_final = []
      images = []

      if perturb:
         pMat = self.get_perturb_Mat(5,5,True)

      for frame in range(numFrame) :
         matrx = 0
         sumWeights = 0
         for i in range(max(0, frame - window), min(frame+window+1, numFrame)):
            weight = magicNum[frame]/numpy.sqrt(numpy.sum(numpy.square(transMats[frame] - mats[i]))/9.0)
            sumWeights += weight
            matrx += weight * mats[i]

         if perturb:
            b4BasePertrb = matrx/sumWeights
            aftrBasePertrb = numpy.dot(pMat, b4BasePertrb)
            frmPMat = self.get_perturb_Mat()
            transMats_final.append(numpy.dot(frmPMat, aftrBasePertrb))
         else:
            transMats_final.append(matrx/sumWeights)

         if not returnA_t:
            face_org = self.get_face_pil_image(dataset, frame)  
            image = self.get_transformed_image(face_org, transMats_final[frame])
            if flip:
               image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if perturb:
               string = 'perturb_'
            else:
               string = 'org_'

            image.show(string+str(frame))
            images.append(image)

      #returns result
      if returnA_t:
         return transMats_final
      else:
         return images
            
   def get_perturb_Mat(self, maxTrans = 3, maxRotDegree = 2, scaling=False):
      Mat = numpy.zeros((3,3))
      Mat[2,:]=[0,0,1]
          
      #tranlation
      transMat = Mat.copy()
      transMat[0,2] = maxTrans * (numpy.random.random((1))-0.5)
      transMat[1,2] = maxTrans * (numpy.random.random((1))-0.5)
      transMat[2,2] = 0.0

      #rotation
      rotMat = Mat.copy()
      theta = (numpy.random.random((1))-0.5) * 3.14159265359*(maxRotDegree/180.0)
      rotMat[0:2,0:2] = [[math.cos(theta), math.sin(theta)],[-1 * math.sin(theta), math.cos(theta)]]

      #scaling
      if scaling:
         scaleMat = Mat.copy()
         scaleFactor = 1 + numpy.random.random((1)) * 0.1 
         scaleMat[0,0] = scaleMat[1,1] = scaleFactor
         perturbMat = numpy.dot(rotMat, numpy.dot(scaleMat,(numpy.eye(3)+transMat)))
      else:
         perturbMat = numpy.dot(rotMat,numpy.eye(3)+transMat)

      return perturbMat


   def apply(self, dataset, index, returnA_t = False,  perturb = False, flip = False):
       oneCol = numpy.ones((self.numKeypoints, 1))
       temp = numpy.random.random((9))
       temp[6:9] = [0,0,1]
   
       def cost_(A_t_flat, pi_t, z_t, oneCol, self):
           A_t = A_t_flat.reshape((3,3))
           cost, A_t_grad = self.cost(A_t, pi_t, z_t, oneCol)
           #print cost
           return [cost, A_t_grad.reshape((9))]

       #getting keypoints
       keypoints = self.get_keypoints(dataset, index)
       z_t = numpy.zeros((2*self.numKeypoints,1))
       pi_t = numpy.zeros((2*self.numKeypoints,1))
       for key in keypoints:
           i = self.keypoint_dictionary[key]
           (x,y) = keypoints[key]
           z_t[2*i] = x
           z_t[2*i+1] = y
           pi_t[2*i] = 1.0
           pi_t[2*i+1] = 1.0
       
       #calculate transformation matrix
       A_t, cost, uselessDict  = scipy.optimize.fmin_l_bfgs_b(func=cost_, x0=temp , fprime=None, args=(pi_t, z_t, oneCol, self))
       A_t = A_t.reshape((3,3))
       
       if perturb:
          A_t = numpy.dot(self.get_perturb_Mat(5,5,True), A_t)

       if returnA_t:
          return A_t

       face_org = self.get_face_pil_image(dataset, index)
          #face_org.show()
       trans_face = self.get_transformed_image(face_org, A_t)

       if  flip:
          return trans_face.transpose(Image.FLIP_LEFT_RIGHT)
       else:
          return trans_face

   
   def draw_template(self, image):
       pixmap = image.load()
       for i in xrange(len(self.meanX)):
          (x,y)= self.meanX[i],self.meanY[i]
          pixmap[int(x),int(y)] = (0,200,0)
       return image


   def get_transformed_image(self, pil_image, A_t):
       pixmap = pil_image.load()
       imgT = Image.new('RGB', pil_image.size, 'black')
       pixmapT = imgT.load()
       A_t_inv = numpy.linalg.inv(A_t)
       width, height = pil_image.size
       indices = numpy.ones((3, width,height))
       tempX  = numpy.asarray(range(width)).reshape((1,width))
       tempY  = numpy.asarray(range(height)).reshape((height,1))
       indices[0,:,:] = indices[0,:,:] * tempX
       indices[1,:,:] = indices[1,:,:] * tempY
       indicesScaled = numpy.copy(indices)
       #indicesScaled[0,:,:] = (indicesScaled[0,:,:])/width                       
       #indicesScaled[1,:,:] = (indicesScaled[1,:,:])/height
       inp =  numpy.dot(A_t_inv, indicesScaled.reshape((3, width*height)) ).reshape((3,width,height))
       inpValid = numpy.logical_and(inp >= 0.0, inp <= 1.0 )
       #inp[0,:,:] = (inp[0,:,:] * width)
       #inp[1,:,:] = (inp[1,:,:] * height)
       #inp = inp * inpValid
       inp = numpy.round(inp[0:2, :,:].reshape((2, width*height)).transpose())
       indices = numpy.round(indices[0:2, :,:].reshape((2, width*height)).transpose())
       for i in range(width*height):
           (x,y) = (inp[i,0], inp[i,1])
           (x_, y_) = (indices[i,0], indices[i,1])
           if x > 0 and y > 0:
               if x < width and y < height:
                   pixmapT[x_, y_] = pixmap[x,y]
             
       return imgT           

   def get_keypoints_based_bbox(self, dataset, index):
       keypoints = dataset.get_keypoints_location(index).copy()
       x0 = 10000
       x1 = 0
       y0 = 10000
       y1 = 0
       for key in keypoints:
           (x,y) = keypoints[key]
           if ( x < x0):
               x0 = x
           if ( y < y0):
               y0 = y
           if ( x > x1):
               x1 = x
           if ( y > y1):
               y1 = y
       return [x0, y0, x1, y1]

   def get_keypoints(self, dataset, index):
       keypoints = dataset.get_keypoints_location(index)
       if keypoints == None:
          return None
       else:
          keypoints = keypoints.copy()

       bbox = dataset.get_bbox(index)
       if bbox == None:
          bbox = self.get_keypoints_based_bbox(dataset, index)
       else:
          bbox = bbox[0]
       width = bbox[2]-bbox[0]
       height = bbox[3]-bbox[1]
       #new bounding box with margin
       x0, y0 = (bbox[0] - self.margin * width, 
                     bbox[1] - self.margin * height)
       keypoint_dictionary = self.keypoint_dictionary

       for key in keypoint_dictionary:
          if key in keypoints:
               (x, y) = keypoints[key]
               (x_, y_) = (x-x0, y-y0)
               x_ = (x_/((1+ 2 * self.margin)*width)) * self.face_size[0]
               y_ = (y_/((1+ 2 * self.margin)*height)) * self.face_size[1]
               keypoints[key] = (x_, y_)
       return keypoints


   def apply_gcn(self, image):
       image = image.convert('RGB')
       npImg = numpy.array(pic.getdata()).reshape(image.size[0], image.size[1], 3)
       npUnrolled = npImg[:]
       mean = numpy.mean(npUnrolled)
       std = numpy.std(npUnrolled)
       npImage = ((npUnrolled - mean)/std).reshape(image.size[0], image.size[1],3)
       return npImage
      
   

   def calculate_mean(self, dataset, indexLocs, keypointDictionary, margin):
       meanX = numpy.zeros((len(keypointDictionary)))
       meanY = numpy.zeros((len(keypointDictionary)))
       num_each_keypoint = numpy.zeros((len(keypointDictionary)))
       for index in indexLocs:
           keypoints = self.get_keypoints(dataset, index)
           if keypoints == None:
              continue

           for key in keypointDictionary:
               if key in keypoints:
                   (x, y) = keypoints[key]
                   meanX[keypointDictionary[key]] += x
                   meanY[keypointDictionary[key]] += y
                   num_each_keypoint[keypointDictionary[key]] += 1
       
       meanX = meanX/num_each_keypoint
       meanY = meanY/num_each_keypoint
       meanX[num_each_keypoint == 0] = 0
       meanY[num_each_keypoint == 0] = 0
       return (meanX, meanY, num_each_keypoint)     
           
      
   def get_face_pil_image(self, dataset, index):
       imagePath = dataset.get_original_image_path(index)
       if imagePath == None:
          image = dataset.get_original_image(index)
          print image
          image = Image.fromarray(image)
          image = image.convert('RGB')
       else:
          image = Image.open(imagePath)
       (imWidth, imHeight) = image.size
       #original bounding box
       bbox = dataset.get_bbox(index)
       if bbox == None:
          bbox = self.get_keypoints_based_bbox(dataset, index)
       else:
          bbox = bbox[0]

       width = bbox[2]-bbox[0]
       height = bbox[3] - bbox[1]
       margin = self.margin
       #new bounding box with margin
       bbox = (int(bbox[0] - margin * width), 
               int(bbox[1] - margin * height), 
               int(bbox[2] + margin * width), 
               int(bbox[3] + margin * height))

       face_crop = image.crop(bbox)
       
       #return resized face_image
       return face_crop.resize(self.face_size)


def dummy_test():
    import pickle

    from emotiw.common.datasets.faces.faceimages import keypoints_names

    #sequence Datasets
    from emotiw.common.datasets.faces.afew2 import AFEW2ImageSequenceDataset

    #Static Datasets
    from emotiw.common.datasets.faces.multipie import MultiPie
        
    #without keypoints
    from emotiw.common.datasets.faces.tfd import TorontoFaceDataset
    from emotiw.common.datasets.faces.afew import AFEWImageSequence

    
    datasetObjs = []
    datasetObjs = pickle.load(open("datasetObjs.pkl","r"))
    alignObj = pickle.load(open( "AlignObj.pkl", "r" ))
    paths = []
    '''
    datasetObjs.append(TorontoFaceDataset())
    print 'added TorrontoFaceDatset'
    datasetObjs.append(MultiPie())
    print 'added Multipie'
    Afew2 = AFEW2ImageSequenceDataset()
    for i in xrange(len(Afew2)):
       print 'sequence:', i
       datasetObjs.append(Afew2.get_sequence(i))
    print 'added AFEW2'

    pickle.dump( datasetObjs, open( "datasetObjs.pkl", "wb" ))
    return 
    
    keys = {}
    index = 0 
    for key in keypoints_names:
        keys[key] = index
        index += 1
        
    alignObj = faceAlign(datasetObjects = datasetObjs, keypoint_dictionary = keys, face_size = (48, 48), margin = 0.3)
    pickle.dump( alignObj, open( "AlignObj.pkl", "wb" ))

    '''

    keysp1 = datasetObjs[0].get_keypoints_location(46)
    keysp2 = datasetObjs[1].get_keypoints_location(0)

    img = Image.new('RGB', (1024, 576), 'black')
    import ImageDraw
    draw = ImageDraw.Draw(img)
    pixmap = img.load()
   
    print keysp1
    print keysp2
    
    keys = {}
    index = 0 
    for key in keypoints_names:
        keys[key] = index
        index += 1
    
    
    for i in keysp1:
       (x,y) = keysp1[i]
       pixmap[int(x), int(y)] = (200,0,0)
       draw.text((int(x), int(y)), str(keys[i]), (0,0,200))
       
    for i in keysp2:
       (x,y) = keysp2[i]
       pixmap[int(x), int(y)] = (0,200,0)
       pixmap[int(x), int(y)] = (200,0,0)
       draw.text((int(x), int(y)), str(keys[i]), (0,0,200))


    img.show()
    
    
    return
    

    alignObj.draw_template(alignObj.apply(datasetObjs[0], 10, perturb=True)).show('perturbed')
    alignObj.draw_template(alignObj.apply(datasetObjs[1], 10, perturb = False)).show()

    #alignObj.apply(datasetObjs[639], 10, True, True)[1].show('perturbed')
    #alignObj.apply(datasetObjs[639], 10)[1].show()

  #  alignObj.apply_sequence(datasetObjs[485], window = 2,  perturb=False)
  #  alignObj.apply_sequence(datasetObjs[485], window = 2,  perturb=True, flip = True)
#    alignObj.apply(obj, 2500)



if __name__ == '__main__':
    dummy_test()

   
       
