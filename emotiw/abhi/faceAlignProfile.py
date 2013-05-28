#class to do basic face alignment based on keypoints
#by Yoshua
#coded by Abhi (abhiggarwal@gmail.com)

import os
import theano.tensor as T
import theano
import numpy
import gzip
import pickle
import scipy
from scipy import io as sio
import Image

def cost_(A_t_flat, pi_t, z_t, oneCol,  self):
   A_t = A_t_flat.reshape((3,3))
   cost, A_t_grad = self.cost(A_t, pi_t, z_t, oneCol)
   return [cost, A_t_grad.reshape((9))]

class faceAlign(object):
   def __init__(self, datasetPaths = None, keyPoints = None):
      
      self.xMax = 1024.0
      self.yMax = 576.0
      self.numKeyPoints = 39
      self.loadPicasaTubePickle()
      loadPrev = 1

      if loadPrev == 1:
         pkl_file = open('faceAlign2.pkl', 'rb')
         self.pose, self.landmarks, self.poseDict, self.images, self.poseCat = pickle.load(pkl_file)
         pkl_file.close()
      else:
         self.loadData()
         output = open('faceAlign2.pkl', 'wb')
         data = (self.pose, self.landmarks, self.poseDict, self.images, self.poseCat)
         pickle.dump(data, output)
         output.close()
         return 

      self.eeta = 0.0000001
      self.mu = theano.shared(10 * numpy.random.random((2*self.numKeyPoints, 1)))
      self.S = theano.shared(numpy.eye(2 * self.numKeyPoints))
      self.alpha = theano.shared(0.1 * numpy.ones((2 * self.numKeyPoints,1)))
      theano.config.compute_test_value = 'warn'
      oneCol = T.col('oneCol')
      oneCol.tag.test_value = numpy.ones((self.numKeyPoints,1))
      pi_t = T.col('pi_t')
      pi_t.tag.test_value = numpy.random.random((2*self.numKeyPoints,1))
      temp = numpy.random.random((3,3))
      #temp = numpy.zeros((3,3))
      temp[2,:] = [0,0,1]
      self.A_t = theano.shared(temp, name='A_t')
      #print_A_t = theano.printing.Print('r_t1')(A_t)
      z_t = T.col('z_t')
      z_t.tag.test_value = numpy.random.random((2*self.numKeyPoints,1))
      z_t1 = z_t.reshape((self.numKeyPoints, 2))

      pts = T.concatenate((z_t1, oneCol), axis=1)
#      pts = theano.printing.Print('pts')(pts)
      r_t = T.dot(self.A_t, pts.transpose()).transpose()
      r_t1 = r_t[:,0:2].reshape((2*self.numKeyPoints,1))
      #pi_tt = theano.printing.Print('pi_t before')(pi_t)
      diff = pi_t * (r_t1 - self.mu)
      difft = diff.reshape((1, 2 * self.numKeyPoints))
      #diff = theano.printing.Print('diff:')(diff)
      cost = T.max(T.dot(T.dot(difft,self.S),diff))
      #cost = theano.printing.Print('cost:')(cost)
      A_t_grad = T.grad(cost=cost, wrt=self.A_t)
      A_t_grad = T.basic.set_subtensor(A_t_grad[2,:],0)
      #A_t_grad = theano.printing.Print('r_t1')(A_t_grad)
      update = (self.A_t, self.A_t - self.eeta * A_t_grad)
      self.align = theano.function(inputs=[pi_t,z_t, oneCol],
                                   outputs=[self.A_t, cost],
                                   updates=[update],
                                   on_unused_input='warn',
                                   allow_input_downcast=True)
      
      #for numpy optimization
      A_t_ = T.matrix('A_t_')
      #A_t_.tag.test_value = temp
      #A_t_ = A_t_.reshape((3,3))
      A_t_.tag.test_value = temp
      #print_A_t = theano.printing.Print('r_t1')(A_t)
      r_t_ = T.dot(A_t_, pts.transpose()).transpose()
      r_t1_ = r_t_[:,0:2].reshape((2*self.numKeyPoints,1))
      #pi_tt = theano.printing.Print('pi_t before')(pi_t)
      diff_ = pi_t * (r_t1_ - self.mu)
      difft_ = diff_.reshape((1, 2 * self.numKeyPoints))
      
      #diff = theano.printing.Print('diff:')(diff)
      cost_1 = T.dot(T.dot(difft_,self.S),diff_)
      #cost_1 = theano.printing.Print('cost is:')(cost_1)
      cost_ = T.max(cost_1)
      
      A_t_grad_ = T.grad(cost=cost_, wrt=A_t_)
      A_t_grad_ = T.basic.set_subtensor(A_t_grad_[2,:],0)
      #A_t_grad_ = A_t_grad_.reshape((9,1))

      self.cost = theano.function(inputs=[A_t_, pi_t, z_t, oneCol],
                                  outputs=[cost_, A_t_grad_])
      i = T.iscalar('index')
      i.tag.test_value = 0
      subS = self.S[2*i:2*i+2, 2*i:2*i+2]
      #subS = theano.printing.Print('subS:')(self.S[2*i:2*i+2, 2*i:2*i+2])
      det = T.abs_(subS[0,0]*subS[1,1] - subS[0,1]*subS[1,0])
      subDiff = diff[(2*i):((2*i)+2)]
      subDifft = difft[0][(2*i):(2*i+2)]
      #intermed = theano.printing.Print('dotProd1:')(T.dot(subDifft,subS))
      intermed = T.dot(subDifft,subS)
      #intermed2 = theano.printing.Print('dotProd2:')(T.dot(intermed,subDiff))
      intermed2 = T.dot(intermed,subDiff)
      numrtr = T.exp(-0.5 * intermed2)
      k  = 2 
      dnmntr = T.sqrt((2**k) * det)
      q = numrtr/dnmntr
      temp = ((1 - self.alpha[2*i:2*i+2]) * q)/(self.alpha[2*i:2*i+2] + (1 - self.alpha[2*i:2*i+2]) * q)
      pi_t_out = T.basic.set_subtensor(pi_t[2*i:2*i+2], temp)
      self.q_pi_update = theano.function(inputs = [i, oneCol, pi_t, z_t], 
                                        outputs = [q,pi_t_out, r_t1],
                                        allow_input_downcast=True)
     
      self.train('12')


   def loadPicasaTubePickle(self):
      path = '/data/lisa/data/faces/EmotiW/picasa_tubes_pickles/'
      numToEmotion = {1:'Angry', 2:'Disgust', 3:'Fear', 4:'Happy', 5:'Neutral', 6:'Sad', 7:'Surprise'}
      folder = 'Train_'
      self.face_tubes = {}
      for i in range(7):
         fileName = folder+numToEmotion[i+1]+'.pkl'
         pkl_file = open(os.path.join(path, fileName), 'rb')
         self.face_tubes[numToEmotion[i+1]] = pickle.load(pkl_file)
         pkl_file.close()

   def loadData(self):
      path = '/data/lisa/data/faces/EmotiW/ramananExtract/matExtract/'
      
      self.pose = []
      self.poseDict = {}
      self.landmarks = []
      self.images = []
      self.poseCat = []
      numToEmotion = {1:'Angry', 2:'Disgust', 3:'Fear', 4:'Happy', 5:'Neutral', 6:'Sad', 7:'Surprise'}
      i = 0
      for root, subdirs, files in os.walk(path):
         for file in files:
            if os.path.splitext(file)[1].lower() in ('.mat'):
               matfile = sio.loadmat(os.path.join(path,file))
               folder = numToEmotion[int(file.split('_')[0])]
               clip = file.split('_')[1].split('-')[0]
               frame = int(file.split('_')[1].split('-')[1].split('.')[0])
               print folder, clip, frame
               if clip in self.face_tubes[folder]:
                  if i in range(len(self.face_tubes[folder][clip])):
                     if frame in self.face_tubes[folder][clip][i]:
                        xs = matfile['xs']
                        ys = matfile['ys']
                        a,b = xs.shape     
                        bbox = self.face_tubes[folder][clip][i][frame]
                        print bbox
                        (x1, y1, x2, y2) = bbox
                        #import pdb; pdb.set_trace();
                        validXs = numpy.logical_and(xs[0] >= x1, xs[0] <= x2,)
                        validYs = numpy.logical_and(ys[0] >= y1, ys[0] <= y2)
                        validXYs = numpy.logical_and(validXs, validYs)
                        numValids =  (validXYs * 1).sum()
                           
                        if(numValids > 0.75 * b): 
                           xs = xs.reshape(b,a)/1024.0
                           ys = ys.reshape(b,a)/576.0
                           if( b == 68):
                              self.poseCat.append('front')
                           else:
                              self.poseCat.append('profile')
                           xsNys = numpy.hstack((xs,ys)).reshape(2*b,1)
                           bs = matfile['bs']
                           self.landmarks.append(xsNys)
                           self.pose.append(bs[0,0]['c'][0])
                           self.images.append(file)
                     else:
                        continue
                  else:
                     continue
               else:
                  continue
               
      for i in range(19):
         self.poseDict[str(i)] = []
      for i in range(len(self.pose)):
         self.poseDict[str(self.pose[i][0])].append(i)
   


   def get_A_t(self, pi_t, z_t, numIter, method = 'gd'):
      oneCol = numpy.ones((self.numKeyPoints, 1))
      if(method == 'gd'):
         for i in range(numIter):
            A_t, cost = self.align(pi_t, z_t, oneCol)
         return A_t, cost 
      else:
         temp = numpy.random.random((9))
         temp[6:9] = [0,0,1]
         A_t, cost, uselessDict  = scipy.optimize.fmin_l_bfgs_b(func=cost_, x0=temp , fprime=None, args=(pi_t, z_t, oneCol, self))
         self.A_t.set_value(A_t.reshape((3,3)))
         return  A_t.reshape((3,3)), cost

   
   def load_image(self, index, transform=False):
      image = self.images[index]
      
      numToEmotion = {1:'Angry', 2:'Disgust', 3:'Fear', 4:'Happy', 5:'Neutral', 6:'Sad', 7:'Surprise'}
      folder = numToEmotion[int(image.split('_')[0])]
      path = '/data/lisa/data/faces/EmotiW/images/Train/'
      
      image = os.path.splitext(image.split('_')[1])[-2]+'.png'
      im = Image.open(os.path.join(path,folder,image))
      imT = self.get_transformed_image(im, self.A_t.get_value())
      savePath = os.path.join( '/data/lisa/data/faces/EmotiW/Aligned_Images/Train/', folder, image)
      #region = (512 - 100, 288 -100, 512 +100, 288+100)
      #face = imT.crop(region)      
#      print 'faceSize:', imT.size
      #imT.resize((96,96,)).show()
      imT.resize((96,96)).save(savePath)
  #   face.show()
      
   def get_transformed_image(self, img, A_t, drawKeyPoints = False):
      
      pixmap = img.load()
      imgT = Image.new('RGB', (200,200), 'black')
      pixmapT = imgT.load()
      A_t_inv = numpy.linalg.inv(A_t)
      indices = numpy.ones((3, 200,200))
      tempX  = numpy.asarray(range(200)).reshape((1,200))
      tempY  = numpy.asarray(range(200)).reshape((200,1))
      indices[0,:,:] = indices[0,:,:] * tempX
      indices[1,:,:] = indices[1,:,:] * tempY
      indicesScaled = numpy.copy(indices)
      indicesScaled[0,:,:] = (indicesScaled[0,:,:] + self.xMax/2 - 120)/self.xMax                       
      indicesScaled[1,:,:] = (indicesScaled[1,:,:] + self.yMax/2 - 120)/self.yMax
      inp =  numpy.dot(A_t_inv, indicesScaled.reshape((3, 200*200)) ).reshape((3,200,200))
      inpValid = numpy.logical_and(inp >= 0.0, inp <= 1.0 )
      inp[0,:,:] = (inp[0,:,:] * self.xMax)
      inp[1,:,:] = (inp[1,:,:] * self.yMax)
      inp = inp * inpValid
      inp = numpy.round(inp[0:2, :,:].reshape((2, 200*200)).transpose())
      indices = numpy.round(indices[0:2, :,:].reshape((2, 200*200)).transpose())
      for i in range(indices.size/2):
         (x,y) = (inp[i,0], inp[i,1])
         (x_, y_) = (indices[i,0], indices[i,1])
         #print (x,y), (x_,y_)
         pixmapT[x_, y_] = pixmap[x%self.xMax,y%self.yMax]
      
      '''
      for i in range(512-100, 512+100):    # for every pixel:
         for j in range(288-120, 288+80):
            x_ = float(i)/1024.0
            y_ = float(j)/576.0
            inp = numpy.dot(A_t_inv, numpy.array([[x_],[y_],[1]]))            
            x = int(1024.0 * inp[0])
            y = int(576.0 * inp[1]) 

            if (x < img.size[0]) and (y < img.size[1]):
               if (x >= 0) and (y >= 0):
                  pixmapT[i-412, j-168] = pixmap[x,y]
      '''      
      if (drawKeyPoints):
         mu = self.mu.get_value()
         for i in range(self.numKeyPoints):
            x = int(1024 * mu[2*i])%1024 - self.xMax/2 + 120
            y = int(576 * mu[2*i+1])%576 - self.yMax/2 + 120
            pixmapT[x, y] = (0,200,0)
               
    #  img.show()                      
    #  imgT.show()
      return imgT
             

   def get_q_pi(self, pi_t, z_t):
      oneCol = numpy.ones((self.numKeyPoints, 1))
      for i in range(self.numKeyPoints):
         q, pi_t, r_t = self.q_pi_update(i, oneCol, pi_t, z_t)
      return pi_t, r_t                           
      
   def train(self, pose):
       print 'in train'
#       method = 'gd'
       method = 'lbfgsb'
       numKeyPoints = self.numKeyPoints
       self.transMat = {}
       numEpochs = 1
       poses = []
       exampleIndices = []
       print 'length:', len(self.landmarks)
       
       for i in range(len(self.images)):
          if(self.poseCat[i] == 'profile' ):
            # if(self.pose[i] in poses):
                exampleIndices.append(i)

       totExamples = len(exampleIndices)
       print 'Number of examples per epoch, Number of Epochs'
       numExamples = totExamples/numEpochs
       print (numExamples, numEpochs)
       
       #setting self.mu as the average of keyPoints for particular pose
       sumZ_t = 0
       data = numpy.zeros((totExamples, 2 * self.numKeyPoints))
       print 'numKeyPoints:', self.numKeyPoints
       for i in range(totExamples):
          sumZ_t += self.landmarks[exampleIndices[i]]
          data[i,:] = self.landmarks[exampleIndices[i]].ravel()

      # print data.shape
       #c = numpy.cov(data.transpose()) + 0.01 * numpy.eye((2*self.numKeyPoints))
       #self.S.set_value(numpy.linalg.inv(c))
       self.mu.set_value(sumZ_t/totExamples)

       for epoch in range(numEpochs):
           print 'epoch:'
           print epoch

           sumMu = numpy.zeros((2*numKeyPoints,1))
           sumC = numpy.zeros((2*numKeyPoints, 2*numKeyPoints))
           sumPi = numpy.zeros((2*numKeyPoints,1))
           sumPiPi = numpy.zeros((2*numKeyPoints, 2*numKeyPoints))
           costEpoch = 0
           for t in range(numExamples):
               index = exampleIndices[epoch * numExamples + t]
               print 'example number:', t
               #get the keyPoint for example 't'
               z_t = self.landmarks[index]
                   
               #block 2
               pi_t = numpy.ones((2 * numKeyPoints,1))               
               numOutLoop = 1
               for i in range(numOutLoop):
                  A_t, cost = self.get_A_t(pi_t,z_t, 10000, method)
                  if i == numOutLoop-1:
                     costEpoch += cost
                  #ipdb.set_trace()
                  #pi_t_prev = pi_t
                  pi_t_temp, r_t = self.get_q_pi(pi_t, z_t)
                  #pi_t = pi_t_prev
                  
                  #pi_t = 0.9 * pi_t_prev + 0.1 * pi_t
               self.load_image(index, True) 
               self.transMat[self.images[index]] = self.A_t.get_value()
               #block 3, trying to eliminate loop by vectorization
               sumPi +=  pi_t
               sumPiPi += numpy.dot(pi_t, pi_t.transpose())
               difference = (self.mu.get_value() - r_t)
               sumC += numpy.dot(pi_t * difference, (pi_t * difference).transpose())
               sumMu += pi_t * r_t
           #update
           print 'Average cost:'
           print costEpoch
           self.alpha.set_value(1 - sumPi/numExamples)
           #print self.alpha.get_value()
           self.mu.set_value(sumMu/sumPi)
           print self.mu.get_value().transpose()
           print self.alpha.get_value().transpose()
           #print self.mu.get_value()

           
           C = sumC/sumPiPi + 0.1 * numpy.eye(2*self.numKeyPoints)
           #C = sumC/sumPiPi
           #print numpy.linalg.eigvalsh(C) 
           self.S.set_value(numpy.linalg.inv(C))
           print self.S.get_value()

       output = open('transMat_pose_profile.pkl', 'wb')
       data = (self.transMat, self.mu, self.S, self.alpha)
       pickle.dump(data, output)
       output.close()
           
       
def test():
    obj = faceAlign()
    

if __name__ == '__main__':
    test()


