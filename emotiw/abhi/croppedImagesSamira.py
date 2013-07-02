import os
import Image
import pickle

def mainFunc():
    face_tubes = loadPicasaTubePickle('Val_')
    path = '/data/lisa/data/faces/EmotiW/images/Val/'
    pathSave = '/data/lisa/data/faces/EmotiW/CroppedFaces20Margin/Val/'
    Width = 1024
    Height = 576
    margin = 0.2
    for root, subdir, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1].lower() in ('.png'):
                key =  root.split('/')[-1]
                clip = file.split('-')[0]
                frame = int(file.split('-')[1].split('.')[0])
                print key, clip, frame
                if key in face_tubes:
                     if clip in face_tubes[key]:
                         if(len(face_tubes[key][clip]) > 0):
                             if frame in face_tubes[key][clip][0]:
                                 bbox = face_tubes[key][clip][0][frame]
                                 w = bbox[2]-bbox[0]
                                 h = bbox[3]-bbox[1]
                                 x1 = max(0, int(bbox[0] - margin * w))
                                 y1 = max(0, int(bbox[1] - margin * h))
                                 x2 = min(1023, int(bbox[2] + margin * w))
                                 y2 = min(575, int(bbox[3] + margin * h))
                                 im = Image.open(os.path.join(root, file))
                                 region = (x1, y1, x2, y2)
                                 img = im.crop(region)
                                 name = os.path.join(pathSave, key, str(x1)+'_'+str(y1)+'_'+str(x2)+'_'+str(y2)+'_'+file)
                                 img.save(name)
                                 
                                 

                                 

                                 
    

def loadPicasaTubePickle(folder):
    path = '/data/lisa/data/faces/EmotiW/picasa_tubes_pickles/'
    numToEmotion = {1:'Angry', 2:'Disgust', 3:'Fear', 4:'Happy', 5:'Neutral', 6:'Sad', 7:'Surprise'}
    face_tubes = {}
    for i in range(7):
        fileName = folder+numToEmotion[i+1]+'.pkl'
        pkl_file = open(os.path.join(path, fileName), 'rb')
        face_tubes[numToEmotion[i+1]] = pickle.load(pkl_file)
        pkl_file.close()
    return face_tubes

def load_image(self, image, A_t, transform=False):
      
    numToEmotion = {1:'Angry', 2:'Disgust', 3:'Fear', 4:'Happy', 5:'Neutral', 6:'Sad', 7:'Surprise'}
    folder = numToEmotion[int(image.split('_')[0])]
    #path = '/data/lisa/data/faces/EmotiW/images/Val/'
    path = self.rPath
      
    image = os.path.splitext(image.split('_')[1])[-2]+'.png'
    im = Image.open(os.path.join(path,folder,image))
    imT = self.get_transformed_image(im, A_t)
    #savePath = os.path.join( '/data/lisa/data/faces/EmotiW/Aligned_Images/Val/', folder, image)
    savePath = os.path.join( self.sPath, folder, image)
      #region = (512 - 100, 288 -100, 512 +100, 288+100)
      #face = imT.crop(region)      
#      print 'faceSize:', imT.size
 #     imT.show()
 #     raw_input('press any key to continue')
    imT.resize((96,96)).save(savePath)
  #   face.show()

if __name__ == '__main__':
    mainFunc()
