import Image
import numpy
from datetime import datetime
import math
import glob
import os
import pickle
from crop_match import match_subregion
import sys
from subprocess import call
import fnmatch

import os


def get_images(original, cropped):
    # Generate the required numpy arrays
    original = Image.open(original)
    cropped = Image.open(cropped)
    #originalArr = numpy.transpose(numpy.array(original), [2, 0, 1]).astype('float32')
    #croppedArr = numpy.transpose(numpy.array(cropped), [2, 0, 1]).astype('float32')
    originalArr = numpy.array(original).astype('float64')
    croppedArr = numpy.array(cropped).astype('float64')
    return originalArr, croppedArr


def save(name, val):
    outf =  open(name, 'a')
    stri = str(int(val[0])) + ', ' + str(int(val[1])) + ', ' +  str(int(val[2])) + ', ' + str(int(val[3])) + '\n'
    outf.write(stri)
    outf.close()
    #else:
	#with open(name, 'w+') as outf:
	#for val in vals:
	 #   str = "{}, {}, {}, {}\n".format(int(val[0]), int(val[1]), int(val[2]), int(val[3]))
	  #  outf.write(str)
	  
def writefile(name, files):
	outf = open(name, 'w+')
	for f in files:
	    outf.write(f + '\n')

if __name__ == "__main__":
		
    orig_path = "/data/lisatmp2/EmotiWTest/Test_Vid_Distr/ExtractedFrame"
    cropped_path = "/data/lisatmp2/EmotiWTest/Test_Vid_Distr/Faces"
    save_path = "/data/lisatmp2/EmotiWTest/Test_Vid_Distr/BoundBoxData"
    
    report_path = "/data/lisatmp2/EmotiWTest/Test_Vid_Distr"
    
    missed = []

    
    report = open(report_path + '/bboxReport.txt', 'a')
    
    
    
    res = []
    #emots = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    #emots = ['Happy']
    
    extractFolders = ["/data/lisatmp2/EmotiWTest/Test_Vid_Distr/ExtractedFrame/000814440", "/data/lisatmp2/EmotiWTest/Test_Vid_Distr/ExtractedFrame/002952120"]
    extractFolders.sort()
    
    print "finished sorting"
    
    
    for folder in extractFolders:
#    for emo in emots:
        
        print "folderpath: ", folder
        
        folder_name = folder.split("/")[-1]
        
        bbox_path = save_path + "/" + folder_name
        if not os.path.exists(bbox_path):
            call(["mkdir", bbox_path])
            
        
        face_list = glob.glob("{}/{}*.jpg".format(cropped_path, folder_name))
        frame_list = glob.glob("{}/*.png".format(folder))
        
        
        if face_list.__len__() == 0:
            print "empty face list for video: ", folder
            report.write("empty face list for video: " + folder)
            continue
            
        face_list.sort()
        frame_list.sort()
        temp = 0
        error = 0
        
        for face in face_list:
            
            cropNameTmp = face.split('/')[-1].rstrip('.jpg').split("-")
            
            cropName = cropNameTmp[0] + "-" + cropNameTmp[1][:3]
            #print "cropName", cropName
            
            found = False
            start = temp
        
            for i in range(start, frame_list.__len__()):
               
                origName = frame_list[i].split('/')[-1].rstrip('_.png')
                
                #print "origName", origName
                
                if not cropName == origName:  
                    temp += 1
            
                
                elif cropName == origName:
                    #print 'orig Frame: ', frame_list[i].split('/')[-1]
                    origJPG = frame_list[i].rstrip('.png') + '.jpg'
                    if not os.path.exists(origJPG):
                        call(["convert", frame_list[i], origJPG])
                    orig_arr, crop_arr = get_images(origJPG, face)
                    #print orig_arr, crop_arr
                        
                    try:
                        res =  match_subregion(orig_arr, crop_arr) #, order = 1)
                        res = [res[1], res[0], res[1] + crop_arr.shape[1], res[0] + crop_arr.shape[0]]
                        found = True
                        break
                    except:
                        raise
                        print "failed: " + face
                        report.write("failed: " + face + '\n')
                    	#missed.append([set_n, emot, orig])
            if not found:
                print "not found: " + face.split('/')[-1]
                report.write("not found: " + face + '\n')
                temp = 0
                error += 1
            else:
                #print "crop found: " , face
                save_name = bbox_path + '/' + cropName + ".txt"
                outf = open(save_name, 'a')
                stri = str(int(res[0])) + ', ' + str(int(res[1])) + ', ' +  str(int(res[2])) + ', ' + str(int(res[3])) + '\n'
                outf.write(stri)
                outf.close()
        if error == 0:
            print "successfully completed for folder: ", folder
        else:
            print "bbox folder errors: ", folder
                
                
                


'''
	elif which == "AFEW":
			orig_path = "/data/lisa/data/faces/AFEW/images/"
			cropped_path = "/data/lisa/data/faces/AFEW/picasa_faces/"
			save_path = "/data/lisa/data/faces/AFEW/picasa_boxes/"
        	missed = []
        	for emot in ["Angry", "Disgust", "Fear", "Surprise", "Sad", "Happy", "Neutral"]:
            print emot, "\n"
            files = glob.glob("{}{}/*.png".format(orig_path, emot))
            for orig in files:
                crop = orig.split('/')[-1].rstrip('.png')
                crops = glob.glob("{}{}/{}*".format(cropped_path, emot, crop))
                corners = []
                for crop in crops:
                    if os.path.isfile(crop):
                        orig_arr, crop_arr = get_images(orig, crop)
                        try:
                            res =  match_subregion(orig_arr, crop_arr) #, order = 1)
                            print "passed: {}".format(crop)
                            res = [res[1], res[0], res[1] + crop_arr.shape[1], res[0] + crop_arr.shape[0]]
                            corners.append(res)
                        except:
                            print "failed: {}".format(crop)
                            missed.append([set_n, emot, orig])
                if len(corners) > 0:
                    print corners
                    save_name = "{}{}/{}.txt".format(save_path, emot, orig.split('/')[-1].rstrip('.png'))
                    save(save_name, corners)

   	 	print "Done, failed on: ", missed
   	 	with open("failed.pkl", 'w') as output:
			pickle.dump(missed, output)
'''
