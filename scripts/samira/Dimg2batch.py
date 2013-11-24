#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# Pierre Froumenty
# Récupère automatiquement les images des dossiers train_path et test_path et crée des batchs de la même structure que ceux utilisés
# par Alex K. dans son exemple avec le dataset CIFAR

import sys

import numpy.random as nr
import numpy
import random as r
import pickle
from os import listdir
from os.path import isfile, join
import math
#import Image  
from PIL import Image  
from distutils.version import StrictVersion
from random import shuffle

# Params
path_in = str(sys.argv[1])
path_out = str(sys.argv[2])
test_path = path_in


num_cases_per_batch = 7178
image_size = 48
channels = 1 # Nombre de canaux (1 pour niveau de gris)
label_names = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
test_eval_label = 'str(0)'  # 0 arbitraire
num_vis = image_size*image_size*channels # nombre de valeurs par image

#################
# assertions
assert(channels==1 or channels==3)

# Lecture des images de test seulement
test_files = [ f for f in listdir(test_path) if isfile(join(test_path,f)) ]
# No need to sort for test only
#test_files.sort(key=lambda x: int(x.split('_')[0])) # IMPORTANT : Tri selon les chiffres et non l ordre alphabetique !!! # AFEW
n_images_test = len(test_files);
n_batch = int(math.ceil(float(n_images_test)/float(num_cases_per_batch)))
print("Number of batches for test : "+str(n_batch))

# Pour chaque batch
for j in range(1,n_batch+1):
	file_start = (j-1)*num_cases_per_batch;
	file_stop = ((j)*num_cases_per_batch)-1 if ((j)*num_cases_per_batch)<=(len(test_files)) else len(test_files)-1
	print("batch for images : "+str(file_start)+"-"+str(file_stop)+"\n")
	batch = {
		"batch_label" : "test batch{0} of {1}".format(j,n_batch),
		"data" : [],
		"filenames" : test_files[file_start:file_stop+1],
		"labels" : []
	}
	# Parcours des fichiers pour 1 batch
	for f in batch["filenames"]:
		if f[-3:] == 'png':
			batch["labels"].append(eval(test_eval_label)) # ajout du label 0 arbitraire
			imagefile = Image.open(join(test_path,f))
			image = list(imagefile.getdata()) # is that C-ordered !!?? (row-major)
			if channels==3:
				image = numpy.transpose(image,(1,0))
				image1 = numpy.reshape(image[0],(image_size*image_size), order='F')
				image2 = numpy.reshape(image[1],(image_size*image_size), order='F')
				image3 = numpy.reshape(image[2],(image_size*image_size), order='F')
				image = numpy.concatenate([image1,image2,image3],axis=0)
			batch["data"].append(image) # ajout de l'image
	batch["data"] = numpy.transpose(numpy.array(batch["data"],dtype=numpy.uint8)) # Conversion en numpy array et transposition
	
	# sauvegarde du batch
	with open(path_out+"/data_batch_"+str(j),"wb") as fileout:
		mypickler = pickle.Pickler(fileout)
		mypickler.dump(batch)
		fileout.close()
		print(fileout.name+" created")

# Meta information do not have to be created (test only)



