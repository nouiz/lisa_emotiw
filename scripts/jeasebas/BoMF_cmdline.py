#BoMF_cmdline.py arg1 arg2 [arg3]
#arg1: Original directory
#arg2: Target directory
#arg3 (optional): positive integer
#BoMF_cmdline.py C:\Users\Sebastien\Documents\Faces_Aligned_Test C:\Users\Sebastien\Documents\Faces_Aligned_Test_Small 55

import sys

#print (sys.argv)

import os
import Image
#from os import chdir #Removed
from pylab import *
import logreg

original_dir = sys.argv[1] #"C:\Users\Sebastien\Documents\Faces_Aligned_Test" #May be modified
target_dir = sys.argv[2] #original_dir + "_Small"
params_dir = sys.argv[3]
pred_dir = sys.argv[4]
batch_size = int(sys.argv[5])
clip_ids = sys.argv[6:]

#if len(sys.argv) >= 6:
#    batch_size = int(sys.argv[5])
#else:
#    batch_size = 50


import Image

if (os.path.isdir(target_dir) == False):
    os.makedirs(target_dir)

#for j in sorted(os.listdir(original_dir)):
for j in clip_ids:
    if not os.path.isdir(os.path.join(target_dir, j)):
        os.makedirs(os.path.join(target_dir,j))
        for k in sorted(os.listdir(os.path.join(original_dir,j))):
            cur_image = Image.open(os.path.join(original_dir,j,k))
            mod_image = cur_image.resize((71,90),Image.ANTIALIAS)
            mod_image.save(os.path.join(target_dir,j,k))


num_training_patches = 10
patchsize=8
eps= 0.000000001
var_treshold = 0.9
size_ = patchsize*patchsize*3
num_centroids = 400
v_min = 44
v_max = 82
h_min = 16
h_max = 66
v_sections = 4
h_sections = 4

v_size = v_max - v_min
h_size = h_max - h_min

#test_directory = "C:\Users\Sebastien\Documents\EmotiW_test_small" #With Irfanview
#test_directory = "C:\Users\Sebastien\Documents\Faces_Aligned_Test_Small" #With PIL: ANTIALIAS

#chdir("C:\Users\Sebastien\Documents\Aligned_faces_org_small_Essentials") #removed
#import logreg

mean_inter = np.load(os.path.join(params_dir, "mean_inter_submitted.npy"))
centroids = np.load(os.path.join(params_dir, "centroids_submitted.npy"))
V_list = np.load(os.path.join(params_dir, "V_list_submitted.npy"))

def feature_extract(array_im): #shape num_im, v_size, h_size, 3
    
    cur_patches = zeros((array_im.shape[0],v_sections*h_sections,((v_size-patchsize)/v_sections)*((h_size-patchsize)/h_sections),size_))
    cur_white_patches = []
    
    for i_v in xrange(v_sections):
        for i_h in xrange(h_sections):
            for i in xrange((v_size-patchsize)/v_sections):
                for j in xrange((h_size-patchsize)/h_sections):
                    patch = array_im[:,i_v*(v_size-patchsize)/v_sections+i:i_v*(v_size-patchsize)/v_sections+i+patchsize,i_h*(h_size-patchsize)/h_sections+j:i_h*(h_size-patchsize)/h_sections+j+patchsize].copy()
                    #print shape(patch)
                    #print (array_im.shape[0],size_)                    
                    patch = reshape(patch, (array_im.shape[0],size_)).transpose(1,0)
                    patch -= mean(patch,0)
                    patch /= (std(patch,0)+eps)
                    patch = patch.transpose(1,0)
                    cur_patches[:,i_v*h_sections+i_h,(h_size-patchsize)/h_sections*i+j] = patch.copy()

    for i_v in xrange(v_sections):
        for i_h in xrange(h_sections):
            cur_patches[:,i_v*h_sections+i_h] -= mean_inter[i_v*h_sections+i_h]
    
    cur_patches = cur_patches.transpose(0,1,3,2)

    for i_v in xrange(v_sections):
        for i_h in xrange(h_sections):
            cur_white_patches.append((dot(V_list[i_v*h_sections+i_h],cur_patches[:,i_v*h_sections+i_h])).transpose(1,0,2))

    feature_map = zeros((array_im.shape[0],v_sections*h_sections,((v_size-patchsize)/v_sections)*((h_size-patchsize)/h_sections),num_centroids))
    
    for i_v in xrange(v_sections):
        for i_h in xrange(h_sections):
            for j in xrange(((v_size-patchsize)/v_sections)*((h_size-patchsize)/h_sections)):
                z = zeros((array_im.shape[0],num_centroids))
                f = zeros((array_im.shape[0],num_centroids))
                for k in range(num_centroids):
                    z[:,k] = sqrt(sum((cur_white_patches[i_v*h_sections+i_h][:,:,j]-centroids[i_v*h_sections+i_h][:,k])**2,axis=1))
                mu = mean(z,1) # one mean per image
                for k in range(num_centroids):
                    f[:,k] = clip(mu - z[:,k],0,inf)
                feature_map[:,i_v*h_sections+i_h,j,:] = f.copy()   
    
    pooled_features = zeros((array_im.shape[0],v_sections*h_sections*num_centroids))

    for i_v in xrange(v_sections):
        for i_h in xrange(h_sections):
            pooled_features[:,(i_v*h_sections+i_h)*num_centroids:(i_v*h_sections+i_h+1)*num_centroids] = mean(feature_map[:,i_v*h_sections+i_h],1)
    
    if array_im.shape[0] == 1:
        pooled_features = reshape(pooled_features, (v_sections*h_sections*num_centroids))
    
    return pooled_features.T

numclasses = 7
wc = 1e-3

lr = logreg.Logreg(numclasses, v_sections*h_sections*num_centroids)
lr.weights = loadtxt(os.path.join(params_dir, "weights_submitted.txt"))
lr.biases = loadtxt(os.path.join(params_dir, "biases_submitted.txt"))

test = []

#for j in sorted(os.listdir(target_dir)):
for j in clip_ids:
    for k in sorted(os.listdir(os.path.join(target_dir,j))):
        test.append(imread(os.path.join(target_dir,j,k))[v_min:v_max,h_min:h_max])

test = asarray(test)
num_test_images = shape(test)[0]

test_features = zeros((v_sections*h_sections*num_centroids,num_test_images))

for j in xrange(num_test_images/batch_size):
    #print j
    test_features[:,batch_size*j:batch_size*(j+1)] = feature_extract(test[batch_size*j:batch_size*(j+1)])

test_features[:,num_test_images - num_test_images%batch_size:num_test_images] = feature_extract(test[num_test_images - num_test_images%batch_size:num_test_images])

test_probabilities = []

start = 0
end = 0
#for j in sorted(os.listdir(target_dir)):
for j in clip_ids:
    end += shape(sorted(os.listdir(os.path.join(target_dir,j))))[0]
    test_probabilities.append(mean(lr.probabilities(test_features[:,start:end]),1))
    start = end

test_probabilities = asarray(test_probabilities)

#test_probabilities_formatted = zeros((shape(os.listdir(target_dir))[0],7))
test_probabilities_formatted = zeros((len(clip_ids), 7))

test_probabilities_formatted[:,0:4] = test_probabilities[:,0:4].copy()
test_probabilities_formatted[:,4:6] = test_probabilities[:,5:7].copy()
test_probabilities_formatted[:,6] = test_probabilities[:,4].copy()

test_probabilities_formatted_v2 = test_probabilities_formatted.copy()

np.save(os.path.join(pred_dir, "BoMF_test_probabilities.npy"),
        test_probabilities_formatted_v2)
