import os
from os import chdir
from pylab import *

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
test_directory = "C:\Users\Sebastien\Documents\Faces_Aligned_Test_Small" #With PIL: ANTIALIAS

chdir("C:\Users\Sebastien\Documents\Aligned_faces_org_small_Essentials")
import logreg

mean_inter = np.load("mean_inter_submitted.npy")
centroids = np.load("centroids_submitted.npy")
V_list = np.load("V_list_submitted.npy")

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
lr.weights = loadtxt("weights_submitted.txt")
lr.biases = loadtxt("biases_submitted.txt")

test = []

for j in sorted(os.listdir(test_directory)):
    for k in sorted(os.listdir(test_directory+"\\"+j)):
        test.append(imread(test_directory+"\\"+j+"\\"+k)[v_min:v_max,h_min:h_max])

test = asarray(test)
num_test_images = shape(test)[0]

test_features = zeros((v_sections*h_sections*num_centroids,num_test_images))

for j in xrange(num_test_images/50):
    print j
    test_features[:,50*j:50*(j+1)] = feature_extract(test[50*j:50*(j+1)])

test_features[:,num_test_images - num_test_images%50:num_test_images] = feature_extract(test[num_test_images - num_test_images%50:num_test_images])

test_probabilities = []

start = 0
end = 0
for j in sorted(os.listdir(test_directory)):
    end += shape(sorted(os.listdir(test_directory+"\\"+j)))[0]
    test_probabilities.append(mean(lr.probabilities(test_features[:,start:end]),1))
    start = end

test_probabilities = asarray(test_probabilities)

test_probabilities_formatted = zeros((shape(os.listdir(test_directory))[0],7))

test_probabilities_formatted[:,0:4] = test_probabilities[:,0:4].copy()
test_probabilities_formatted[:,4:6] = test_probabilities[:,5:7].copy()
test_probabilities_formatted[:,6] = test_probabilities[:,4].copy()

test_probabilities_formatted_v2 = test_probabilities_formatted.copy()

np.save("test_probabilities_submitted.npy",test_probabilities_formatted_v2)