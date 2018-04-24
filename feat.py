#This file contains the python implementation feature based change detector
#Author: Bhavan Vasu
import tensorflow as tf
import keras
import cv2 as cv2
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage import filters
from skimage import exposure
from keras import backend as K

# Function to convert rgb to gray
def rgb2gray(rgb):
    r, g, b = rgb[:,:,:,0], rgb[:,:,:,1], rgb[:,:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

sess = tf.InteractiveSession()
#Using a VGG19 as feature extractor
base_model = VGG19(weights='imagenet',include_top=False)

#Two aerial patches with change or No change
img_path1 = './Im5.tiff'
img_path2 = './Im6.tiff'

#Function to retrieve features from intermediate layers
def get_activations(model, layer_idx, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer_idx].output,])
    activations = get_activations([X_batch,0])
    return activations

#Function to extract features from intermediate layers
def extra_feat(img_path):
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
        block1_pool_features=get_activations(base_model, 3, x)
        block2_pool_features=get_activations(base_model, 6, x)
        block3_pool_features=get_activations(base_model, 10, x)
        block4_pool_features=get_activations(base_model, 14, x)
        block5_pool_features=get_activations(base_model, 18, x)

	x1 = tf.image.resize_images(block1_pool_features[0],[112,112])
	x2 = tf.image.resize_images(block2_pool_features[0],[112,112])
	x3 = tf.image.resize_images(block3_pool_features[0],[112,112])
	x4 = tf.image.resize_images(block4_pool_features[0],[112,112])
	x5 = tf.image.resize_images(block5_pool_features[0],[112,112])
	
	F = tf.concat([x1,x2,x3,x4,x5],3) #Change to only x1, x1+x2,x1+x2+x3..so on inorder to visualize features from diffetrrnt blocks
        return F

F1=extra_feat(img_path1) #Features from image patch 1
F1=tf.square(F1)
F2=extra_feat(img_path2) #Features from image patch 2
F2=tf.square(F2)
d=tf.subtract(F1,F2)
d=tf.square(d)


d=tf.reduce_sum(d,axis=3) 

dis=(d.eval())   #The change map formed showing change at each pixels
dis=np.resize(dis,[112,112])

img = image.load_img(img_path1, target_size=(112, 112))

x1 = image.img_to_array(img)
x1 = np.expand_dims(x1, axis=0)
x1 = preprocess_input(x1)
img = rgb2gray(x1)
img=np.resize(img,[112,112])

# Calculating threshold using Otsu's Segmentation method
val = filters.threshold_otsu(dis[:,:])
hist, bins_center = exposure.histogram(dis[:,:],nbins=256)
plt.figure(figsize=(9, 4))
img1 = image.load_img(img_path1, target_size=(224, 224))

plt.subplot(141)
plt.title('Source Image')
plt.imshow(img1)
plt.axis('off')
img2 = image.load_img(img_path2, target_size=(224, 224))

plt.subplot(142)
plt.title('Target Image')
plt.imshow(img2)
plt.axis('off')

plt.subplot(143)
plt.title('Unstructured change')
plt.imshow(dis[:,:] < val, cmap='gray', interpolation='bilinear')
plt.axis('off')

plt.subplot(144)
plt.title('Otsu Threshold selection')
plt.plot(bins_center, hist, lw=2)
plt.axvline(val, color='k', ls='--')

plt.tight_layout()
plt.show()

   
