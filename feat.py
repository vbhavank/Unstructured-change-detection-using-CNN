#This file contains the python implementation feature based change detector
#Author: Bhavan Vasu

import tensorflow as tf
import keras
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
import sys
from skimage import filters #change to 'import filter' for Python>v2.7
from skimage import exposure
from keras import backend as K

#Function to retrieve features from intermediate layers
def get_activations(model, layer_idx, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer_idx].output,])
    activations = get_activations([X_batch,0])
    return activations

#Function to extract features from intermediate layers
def extra_feat(img_path):
        #Using a VGG19 as feature extractor
        base_model = VGG19(weights='imagenet',include_top=False)
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
	
	F = tf.concat([x3,x2,x1,x4,x5],3) #Change to only x1, x1+x2,x1+x2+x3..so on, inorder to visualize features from diffetrrnt blocks
        return F
def main():
  if (len(sys.argv))>3:
    print "Invalid number of input arguments "
    exit(0)

  #Two aerial patches with change or No change
  img_path1=sys.argv[1]
  img_path2=sys.argv[2]

  sess = tf.InteractiveSession()

  F1=extra_feat(img_path1) #Features from image patch 1
  F1=tf.square(F1)
  F2=extra_feat(img_path2) #Features from image patch 2
  F2=tf.square(F2)
  d=tf.subtract(F1,F2)
  d=tf.square(d) 
  d=tf.reduce_sum(d,axis=3) 

  dis=(d.eval())   #The change map formed showing change at each pixels
  dis=np.resize(dis,[112,112])

  # Calculating threshold using Otsu's Segmentation method
  val = filters.threshold_otsu(dis[:,:])
  hist, bins_center = exposure.histogram(dis[:,:],nbins=256)

  plt.title('Unstructured change')
  plt.imshow(dis[:,:] < val, cmap='gray', interpolation='bilinear')
  plt.axis('off')
  plt.tight_layout()
  plt.show()
  """
  Uncomment For veiwing a graph for visualizing threshold selection
  plt.subplot(144)
  plt.title('Otsu Threshold selection')
  plt.plot(bins_center, hist, lw=2)
  plt.axvline(val, color='k', ls='--')

  plt.tight_layout()
  plt.show()
  """
if __name__ == "__main__":
    main()
