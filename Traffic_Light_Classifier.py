# import the libraries and resources

import cv2 # computer vision library
import helpers # helper functions

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images

%matplotlib inline

# Image data directories
IMAGE_DIR_TRAINING = "traffic_light_images/training/"
IMAGE_DIR_TEST = "traffic_light_images/test/"

# Using the load_dataset function in helpers.py
# Load training data
IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)

# Pre-process the Data

# This function should take in an RGB image and return a new, standardized version
def standardize_input(image):
    
    ## Resize image and pre-process so that all "standard" images are the same size  
    standard_im = np.copy(image)
    standard_im_resize = cv2.resize(standard_im,(32,32)) 
    
    return standard_im_resize
    
## One hot encode an image label
## Given a label - "red", "green", or "yellow" - return a one-hot encoded label

# Examples: 
# one_hot_encode("red") should return: [1, 0, 0]
# one_hot_encode("yellow") should return: [0, 1, 0]
# one_hot_encode("green") should return: [0, 0, 1]

def one_hot_encode(label):
    
    ## Create a one-hot encoded label that works for all classes of traffic lights
    one_hot_encoded = [] 
    if (label == "red"):
        one_hot_encoded = [1,0,0]
    elif (label == "yellow"):
        one_hot_encoded = [0,1,0]
    else:
        one_hot_encoded =[0,0,1]