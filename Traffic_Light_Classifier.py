# %%
# import the libraries and resources

import cv2 # computer vision library
import helpers # helper functions

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images

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
    
    return one_hot_encoded

# Importing the tests
import test_functions
tests = test_functions.Tests()

# Test for one_hot_encode function
tests.test_one_hot(one_hot_encode)

## Construct a STANDARDIZED_LIST of input images and output labels.

def standardize(image_list):
    
    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # One-hot encode the label
        one_hot_label = one_hot_encode(label)    

        # Append the image, and it's one hot encoded label to the full, processed list of image data 
        standard_list.append((standardized_im, one_hot_label))
        
    return standard_list

# Standardize all training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)

## Convert and image to HSV colorspace
## Visualize the individual color channels

image_num = 0
test_im = STANDARDIZED_LIST[image_num][0]
test_label = STANDARDIZED_LIST[image_num][1]

# Convert to HSV
hsv = cv2.cvtColor(test_im, cv2.COLOR_RGB2HSV)

# Print image label
print('Label [red, yellow, green]: ' + str(test_label))

# HSV channels
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

# Plot the original image and the three channels
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
ax1.set_title('Standardized image')
ax1.imshow(test_im)
ax2.set_title('H channel')
ax2.imshow(h, cmap='gray')
ax3.set_title('S channel')
ax3.imshow(s, cmap='gray')
ax4.set_title('V channel')
ax4.imshow(v, cmap='gray')

## Calculates average red, green ,blue pixels in an image
def avg_rgb(rgb_image):
    red_sum = np.sum(rgb_image[:,:,0])
    green_sum = np.sum(rgb_image[:,:,1])
    blue_sum = np.sum(rgb_image[:,:,2])
    total = rgb_image.shape[0] * rgb_image.shape[1]
    
    return [red_sum/total,green_sum/total,blue_sum/total]

## This creates a brightness feature that takes in an RGB image and outputs a feature vector and/or value
## This feature uses HSV colorspace values
def create_feature(rgb_image):
    
# Convert image to HSV color space
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    
# Create empty list of feature 
    feature = []
    lowrange = np.array([0,0,20
                        ])
    highrange = np.array([256,60,205])
# Mask the image
    mask = cv2.inRange(hsv,lowrange,highrange)
    
    masked_image = np.copy(hsv)
    
    masked_image[mask != 0] = [0,0,0]
    conversion_image = np.copy(masked_image)
# Convert to RGB
    rgb = cv2.cvtColor(conversion_image, cv2.COLOR_HSV2RGB)
    
    rowcrop = 9
    colcrop = 6
    
    crop = np.copy(rgb)
    crop = crop[colcrop:-colcrop, rowcrop:-rowcrop,:]

    
    rgb_average = avg_rgb(crop)
    feature.append(rgb_average)
    
    


    return feature

## This function should take in RGB image input
# Analyze that image using your feature creation code and output a one-hot encoded label
def estimate_label(rgb_image):
    features = create_feature(rgb_image)
    ## Extract feature(s) from the RGB image and use those features to
    ## classify the image and output a one-hot encoded label
    predicted_label = []
    red = features[0][0]
    green = features[0][1]
    blue = features[0][2]
    
    if red > green:
        predicted_label = [1,0,0]
    
    elif (red < green) and (blue > 180):
        predicted_label = [1,0,0]
        
    elif red < green:
        predicted_label = [0,0,1]
        
    elif red == green:
        predicted_label = [0,1,0]
    
    else:
        predicted_label = [0,1,0]
    
    
    return predicted_label   
    
# %%
