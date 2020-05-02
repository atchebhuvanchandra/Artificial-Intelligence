from __future__ import print_function
import numpy as np
from numpy import *
import cv2
np.random.seed(1337)  # for reproducibility
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from PIL import Image # This will be used to read/modify images (can be done via OpenCV too)
from keras import backend as K

batch_size = 128
nb_classes = 2
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 128, 128
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# define path to images:

# This is the path of our positive input dataset
pos_im_path = r"positive" 

# define the same for negatives
neg_im_path= r"negative"

# read the image files:

pos_im_listing = os.listdir(pos_im_path) # it will read all the files in the positive image path (so all the required images)
neg_im_listing = os.listdir(neg_im_path)
num_pos_samples = size(pos_im_listing) # simply states the total no. of images
num_neg_samples = size(neg_im_listing)

print('training data :')
print('No. of positive samples  '+str(num_pos_samples))# prints the number value of the no.of samples in positive dataset
print('No. of Negative samples  '+str(num_neg_samples))

dataset_x = []
dataset_y = []
for file in pos_im_listing: #this loop enables reading the files in the pos_im_listing variable one by one

    path = pos_im_path + '\\' + file
    img = Image.open(path)
    im = img.resize((img_rows,img_cols))
    gray = im.convert('L')
    dataset_x.append(np.reshape(gray, [img_rows,img_cols,1]))
    dataset_y.append(1)

# Same for the negative images
for file in neg_im_listing:
    
    path = neg_im_path + '\\' + file
    img = Image.open(path)
    im = img.resize((img_rows,img_cols))
    gray = im.convert('L')
    dataset_x.append(np.reshape(gray, [img_rows,img_cols,1]))
    dataset_y.append(0)

ds_y = []
dataset_x = np.array(dataset_x)

"""shuffle dataset"""
p = np.random.permutation(len(dataset_x))
print(p)
dataset_x = dataset_x[p]
for i in p:
    ds_y.append(dataset_y[i])
        
X_test = dataset_x[:int(len(dataset_x)/3)]
Y_test = ds_y[:int(len(dataset_x)/3)]
X_train = dataset_x[int(len(dataset_x)/3):]
Y_train = ds_y[int(len(dataset_x)/3):]

print("normalizing")

# Reshaping the array to 4-dims so that it can work with the Keras API
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
# Making sure that the values are float so that we can get decimal points after division
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
X_train /= 255
X_test /= 255
print('x_train shape:', X_train.shape)
print('Number of images in x_train', X_train.shape[0])
print('Number of images in x_test', X_test.shape[0])


# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape, init='glorot_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Activation('relu'))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(x=X_train,y=Y_train, epochs=10)


print("Miss Rate and Accuracy for adam optimizer: ", model.evaluate(X_test, Y_test))


# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape, init='glorot_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Activation('relu'))

model.compile(optimizer='RMSProp', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(x=X_train,y=Y_train, epochs=10)


print("Miss Rate and Accuracy for SGD optimizer: ", model.evaluate(X_test, Y_test))


# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape, init='glorot_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.compile(optimizer='SGD', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(x=X_train,y=Y_train, epochs=10)


print("Miss Rate and Accuracy : ", model.evaluate(X_test, Y_test))
