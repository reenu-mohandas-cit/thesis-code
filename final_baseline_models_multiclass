#this code contains the baseline models, M1, M2, M3, M4, M5, M6 used for initial analysis using the datasets

import os
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.image as mpimg
from sklearn import preprocessing

from google.colab import drive
drive.mount('/content/gdrive')


import os
import numpy as np
from PIL import Image
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D


def model_M1():
    input_shape = (32, 32, 3)
     #single convolutional layer followed by a pooling layer
    model = Sequential()
    
    model.add(Conv2D(16, (5, 5), padding = "same", input_shape = input_shape, activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Conv2D(16, (3, 3), padding = "same", activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Conv2D(32, (3, 3), padding = "same", activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model

def model_M2():
    input_shape = (32, 32, 3)
    model = Sequential()
    
    model.add(Conv2D(32, (5, 5), padding = "same", input_shape = input_shape, activation = 'relu'))
    model.add(Conv2D(32, (5, 5), padding = "same", activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Conv2D(64, (3, 3), padding = "same", activation = 'relu'))
    model.add(Conv2D(64, (3, 3), padding = "same", activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    return model
   
  
def model_M3():
    input_shape = (32, 32, 3)
    model = Sequential()
    
    model.add(Conv2D(32, (5, 5), padding = "same", input_shape = input_shape, activation = 'relu'))
    model.add(Conv2D(32, (5, 5), padding = "same", activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Conv2D(64, (5, 5), padding = "same", activation = 'relu'))
    model.add(Conv2D(64, (5, 5), padding = "same", activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Conv2D(128, (3, 3), padding = "same", activation = 'relu'))
    model.add(Conv2D(128, (3, 3), padding = "same", activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    return model
  
  
def model_M4():
    input_shape = (32, 32, 3)
    model = Sequential()
    
    model.add(Conv2D(32, (5, 5), padding = "same", input_shape = input_shape, activation = 'relu'))
    model.add(Conv2D(32, (5, 5), padding = "same", activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Conv2D(64, (5, 5), padding = "same", activation = 'relu'))
    model.add(Conv2D(64, (5, 5), padding = "same", activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Conv2D(64, (3, 3), padding = "same", activation = 'relu'))
    model.add(Conv2D(128, (3, 3), padding = "same", activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    return model
  


def model_M5():
    input_shape = (32, 32, 3)
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), padding = "same", input_shape = input_shape, activation = 'relu'))
    model.add(Conv2D(32, (3, 3), padding = "same", activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Conv2D(64, (3, 3), padding = "same", activation = 'relu'))
    model.add(Conv2D(64, (3, 3), padding = "same", activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    return model
  

def model_M6():
    input_shape = (32, 32, 3)
    model = Sequential()
    
    model.add(Conv2D(64, (3, 3), padding = "same", input_shape = input_shape, activation = 'relu'))
    model.add(Conv2D(64, (3, 3), padding = "same", activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Conv2D(128, (3, 3), padding = "same", activation = 'relu'))
    model.add(Conv2D(128, (3, 3), padding = "same", activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    return model
