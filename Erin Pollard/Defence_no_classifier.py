#!/usr/bin/env python
# coding: utf-8

# # Comparing the performance with and without the bright-dark filter 

# In this notebook, we shall compare the performance on the basic model (ran on just one epoch) with images which have been resized against images which have been resized and had the bright-dark filter applied.

# ### Model trained on resized images

# Below we load the necessary packages.

# In[1]:


import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
np.random.seed(42)

from matplotlib import style
style.use('fivethirtyeight')


# If you would like to replicate this notebook, change the paths below to the file paths where the unedited train and test npy files are stored on your desktop. They can be downloaded from [here](https://drive.google.com/drive/u/0/folders/1tg24731_oseORhm1v2AMW-mOtC2wW8IJ).

# In[2]:


bd_training_data = np.load('filter_denoised .npy')
black_patch_data = np.load('black_patched_denoised.npy')
white_patch_data = np.load('white_patched_training.npy')
noisy_data = np.load('noisy_train_denoised.npy')
same_patch_data = np.load('same_image_patched_denoised.npy')
bd_training_label = np.load('32_filter_training_labels.npy')
black_patch_label = np.load('32_filter_training_labels.npy')


# In[3]:


NUM_CATEGORIES = 43


# In[5]:


training_data = np.concatenate((bd_training_data,noisy_data,black_patch_data,white_patch_data,same_patch_data),axis=0,dtype='float32')
training_label = np.concatenate((bd_training_label,black_patch_label,black_patch_label,black_patch_label,black_patch_label),axis=0,dtype='float32')


# Next, the train validation split is performed. Note, the pixel values are normalised to be between 0-1 already.

# In[8]:


X_train, X_val, y_train, y_val = train_test_split(training_data, training_label, test_size=0.3, random_state=42, shuffle=True)


# This next section of code converts the labels by one-hot encoding.

# In[9]:


y_train = keras.utils.to_categorical(y_train, NUM_CATEGORIES)
y_val = keras.utils.to_categorical(y_val, NUM_CATEGORIES)


# Here the structure of the model is specified, e.g., the number of layers, number of neurons per layer.

# In[10]:


model = keras.models.Sequential([    
    keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(32,32,3)),
    keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.BatchNormalization(axis=-1),
    
    keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.BatchNormalization(axis=-1),
    
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(rate=0.5),
    
    keras.layers.Dense(43, activation='softmax')
])


# The optimiser is defined here, as well as the number of epochs (which is specified as 1 for time-saving purposes).

# In[11]:


lr = 0.001
epochs = 200

opt = tf.keras.optimizers.legacy.Adam(lr=lr, decay=lr / (epochs * 0.5))
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


# Below the model is fitted, but not before data augmentation is applied.

# In[12]:


aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode="nearest")

history = model.fit(aug.flow(X_train, y_train, batch_size=32), epochs=epochs, validation_data=(X_val, y_val))


# In[13]:


model.save('defence_no_classifier200.h5')


# In[ ]:




