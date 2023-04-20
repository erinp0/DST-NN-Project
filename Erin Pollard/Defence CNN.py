#!/usr/bin/env python
# coding: utf-8

# # Comparing the performance with and without the bright-dark filter 

# In this notebook, we shall compare the performance on the basic model (ran on just one epoch) with images which have been resized against images which have been resized and had the bright-dark filter applied.

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


# In[2]:


NUM_CATEGORIES = 43


# ### Model trained on resized and bright-dark filter images

# Below the bright_dark filter is defined. It increases the brightness of dark images and darkens those which are too bright (perhaps due to camera flash). Note, the pixel values are normalised.

# In[3]:


bd_training_data = np.load('filter_denoised .npy')
black_patch_data = np.load('black_patched_denoised.npy')
white_patch_data = np.load('white_patched_denoised.npy')
noisy_data = np.load('noisy_train_denoised.npy')
same_patch_data = np.load('same_image_patched_denoised.npy')
bd_training_label = np.load('32_filter_training_labels.npy')
black_patch_label = np.load('32_filter_training_labels.npy')


# In[4]:


training_data = np.concatenate((bd_training_data,noisy_data,black_patch_data,white_patch_data,same_patch_data),axis=0,dtype='float32')
training_label = np.concatenate((bd_training_label,black_patch_label,black_patch_label,black_patch_label,black_patch_label),axis=0,dtype='float32')




# In[6]:


ones = np.ones(117627,dtype='float32')
zeros = np.zeros(78418,dtype='float32')
adv_label = np.concatenate((zeros,ones),axis=0)


# Note, the bright-dark filter data has already been normalised to between 0-1.

# In[7]:


X_train2, X_val2, y_train2, y_val2 = train_test_split(training_data, training_label, test_size=0.3, random_state=42, shuffle=True)
X_adv_train2, X_adv_val2, y_discard2, y_discard3 = train_test_split(adv_label, training_label, test_size=0.3, random_state=42, shuffle=True)


# In[8]:


y_train2 = keras.utils.to_categorical(y_train2, NUM_CATEGORIES)
y_val2 = keras.utils.to_categorical(y_val2, NUM_CATEGORIES)


# In[9]:


import tensorflow as tf
from tensorflow import keras

input1 = keras.layers.Input(shape=(32, 32, 3), name='pixel_input')
input2 = keras.layers.Input(shape=(1,), name='aux_input')
    
conv1 = keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu')(input1)
conv2 = keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')(conv1)
pool1 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv2)
bn1 = keras.layers.BatchNormalization(axis=-1)(pool1)
    
conv3 = keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu')(bn1)
conv4 = keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu')(conv3)
pool2 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv4)
bn2 = keras.layers.BatchNormalization(axis=-1)(pool2)
    
flatten = keras.layers.Flatten()(bn2)

merged = keras.layers.concatenate([flatten, input2])

dense1 = keras.layers.Dense(512, activation='relu')(merged)
bn3 = keras.layers.BatchNormalization()(dense1)
dropout = keras.layers.Dropout(rate=0.5)(bn3)

output = keras.layers.Dense(43, activation='softmax')(dropout)

model = keras.models.Model(inputs=[input1, input2], outputs=output)


# In[10]:


lr = 0.001
epochs = 200

opt = tf.keras.optimizers.legacy.Adam(lr=lr, decay=lr / (epochs * 0.5))
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


# In[11]:


aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode="nearest")
#history2 = model.fit([pixel_data, aux_data], labels, epochs=10, batch_size=32)
history2 = model.fit(aug.flow([X_train2,X_adv_train2], y_train2, batch_size=32), epochs=epochs, validation_data=([X_val2,X_adv_val2], y_val2))


# In[13]:


model.save('DEFENCE_cnn_200.h5')


# In[ ]:




