#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

import tensorflow.keras
import pandas as pd
import sklearn as sk
import tensorflow as tf
import platform


# In[2]:


print(f"Python Platform: {platform.platform()}")
print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tensorflow.keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")


# In[3]:


# Load the original training data 
import numpy as np
original_train = np.load('/Users/xinyu/Downloads/32_filter_training_data.npy')


# In[4]:


# Load the data with noise added
noisy_train = np.load('/Users/xinyu/Downloads/noisy_training (1).npy')


# In[5]:


# Split the noisy data into training and testing sets

n_original = original_train.shape[0]
n_train = int(n_original * 0.9)
n_test = n_original - n_train

x_train_noisy = noisy_train[:n_train]
x_train_original = original_train[:n_train]
x_test_noisy = noisy_train[n_train:]
x_test_original = original_train[n_train:]


# In[11]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import UpSampling2D


# In[12]:


# Encoder
input_shape = (32, 32, 3)
input_img = Input(shape=input_shape)
x = Conv2D(48, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
encoded = Conv2D(32, (1, 1), activation='relu', padding='same')(x)


# In[13]:


# LATENT SPACE
latentSize = (4,4,32)


# In[14]:


# Decoder
direct_input = Input(shape=latentSize)
x = Conv2D(192, (1, 1), activation='relu', padding='same')(direct_input)
x = UpSampling2D((2, 2))(x)
x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(48, (3, 3), activation='relu', padding='same')(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)


# In[15]:


# Define the model
encoder = Model(input_img, encoded)
decoder = Model(direct_input, decoded)
autoencoder = Model(input_img, decoder(encoded))



# In[16]:


# Compile the model
autoencoder.compile(optimizer='Adam', loss='binary_crossentropy')


# In[407]:


# Train the model
autoencoder.fit(x_train_noisy, x_train_original, epochs=200)


# In[408]:


# save the trained model
autoencoder.save('denoising_autoencoder.h5')

