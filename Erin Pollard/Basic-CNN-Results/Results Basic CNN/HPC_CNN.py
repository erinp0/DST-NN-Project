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


# If you would like to replicate this notebook, change the paths below to the file paths where the unedited train and test npy files are stored on your desktop. They can be downloaded from [here]()

# In[15]:


training_data = np.load('32_original_train_data.npy')
training_label = np.load('32_original_train_labels.npy')


# In[6]:


NUM_CATEGORIES = 43


# Next, the train validation split is performed. Note, the pixel values are normalised to be between 0-1.

# In[17]:


X_train, X_val, y_train, y_val = train_test_split(training_data, training_label, test_size=0.3, random_state=42, shuffle=True)

X_train = X_train/255 
X_val = X_val/255


# This next section of code converts the labels by one-hot encoding.

# In[18]:


y_train = keras.utils.to_categorical(y_train, NUM_CATEGORIES)
y_val = keras.utils.to_categorical(y_val, NUM_CATEGORIES)


# Here the structure of the model is specified, e.g., the number of layers, number of neurons per layer.

# In[19]:


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

# In[20]:


lr = 0.001
epochs = 100

opt = tf.keras.optimizers.legacy.Adam(lr=lr, decay=lr / (epochs * 0.5))
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


# Below the model is fitted, but not before data augmentation is applied.

# In[21]:


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


# In[ ]:


model.save('basic_cnn_model_original.h5')


# Here the test data is prepared and the model is used to make predictions.

# In[22]:


test_data = np.load('32_original_test_data.npy')
test_labels = np.load('32_original_test_labels.npy')


# The predictions are compared to the ground truth.

# In[24]:



X_test = test_data
X_test = X_test/255

pred = np.argmax(model.predict(X_test), axis=-1)

#Accuracy with the test data
print('Test Data accuracy: ',accuracy_score(test_labels, pred)*100)


# In[25]:


from sklearn.metrics import confusion_matrix
cf = confusion_matrix(test_labels, pred)


# In[26]:


from sklearn.metrics import classification_report

print(classification_report(test_labels, pred))


# ### Model trained on resized and bright-dark filter images

# Below the bright_dark filter is defined. It increases the brightness of dark images and darkens those which are too bright (perhaps due to camera flash). Note, the pixel values are normalised.

# In[2]:


bd_training_data = np.load('32_filter_training_data.npy')
bd_training_label = np.load('32_filter_training_labels.npy')


# Note, the bright-dark filter data has already been normalised to between 0-1.

# In[ ]:


X_train2, X_val2, y_train2, y_val2 = train_test_split(bd_training_data, bd_training_label, test_size=0.3, random_state=42, shuffle=True)


# In[7]:


y_train2 = keras.utils.to_categorical(y_train2, NUM_CATEGORIES)
y_val2 = keras.utils.to_categorical(y_val2, NUM_CATEGORIES)


# In[8]:


model2 = keras.models.Sequential([    
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


# In[9]:


lr = 0.001
epochs = 100

opt = tf.keras.optimizers.legacy.Adam(lr=lr, decay=lr / (epochs * 0.5))
model2.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


# In[10]:


aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode="nearest")

history2 = model2.fit(aug.flow(X_train2, y_train2, batch_size=32), epochs=epochs, validation_data=(X_val2, y_val2))


# In[ ]:


model2.save('basic_cnn_model_filter.h5')


# In[11]:


test_data = np.load('32_filter_test_data.npy')
test_labels = np.load('32_filter_test_label.npy')


# In[13]:



pred = np.argmax(model2.predict(test_data), axis=-1)

#Accuracy with the test data
print('Test Data accuracy: ',accuracy_score(test_labels, pred)*100)


# In[14]:


from sklearn.metrics import confusion_matrix
cf = confusion_matrix(test_labels, pred)


# ## Conclusion

# As we can see, the model trained on the image vectors which had the bright-dark filter applied first, performed better than the model trained on images without the filter (94.4% in comparison to 93.0% test set accuracy). We therefore have decided to use the image vectors with the filter applied going forward in our model.

# In[ ]:




