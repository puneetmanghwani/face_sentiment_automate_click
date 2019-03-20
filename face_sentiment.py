#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


dataset=pd.read_csv("fer2013.csv")


# In[8]:


dataset.head()


# In[9]:


dataset['pixels'][0]


# In[10]:


training = dataset[dataset['Usage']=="Training"]
validation = dataset[dataset["Usage"]=="PublicTest"]
test = dataset[dataset["Usage"]=="PrivateTest"]


# In[11]:


x_train =np.array([ np.fromstring(image, np.uint8, sep=" ").reshape(48,48) for image in training.pixels])
x_valid =np.array([ np.fromstring(image, np.uint8, sep=" ").reshape(48,48) for image in validation.pixels])
x_test =np.array([ np.fromstring(image, np.uint8, sep=" ").reshape(48,48) for image in test.pixels])


# In[12]:


x_train = x_train.reshape((-1,48,48,1)).astype(np.float32)
x_valid = x_valid.reshape((-1,48,48,1)).astype(np.float32)
x_test = x_test.reshape((-1,48,48,1)).astype(np.float32)


# In[13]:


x_train = x_train/255.
x_valid = x_valid/255.
x_test = x_test/255.


# In[14]:


from keras.utils import to_categorical


# In[15]:


y_train=training.emotion.values
y_valid=validation.emotion.values
y_test=test.emotion.values


# In[16]:


np.unique(y_train)


# In[17]:


y_type_train=to_categorical(y_train,7)
y_type_valid=to_categorical(y_valid,7)
y_type_test=to_categorical(y_test,7)


# In[18]:


import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten


# In[19]:


with tf.device('/gpu:0'): 
    model = Sequential()


    model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=(48, 48,1), activation='relu',))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.20))

    model.add(Conv2D(filters=128, kernel_size=(5,5),input_shape=(48, 48, 1), activation='relu',))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.20))


    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.5))

    model.add(Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.5))


    model.add(Dense(7, activation='softmax'))


    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])


# In[20]:


model.summary()


# In[21]:


model.fit(x_train,y_type_train,epochs=50)


# In[22]:


model.metrics_names


# In[23]:


model.evaluate(x_valid,y_type_valid)


# In[24]:


from sklearn.metrics import classification_report


# In[25]:


predictions = model.predict_classes(x_test)


# In[26]:


y_type_test.shape


# In[27]:


predictions.shape


# In[28]:


print(classification_report(y_test,predictions))


# In[29]:


x_test[0].shape


# In[33]:


import cv2


# In[62]:


frame=cv2.imread('12345.jpg')
gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


# In[65]:


frame=cv2.resize(gray,(48,48))
frame1=frame.reshape(1,48,48,1)


# In[69]:


predict_value=model.predict(frame1)


# In[70]:


predict_value


# In[ ]:




