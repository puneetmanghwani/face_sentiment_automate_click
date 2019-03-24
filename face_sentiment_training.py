#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
%matplotlib inline

dataset=pd.read_csv("../input/facial/fer2013.csv")


dataset.head()


dataset['pixels'][0]



training = dataset[dataset['Usage']=="Training"]
evaluation = dataset[dataset["Usage"]=="PublicTest"]
test = dataset[dataset["Usage"]=="PrivateTest"]



x_train =np.array([ np.fromstring(image, np.uint8, sep=" ").reshape(48,48) for image in training.pixels])
x_valid =np.array([ np.fromstring(image, np.uint8, sep=" ").reshape(48,48) for image in evaluation.pixels])
x_test =np.array([ np.fromstring(image, np.uint8, sep=" ").reshape(48,48) for image in test.pixels])




x_train = x_train.reshape((-1,48,48,1)).astype(np.float32)
x_valid = x_valid.reshape((-1,48,48,1)).astype(np.float32)
x_test = x_test.reshape((-1,48,48,1)).astype(np.float32)



x_train = x_train/255.
x_valid = x_valid/255.
x_test = x_test/255.


from keras.utils import to_categorical


y_train=training.emotion.values
y_valid=evaluation.emotion.values
y_test=test.emotion.values


np.unique(y_train)



y_type_train=to_categorical(y_train,7)
y_type_valid=to_categorical(y_valid,7)
y_type_test=to_categorical(y_test,7)



import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten


with tf.device('/gpu:0'): 
    model = Sequential()


    model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=(48, 48,1), activation='relu',))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=(5,5),input_shape=(48, 48, 1), activation='relu',))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))
    
    model.add(Conv2D(filters=128, kernel_size=(5,5),input_shape=(48, 48, 1), activation='relu',))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))
    
    


    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.5))

    model.add(Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.5))


    model.add(Dense(7, activation='softmax'))


    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])


model.summary()


model.fit(x_train,y_type_train,batch_size=256,epochs=50, shuffle=True)


model.metrics_names


model.evaluate(x_valid,y_type_valid)


from keras.models import model_from_json


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)


model.save_weights("model.h5")


from sklearn.metrics import classification_report


predictions = model.predict_classes(x_test)



y_type_test.shape


predictions.shape



print(classification_report(y_test,predictions))


x_test[0].shape


import cv2


frame_sad=cv2.imread('../input/testpic1/testpic1.jpg')
gray_sad=cv2.cvtColor(frame_sad,cv2.COLOR_BGR2GRAY)


face_cascade = cv2.CascadeClassifier('../input/haarcascade/haarcascade_frontalface_default.xml')


face_rects_sad = face_cascade.detectMultiScale(gray_sad,scaleFactor=1.2) 


for (x,y,w,h) in face_rects_sad:
    imgcrop_sad = gray_sad[y:y+h,x:x+w]


imgcrop_sad=imgcrop_sad/255


plt.imshow(imgcrop_sad)



frame=cv2.resize(imgcrop_sad,(48,48))
frame1=frame.reshape(1,48,48,1)



predict_value=model.predict(frame1)


predict_value



np.argmax(predict_value)



frame_happy=cv2.imread('../input/testpic2/testpic2.jpg')
gray_happy=cv2.cvtColor(frame_happy,cv2.COLOR_BGR2GRAY)


face_rects_happy = face_cascade.detectMultiScale(gray_happy,scaleFactor=1.2) 



for (x,y,w,h) in face_rects_happy:
    imgcrop_happy = gray_happy[y:y+h,x:x+w]



imgcrop_happy=imgcrop_happy/255




plt.imshow(imgcrop_happy)




frame_1=cv2.resize(imgcrop_happy,(48,48))
frame_1=frame_1.reshape(1,48,48,1)




predict_value1=model.predict(frame_1)

predict_value1

np.argmax(predict_value1)

np.argmax(predict_value)
