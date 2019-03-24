
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.models import model_from_json


model_file = open('model_sent.json', 'r')
model_json = model_file.read()
model_file.close()
model = model_from_json(model_json)
model.load_weights("model_sent.h5")


model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

i=0
face_cascade = cv2.CascadeClassifier('/home/punit/haarcascades/haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture('http://25.29.42.77:8080//video?x.mjpg')

while cap.isOpened():

    status,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('camera',frame)
    face_rects= face_cascade.detectMultiScale(gray,scaleFactor=1.2) 
    for (x,y,w,h) in face_rects:
        imgcrop= gray[y:y+h,x:x+w]
    imgcrop=imgcrop/255
    frame_gray=cv2.resize(imgcrop,(48,48))
    frame_gray=frame_gray.reshape(-1,48,48,1)
    prediction=model.predict(frame_gray)
    if np.argmax(prediction)==3:
    	cv2.imwrite('/home/punit/pic/pic{}.jpg'.format(i), frame)
    	i+=1 

    if cv2.waitKey(1) & 0xFF == ord('q') :
        
        break

cap.release()
cv2.destroyAllWindows()