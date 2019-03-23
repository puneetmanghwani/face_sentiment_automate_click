#!/usr/bin/python3

import cv2
import face_sentiment.py as fs
import numpy as np
i=0
cap=cv2.VideoCapture(0)

while cap.isOpened():

    status,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('camera',frame)
    frame_gray=cv2.resize(gray,(48,48))
    frame_gray=frame_gray.reshape(48,48,1)
    prediction=fs.predict(frame_gray)
    if np.argmax(prediction)==3:
    	cv2.imwrite('/home/punit/pic/pic{}.jpg'.format(i), frame)
        i+=1 

    if cv2.waitKey(1) & 0xFF == ord('q') :
        
        break

cap.release()
cv2.destroyAllWindows()
