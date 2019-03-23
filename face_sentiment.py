#!/usr/bin/env python


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


dataset=pd.read_csv("../input/facial/fer2013.csv")


# In[ ]:


dataset.head()


# In[ ]:


dataset['pixels'][0]


# In[ ]:


training = dataset[dataset['Usage']=="Training"]
validation = dataset[dataset["Usage"]=="PublicTest"]
test = dataset[dataset["Usage"]=="PrivateTest"]


# In[ ]:


x_train =np.array([ np.fromstring(image, np.uint8, sep=" ").reshape(48,48) for image in training.pixels])
x_valid =np.array([ np.fromstring(image, np.uint8, sep=" ").reshape(48,48) for image in validation.pixels])
x_test =np.array([ np.fromstring(image, np.uint8, sep=" ").reshape(48,48) for image in test.pixels])


# In[ ]:


x_train = x_train.reshape((-1,48,48,1)).astype(np.float32)
x_valid = x_valid.reshape((-1,48,48,1)).astype(np.float32)
x_test = x_test.reshape((-1,48,48,1)).astype(np.float32)


# In[ ]:


x_train = x_train/255.
x_valid = x_valid/255.
x_test = x_test/255.


# In[ ]:


from keras.utils import to_categorical


# In[ ]:


y_train=training.emotion.values
y_valid=validation.emotion.values
y_test=test.emotion.values


# In[ ]:


np.unique(y_train)


# In[ ]:


y_type_train=to_categorical(y_train,7)
y_type_valid=to_categorical(y_valid,7)
y_type_test=to_categorical(y_test,7)


# In[ ]:


import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten


# In[ ]:


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


# In[ ]:


model.summary()


# In[ ]:


model.fit(x_train,y_type_train,batch_size=256,epochs=50, shuffle=True)


# In[ ]:


model.metrics_names


# In[ ]:


model.evaluate(x_valid,y_type_valid)


# In[ ]:


from keras.models import model_from_json


# In[ ]:


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)


# In[ ]:


model.save_weights("model.h5")


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


predictions = model.predict_classes(x_test)


# In[ ]:


y_type_test.shape


# In[ ]:


predictions.shape


# In[ ]:


print(classification_report(y_test,predictions))


# In[ ]:


x_test[0].shape


# In[ ]:


import cv2


# In[ ]:


frame_sad=cv2.imread('../input/testpic1/testpic1.jpg')
gray_sad=cv2.cvtColor(frame_sad,cv2.COLOR_BGR2GRAY)


# In[ ]:


face_cascade = cv2.CascadeClassifier('../input/haarcascade/haarcascade_frontalface_default.xml')


# In[ ]:


face_rects_sad = face_cascade.detectMultiScale(gray_sad,scaleFactor=1.2) 


# In[ ]:


for (x,y,w,h) in face_rects_sad:
    imgcrop_sad = gray_sad[y:y+h,x:x+w]


# In[ ]:


imgcrop_sad=imgcrop_sad/255


# In[ ]:


plt.imshow(imgcrop_sad)


# In[ ]:


frame=cv2.resize(imgcrop_sad,(48,48))
frame1=frame.reshape(1,48,48,1)


# In[ ]:


predict_value=model.predict(frame1)


# In[ ]:


predict_value


# In[ ]:


np.argmax(predict_value)


# In[ ]:


frame_happy=cv2.imread('../input/testpic2/testpic2.jpg')
gray_happy=cv2.cvtColor(frame_happy,cv2.COLOR_BGR2GRAY)


# In[ ]:


face_rects_happy = face_cascade.detectMultiScale(gray_happy,scaleFactor=1.2) 


# In[ ]:


for (x,y,w,h) in face_rects_happy:
    imgcrop_happy = gray_happy[y:y+h,x:x+w]


# In[ ]:


imgcrop_happy=imgcrop_happy/255


# In[ ]:


plt.imshow(imgcrop_happy)


# In[ ]:





# In[ ]:


frame_1=cv2.resize(imgcrop_happy,(48,48))
frame_1=frame_1.reshape(1,48,48,1)


# In[ ]:


predict_value1=model.predict(frame_1)


# In[ ]:


predict_value1


# In[ ]:


np.argmax(predict_value1)


# In[ ]:


np.argmax(predict_value)


# test_pic1=cv2.imread('../input/testingpic/pic_kaggle/pic.jpg')
# test_pic2=cv2.imread('../input/testingpic/pic_kaggle/pic1.jpg')
# test_pic3=cv2.imread('../input/testingpic/pic_kaggle/pic2.jpg')
# test_pic4=cv2.imread('../input/testingpic/pic_kaggle/pic3.jpg')
# test_pic5=cv2.imread('../input/testingpic/pic_kaggle/pic5.jpg')
# 

# gray1=cv2.cvtColor(test_pic1,cv2.COLOR_BGR2GRAY)
# gray2=cv2.cvtColor(test_pic2,cv2.COLOR_BGR2GRAY)
# gray3=cv2.cvtColor(test_pic3,cv2.COLOR_BGR2GRAY)
# gray4=cv2.cvtColor(test_pic4,cv2.COLOR_BGR2GRAY)
# gray5=cv2.cvtColor(test_pic5,cv2.COLOR_BGR2GRAY)

# face_rects1 = face_cascade.detectMultiScale(gray1,scaleFactor=1.2) 
# face_rects2 = face_cascade.detectMultiScale(gray2,scaleFactor=1.2) 
# face_rects3 = face_cascade.detectMultiScale(gray3,scaleFactor=1.2) 
# face_rects4 = face_cascade.detectMultiScale(gray4,scaleFactor=1.2) 
# face_rects5 = face_cascade.detectMultiScale(gray5,scaleFactor=1.2) 

# for (x,y,w,h) in face_rects5:
#     imgcrop5 = gray5[y:y+h,x:x+w]

# imgcrop5=imgcrop5/255

# plt.imshow(imgcrop5)

# frame_5=cv2.resize(imgcrop5,(48,48))
# frame_5=frame_5.reshape(1,48,48,1)
# 

# predict_value5=model.predict(frame_5)

# np.argmax(predict_value4)

# In[ ]:




