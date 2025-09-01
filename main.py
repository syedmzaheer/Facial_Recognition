#Import standard dependencies
import cv2
import os
import uuid
import random
import numpy as np
from matplotlib import pyplot as plt

#Import tensorflow dependencies - FUnctional API
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, MaxPooling2D, Layer
import tensorflow as tf

#Avoid OOM errors by setting GPU Memory Consumpktion Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#Setup paths
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

#Make thedirectories
os.makedirs(POS_PATH)
os.makedirs(NEG_PATH)
os.makedirs(ANC_PATH)

#https://vis-www.cs.umass.edu/lfw/

#Uncompress Tar GZ Labbled Faces in the Wild Dataset
!tar -xf lfw.tgz

#Move LFW Images to the following repository data/negetive
for directory in os.listdir('lfw'):
    for file in os.listdir(os.path.join('lfw', directory)):
        EX_PATH = os.path.join('lfw', directory, file)
        NEW_PATH = os.path.join(NEG_PATH, file)
        os.replace(EX_PATH, NEW_PATH)

#Import uuid library to generate unique image names
import uuid

os.pathj.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))

#Establish a connection to the webcam
cap = cv2.VideoCapture(4)
while cap.isOpened():
    ret, frame = cap.read()
    
    #Cut down fram to 250x250px
    frame = frame[120:120+250, 200:200+250, :]

    #Collect anchors
    if cv2.waitKey(1) & 0xFF == ord('a'):
        imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
        #Write out anchor image
        cv2.imwrite(imgname, frame)

    #Collect positives
    if cv2.waitKey(1) & 0xFF == ord('p'):
        imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
        #Write out positive image
        cv2.imwrite(imgname, frame)

    #Show the image back to the screen
    cv2.imshow('Image Collection', frame)

    #Break gracefully
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Release the webcam
cap.release()
cv2.destroyAllWindows()

plt.imshow(frame[120:120+250, 200:200+250, :])

def data_aug(img):
    data = []
    for i in range(9):
        img = tf.image.stateless_random_brightness(img, max_delta=0.02, seed=(1,2))
        img = tf.image.stateless_random_contrast(img, lower=0.6, upper=1, seed=(1,3))
        #img = tf.image.stateless_randon_crop(img, size=(20,20,3), seed=(1,2))
        img= tf.image.stateless_random_flip_left_right(img, seed=(np.random.randint(100), np.random.randit(100)))
        img = tf.image.stateless_random_jpeg_quality(img, min_jpeg_quality=90, max_jpeg_quality=100, seed=(np.random.randint(100), np.random.randint(100)))
        img = tf.image.stateless_random_saturation(img, lower=0.9, upper=1, seed=(np.random.randint(100), np.random.randint(100)))
        
        data.append(img)
    return data