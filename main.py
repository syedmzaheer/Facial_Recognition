#Import standard dependencies
import cv2
import os
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
POS_PATH = os.path.join