import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models
import os



ROOT_DIR =os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join( ROOT_DIR , 'saved_model/ARS_REC_model_gray_v3.h5') 
model = models.load_model(path)

IMG_SIZE = 64

CATEGORIES = ['ain', 'al', 'aleff', 'bb', 'dal', 'dha', 'dhad', 'fa', 
             'gaaf', 'ghain', 'ha', 'haa', 'jeem', 'kaaf', 'khaa', 'la', 
             'laam', 'meem', 'nun', 'ra', 'saad', 'seen', 'sheen', 'ta', 
             'taa', 'thaa', 'thal', 'toot', 'waw', 'ya', 'yaa', 'zay']


def predict_img(image):
    g_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    l_img = [cv2.resize(g_img, (IMG_SIZE, IMG_SIZE))]
    input = np.array(l_img).reshape(-1 , IMG_SIZE, IMG_SIZE , 1 )
    #convert to flaot
    input = input.astype('float32')
    #converting value from [0,255] to [0,1]
    input /= 255.0
    prediction = model.predict(input)
    i = np.argmax(prediction)
    return i

