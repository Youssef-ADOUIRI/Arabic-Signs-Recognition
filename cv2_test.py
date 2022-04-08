import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models


model = models.load_model('saved_model/ARS_REC_model_gray.h5')

IMG_SIZE = 64

CATEGORIES = ['ain', 'al', 'aleff', 'bb', 'dal', 'dha', 'dhad', 'fa', 
             'gaaf', 'ghain', 'ha', 'haa', 'jeem', 'kaaf', 'khaa', 'la', 
             'laam', 'meem', 'nun', 'ra', 'saad', 'seen', 'sheen', 'ta', 
             'taa', 'thaa', 'thal', 'toot', 'waw', 'ya', 'yaa', 'zay']




def predict_imgs(image):
    l_img = [image]
    input = np.array(l_img).reshape(-1 , IMG_SIZE, IMG_SIZE , 1 )
    #convert to flaot
    input = input.astype('float32')
    #converting value from [0,255] to [0,1]
    input /= 255.0
    prediction = model.predict(input)
    i = np.argmax(prediction)
    return i



vid = cv2.VideoCapture(0)

while(vid.isOpened()):
    ret, frame = vid.read()
    cv2.rectangle(frame , (300,300) , (100,100), (0,255,0) , 0)
    crop_img = frame[100:300, 100:300]
    g_img = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', frame)
    new_array = cv2.resize(g_img, (IMG_SIZE, IMG_SIZE))
    i = predict_imgs(new_array)
    print('Prediction is : ' , CATEGORIES[i])
    # the 'q' button is for quitting
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



vid.release()
cv2.destroyAllWindows()