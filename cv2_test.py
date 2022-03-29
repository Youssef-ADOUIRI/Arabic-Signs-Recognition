import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models

def import_model():
    return models.load_model('C:/Users/YOUSSEF/Documents/GitHub/Arab-Signs/saved_model/ARS_REC_model.h5')

model = import_model()

IMG_SIZE = 64

CATEGORIES = ['ain', 'al', 'aleff', 'bb', 'dal', 'dha', 'dhad', 'fa', 
             'gaaf', 'ghain', 'ha', 'haa', 'jeem', 'kaaf', 'khaa', 'la', 
             'laam', 'meem', 'nun', 'ra', 'saad', 'seen', 'sheen', 'ta', 
             'taa', 'thaa', 'thal', 'toot', 'waw', 'ya', 'yaa', 'zay'] 




def predict_imgs(image):
    l_img = [image]
    input = np.array(l_img).reshape(-1 , IMG_SIZE, IMG_SIZE , 3 )
    #convert to flaot
    input = input.astype('float32')
    #converting value from [0,255] to [0,1]
    input /= 255.0
    prediction = model.predict(input)
    i = np.argmax(prediction)
    return i


#temp 
img = cv2.imread('YA.JPG')
nimg = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
print(CATEGORIES[predict_imgs(nimg)])


vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 64*10)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 64*10)

while(True):
    ret, frame = vid.read()
    cv2.imshow('frame', frame)
    new_array = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    i = predict_imgs(new_array)
    print('Prediction is : ' , CATEGORIES[i])
    # the 'q' button is for quitting
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



vid.release()
cv2.destroyAllWindows()