import tensorflow as tf
from tensorflow.keras import layers, models , Sequential
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
#from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
import numpy as np
from PIL import Image
import random
import os

#preparing the data 
path_train = 'data/unaugmented/416/train'
path_test = 'data/unaugmented/416/test'
#CATEGORIES = ['alf','bae','jim','dal','hae'] # all the arabic signs
IMG_SIZE = 200

#training parameters 
batch_size = 16
nb_classes =4
nb_epochs = 5
img_rows, img_columns = 200, 200
img_channel = 3
nb_filters = 32
nb_pool = 2
nb_conv = 3


#loading data
training=[]
training_img = []
img_path_train = os.path.join(path_train,'images')
lab_path_train = os.path.join(path_train,'labels')
dir_label_train = os.listdir(lab_path_train)

testing=[]
testing_img = []
img_path_test = os.path.join(path_test,'images')
lab_path_test = os.path.join(path_test,'labels')
dir_label_test = os.listdir(lab_path_test)


DATA_LEN_TEST = len(dir_label_test)

def createTrainingData():
    i = 0
    for img in os.listdir(img_path_train):
        #image
        img_array = cv2.imread(os.path.join(img_path_train,img))
        
        print(img , ' ', img_array.shape())
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        #training_img.append(new_array)
        #label
        f_label = open(os.path.join(lab_path_train,dir_label_train[i]), 'r')
        line = f_label.readline()
        lab_array = line.split()
        y = int(lab_array[0])
        i += 1
        couple = [new_array , y ]
        training.append(couple)
    print('Loading train_DATA is finshed')

def createTestingData():
    i = 0
    for img in os.listdir(img_path_test):
        #image
        img_array = cv2.imread(os.path.join(img_path_test,img))
        
        print(img)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        #testing_img.append(new_array)
        #label
        f_label = open(os.path.join(lab_path_test,dir_label_test[i]), 'r')
        line = f_label.readline()
        lab_array = line.split()
        y = int(lab_array[0])
        i += 1
        couple = [new_array , y ]
        testing.append(couple)
    print('Loading testing_DATA is finshed')


createTrainingData()




#Randomize data
random.shuffle(training)

#Seprate features and labels
features = []
labels = []

for f , l in training:
    features.append(f)
    labels.append(l)

input = np.array(features).reshape(-1 , IMG_SIZE, IMG_SIZE , 3 )
output = np.array(labels)

#convert to flaot
input = input.astype('float32')
#converting value from [0,255] to [0,1]
input /= 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    im = cv2.cvtColor(input[i], cv2.COLOR_BGR2RGB)
    plt.imshow(im)
    plt.xlabel(output[i])
    
plt.show()

Y = tf.keras.utils.to_categorical(output)

#model = Sequential(layers.Conv2D(32,(3,3),padding='same',activation=tf.nn.sigmoid,input_shape=()))

# load test data
createTestingData()