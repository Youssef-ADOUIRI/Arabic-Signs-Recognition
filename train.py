import tensorflow as tf
from tensorflow.keras import layers, models , Sequential
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
import numpy as np
import random
import pandas as pd
import os


# all the arabic signs
CATEGORIES = ['ain', 'al', 'aleff', 'bb', 'dal', 'dha', 'dhad', 'fa', 
             'gaaf', 'ghain', 'ha', 'haa', 'jeem', 'kaaf', 'khaa', 'la', 
             'laam', 'meem', 'nun', 'ra', 'saad', 'seen', 'sheen', 'ta', 
             'taa', 'thaa', 'thal', 'toot', 'waw', 'ya', 'yaa', 'zay'] 


#training parameters
IMG_SIZE = 64
targetCount = len(CATEGORIES) #the arabic alphabet count : 32
BATCH_SIZE = 5
NB_EPOCHS = 1
path = 'data/ArASL'

#loading data
training=[]
#labels = pd.read_csv('data/ArSL_Data_Labels.csv')


def createTrainingData():
    for Class in os.listdir(path):
        Class_Path = os.path.join(path ,  Class)
        for img in os.listdir(Class_Path):
            #image
            img_array = cv2.imread(os.path.join(Class_Path , img))
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            print(Class,img , img_array.shape)
            #label
            training.append([new_array , CATEGORIES.index(Class)])
    print('Loading train_DATA is finshed...')
    return training


createTrainingData()

#Randomize data
random.shuffle(training)
print('Data is shuffeled...')

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

output = tf.keras.utils.to_categorical(output,targetCount)

X_train , X_test , Y_train , Y_test = train_test_split(input , output , test_size = 0.2, random_state = 4 )

#model creation 
model = Sequential(name='ARABIC_SIGNS')
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(IMG_SIZE, IMG_SIZE, 3) ) )
model.add(layers.MaxPooling2D(pool_size = (2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu' ,input_shape=(IMG_SIZE, IMG_SIZE, 3) ) )
model.add(layers.MaxPooling2D(pool_size = (2, 2)))
#model.add(layers.Conv2D(128, (3, 3), activation='relu' ,input_shape=(200, 200, 3) ) )
#model.add(layers.MaxPooling2D(pool_size = (2, 2)))

#classification layers 
model.add(layers.Flatten())
model.add(layers.Dense(512/2, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(targetCount, activation='softmax'))
print(model.summary())

#compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(X_train.shape , Y_train.shape)
print(X_test.shape , Y_test.shape)

history = model.fit(X_train , Y_train, batch_size = BATCH_SIZE, epochs = NB_EPOCHS)

#Evaluation
score = model.evaluate( X_test , Y_test, verbose = 1 )
print("Test Score: ", score[0])
print("Test accuracy: ", score[1])
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

#save the model
model.save('saved_model/ARS_model.h5')
print('finished')
'''
'''