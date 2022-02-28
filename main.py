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
targetCount = 28 #the arabic alphabet count
batch_size = 100
nb_classes =4
nb_epochs = 5
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
        
        print(img , ' ', img_array.shape)
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

# load test data
createTestingData()
#Randomize data
random.shuffle(testing)

#Seprate features and labels
features_ts = []
labels_ts = []
for f , l in testing:
    features_ts.append(f)
    labels_ts.append(l)

test_input = np.array(features_ts).reshape(-1 , IMG_SIZE, IMG_SIZE , 3 )
test_output = np.array(labels_ts)

#convert to flaot
test_input = test_input.astype('float32')
#converting value from [0,255] to [0,1]
test_input /= 255.0


Y = tf.keras.utils.to_categorical(output,targetCount)
Y_ts = tf.keras.utils.to_categorical(test_output,targetCount)

#model creation
model = Sequential(name='nn')
model.add(layers.Conv2D(32,(3,3),padding='same',activation=tf.nn.relu,input_shape=(200, 200, 3)))
model.add(layers.MaxPooling2D(pool_size = (2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(64, (3, 3), activation='relu' ,input_shape=(200, 200, 3) ) )
model.add(layers.MaxPooling2D(pool_size = (2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(128, (3, 3), activation='relu' ,input_shape=(200, 200, 3) ) )
model.add(layers.MaxPooling2D(pool_size = (2, 2)))
model.add(layers.Dropout(0.25))


#classification layers 
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(targetCount, activation='softmax'))
print(model.summary())

#compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(input.shape , Y.shape)
print(test_input.shape , Y_ts.shape)

history = model.fit(input, Y, batch_size = batch_size, epochs = 12, validation_data = (test_input, Y_ts))

score = model.evaluate(test_input, Y_ts, verbose = 1 )
print("Test Score: ", score[0])
print("Test accuracy: ", score[1])
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
print('finished')