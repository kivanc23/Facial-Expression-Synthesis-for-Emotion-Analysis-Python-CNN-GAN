
from google.colab import drive
drive.mount('/content/gdrive')
!ls '/content/gdrive/My Drive/cnn-duygu-tanima'

import tensorflow as tf

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D,BatchNormalization,Conv2DTranspose
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape,UpSampling2D,ZeroPadding2D

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os

from keras import initializers
from keras.initializers import RandomNormal

from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from numpy import genfromtxt

root = '/content/gdrive/My Drive/imagedosyalari/'
root1 = '/content/gdrive/My Drive/cnn-duygu-tanima/images'

batch_size = 256
epochs = 5

##NORMALİZASYON İŞLEMLERİMİZ DATASET ÇEKME İŞLEMLERİ

num_classes = 7 #angry, disgust, fear, happy, sad, surprise, neutral - sinirli, iğrenmiş, korku, mutlu, üzgün, şaşkın, doğal
class_names = ['kızgın', 'nefret', 'korku', 'mutlu', 'üzgün', 'şaşırma', 'doğal']
with open(root + "fer2013.csv") as f:
    content = f.readlines()
lines = np.array(content)   

df = pd.read_csv(root + "fer2013.csv") #df olarak feri okuma

df.info() #df infosu


num_of_instances = lines.size
print("number of instances: ",num_of_instances)
print("instance length: ",len(lines[1].split(",")[1].split(" ")))

#------------------------------
# Eğitim seti ve test seti ilklendirme
x_train, y_train, x_test, y_test = [], [], [], []
# Test ve eğitim verisinin transfer edilmesi
for i in range(1,num_of_instances):
    
    emotion, img, usage = lines[i].split(",")
      
    val = img.split(" ")
        
    pixels = np.array(val, 'float32')
        
    emotion = keras.utils.to_categorical(emotion, num_classes)
    
    if 'Training' in usage:
        y_train.append(emotion)
        x_train.append(pixels)
    elif 'PublicTest' in usage:
        y_test.append(emotion)
        x_test.append(pixels)
# Eğtitim ve test kümelerinin diziye tranformasyonu
x_train = np.array(x_train, 'float32')
y_train = np.array(y_train, 'float32')
x_test = np.array(x_test, 'float32')
y_test = np.array(y_test, 'float32')

x_train /= 255 # [0, 1] aralığına normalize etme işlemi
x_test /= 255
img_rows, img_cols, channels = 48, 48, 1

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train.reshape(x_train.shape[0], 48, 48)
x_test = x_test.reshape(x_test.shape[0], 48, 48)
from skimage import data
from skimage.color import gray2rgb

x_train = gray2rgb(x_train)
x_test = gray2rgb(x_test)

print(x_train.shape, x_test.shape)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)

print(y_train.shape, y_test.shape)

import keras.backend as K
from keras.layers import Conv2D, MaxPool2D, Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.applications import VGG16
from keras.callbacks import ReduceLROnPlateau

base_model = VGG16(weights = 'imagenet', include_top=False, input_shape= (48,48,3), classes= 7)
base_model.summary()

# Freezing Layers
for layer in base_model.layers[:-4]:
    layer.trainable=False

# Building Model
model1=Sequential()
model1.add(base_model)
model1.add(Dropout(0.5))
model1.add(Flatten())
model1.add(BatchNormalization())
model1.add(Dense(32,kernel_initializer='he_uniform'))
model1.add(BatchNormalization())
model1.add(Activation('relu'))
model1.add(Dropout(0.5))
model1.add(Dense(32,kernel_initializer='he_uniform'))
model1.add(BatchNormalization())
model1.add(Activation('relu'))
model1.add(Dropout(0.5))
model1.add(Dense(32,kernel_initializer='he_uniform'))
model1.add(BatchNormalization())
model1.add(Activation('relu'))
model1.add(Dense(7,activation='softmax'))

#model1.summary()



def f1_score(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
  
METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),  
      tf.keras.metrics.AUC(name='auc'),
      f1_score,
        
]

from keras.callbacks import ModelCheckpoint,EarlyStopping
lrd = ReduceLROnPlateau(monitor = 'val_loss',patience = 20,verbose = 1,factor = 0.50, min_lr = 1e-10)
mcp = ModelCheckpoint('model.h5')
es = EarlyStopping(verbose=1, patience=20)

model1.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=METRICS)

#model1.load_weights(root1 + "tlmodel.h5")

history1=model1.fit(x_train,y_train,epochs = 60,verbose = 1,callbacks=[lrd,mcp,es])

model1.save_weights(root1 + "tlmodel.h5")