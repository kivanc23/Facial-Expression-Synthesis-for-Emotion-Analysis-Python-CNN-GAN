
from google.colab import drive
drive.mount('/content/gdrive')

!ls '/content/gdrive/My Drive/cnn-duygu-tanima'

!pip3 install -q keras

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report, confusion_matrix

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

root = '/content/gdrive/My Drive/resimler/'
root1 = '/content/gdrive/My Drive/cnn-duygu-tanima/images1'

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

#------------------------------
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

#------------------------------
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

x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
#x_train = x_train[np.where(y_train == 0)[0]].reshape((-1, img_rows, img_cols, channels))
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

##CNN MODELİMİZ

#------------------------------
CNN = Sequential()
CNN.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
CNN.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))
CNN.add(Conv2D(64, (3, 3), activation='relu'))
CNN.add(Conv2D(64, (3, 3), activation='relu'))
CNN.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))
CNN.add(Conv2D(128, (3, 3), activation='relu'))
CNN.add(Conv2D(128, (3, 3), activation='relu'))
CNN.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

CNN.add(Flatten())

CNN.add(Dense(1024, activation='relu'))
CNN.add(Dropout(0.2))
CNN.add(Dense(1024, activation='relu'))
CNN.add(Dropout(0.2))

CNN.add(Dense(num_classes, activation='softmax'))
#------------------------------
##VERİ ÇOKLUĞU İÇİN DATA AUGMENTATİON UYGULUYORUZ. DEFAULT VALUES İLE
gen = ImageDataGenerator()
train_generator = gen.flow(x_train, y_train, batch_size=batch_size)

#------------------------------

CNN.compile(loss='categorical_crossentropy'
    , optimizer=keras.optimizers.Adam()
    , metrics=['accuracy']
)

fit = True
#------------------------------

CNN.summary()

if fit == True:

  #Training the CNN model
  history = CNN.fit(train_generator,  epochs = 50)
	#CNN.fit_generator(x_train, y_train, epochs=epochs) #Tüm veri kümesi için eğit
	#CNN.fit_generator(train_generator, steps_per_epoch=batch_size, epochs=epochs) #rastgele bir eğtiim yap
else:
	CNN.load_weights(root + '/model_weights.h5') #Öğrenilmiş ağırlıkları yüklemek

## DRİVEDAN DOSYA ÇEKME
test_img_path = root1 + "/şaşkın.jpg"


img_orj = image.load_img(test_img_path)
img = image.load_img(test_img_path, grayscale=True, target_size=(48, 48))

x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)

x /= 255

custom = CNN.predict(x)
#Duygu Analizi(custom[0])


#1
objects = ('kızgın', 'nefret', 'korku', 'mutlu', 'üzgün', 'şaşırma', 'doğal')
y_pos = np.arange(len(objects))
    
plt.bar(y_pos, custom[0], align='center', alpha=0.5, color='g')
plt.xticks(y_pos, objects)
plt.ylabel('yüzde')
plt.title('duygu')
plt.show()

#2
x = np.array(x, 'float32')
x = x.reshape([48, 48]);
plt.axis('off')
plt.gray()
plt.imshow(img_orj)

plt.show()
#------------------------------

test_true = np.argmax(y_test, axis=1)
test_pred = np.argmax(CNN.predict(x_test), axis=1)
print("CNN Model Accuracy on test set: {:.4f}".format(accuracy_score(test_true, test_pred)))

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots(figsize=(12,6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

plot_confusion_matrix(test_true, test_pred, classes=class_names, normalize=True, title='Normalized confusion matrix')
plt.show()