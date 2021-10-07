
from google.colab import drive
drive.mount('/content/gdrive')

!ls '/content/gdrive/My Drive/imagedosyalari'


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

!ls '/content/gdrive/My Drive/imagedosyalari'

root = '/content/gdrive/My Drive/imagedosyalari/'

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

df.head()

df["Usage"].value_counts()

image = df["pixels"][99] # 99th image
val = image.split(" ")
x_pixels = np.array(val, 'float32')
x_pixels /= 255
x_reshaped = x_pixels.reshape(48,48)
plt.imshow(x_reshaped, cmap= "gray", interpolation="nearest")
plt.axis("off")

# Consistent results
np.random.seed(1337)

# The dimension of z
noise_dim = 100

batch_size = 16
steps_per_epoch = 312 # 50000 / 16
epochs = 800

save_path = 'dcgan-images'



optimizer = Adam(0.0002, 0.5)

# Create path for saving images
if save_path != None and not os.path.isdir(save_path):
    os.mkdir(save_path)

def create_generator():
    generator = Sequential()
    
    d = 4
    generator.add(Dense(d*d*256, kernel_initializer=RandomNormal(0, 0.02), input_dim=noise_dim))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Reshape((d, d, 256)))
    
    generator.add(Conv2DTranspose(128, (4, 4), strides=2, padding='same', kernel_initializer=RandomNormal(0, 0.02)))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Conv2DTranspose(128, (4, 4), strides=3, padding='same', kernel_initializer=RandomNormal(0, 0.02)))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Conv2DTranspose(128, (4, 4), strides=2, padding='same', kernel_initializer=RandomNormal(0, 0.02)))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Conv2D(channels, (3, 3), padding='same', activation='tanh', kernel_initializer=RandomNormal(0, 0.02)))
    
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return generator

def create_discriminator():
    discriminator = Sequential()
    
    discriminator.add(Conv2D(64, (3, 3), padding='same', kernel_initializer=RandomNormal(0, 0.02), input_shape=(48, 48, 1)))
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Conv2D(128, (3, 3), strides=2, padding='same', kernel_initializer=RandomNormal(0, 0.02)))
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Conv2D(128, (3, 3), strides=2, padding='same', kernel_initializer=RandomNormal(0, 0.02)))
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Conv2D(256, (3, 3), strides=2, padding='same', kernel_initializer=RandomNormal(0, 0.02)))
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Flatten())
    discriminator.add(Dropout(0.4))
    discriminator.add(Dense(1, activation='sigmoid', input_shape=(48, 48, 1)))
    
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return discriminator

discriminator = create_discriminator()
generator = create_generator()
generator.summary()
print("**************************")
discriminator.summary()

# Make the discriminator untrainable when we are training the generator.  This doesn't effect the discriminator by itself
def create_gan(discriminator, generator):
  discriminator.trainable = False

# Link the two models to create the GAN
  gan_input = Input(shape=(noise_dim,))
  print(gan_input)
  fake_image = generator(gan_input)
  print(fake_image)
  gan_output = discriminator(fake_image)

  gan = Model(gan_input, gan_output)
  gan.compile(loss='binary_crossentropy', optimizer=optimizer)
  return gan

gan = create_gan(discriminator, generator)

def show_images(noise, epoch=None):
    generated_images = generator.predict(noise)
    plt.figure(figsize=(10, 10))
    
    for i, image in enumerate(generated_images):
        plt.subplot(10, 10, i+1)
        if channels == 1:
            plt.imshow(np.clip(image.reshape((img_rows, img_cols)), 0.0, 1.0), cmap='gray')
        else:
            plt.imshow(np.clip(image.reshape((img_rows, img_cols, channels)), 0.0, 1.0))
        plt.axis('off')
    
    plt.tight_layout()
    
    if epoch != None and save_path != None:
        plt.savefig(f'{save_path}/gan-images_epoch-{epoch}.png')
    plt.show()

static_noise = np.random.normal(0, 1, size=(100, noise_dim))

#generator.load_weights((root + 'gans_model_son_model.h5'))

def show(images, n_cols=None):
    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1
    if images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)
    plt.figure(figsize=(n_cols, n_rows))
    for index, image in enumerate(images):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(image, cmap="gray")
        plt.axis("off")


# Training loop
for epoch in range(epochs):
    for batch in range(steps_per_epoch):
        noise = np.random.normal(0, 1, size=(batch_size, noise_dim))
        real_x = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

        fake_x = generator.predict(noise)

        x = np.concatenate((real_x, fake_x))

        disc_y = np.zeros(2*batch_size)
        disc_y[:batch_size] = 0.9

        d_loss = discriminator.train_on_batch(x, disc_y)

        y_gen = np.ones(batch_size)
        g_loss = gan.train_on_batch(noise, y_gen)

    print(f'Epoch: {epoch} \t Discriminator Loss: {d_loss} \t\t Generator Loss: {g_loss}')
    if epoch % 10 == 0:
        show_images(static_noise, epoch)

#generator.save_weights(root + 'gans_model_son_model.h5')


generated_images = generator.predict(static_noise)
generated_images = generated_images.reshape(100,48,48)
plt.imshow(generated_images[6], cmap='gray') #62
show(generated_images,15)

#objects = ('kızgın', 'nefret', 'korku', 'mutlu', 'üzgün', 'şaşırma', 'doğal') #y_test
#anomali seç aralardan
plt.grid(False)
plt.axis('off')
plt.show()