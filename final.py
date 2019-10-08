from glob import glob
import pandas as pd
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from sklearn.preprocessing import MinMaxScaler
from keras import optimizers, regularizers, metrics, regularizers, models, layers, utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D,MaxPooling2D,Activation,Flatten,Dense,Dropout
import tensorflow as tf
import ntpath
from keras.applications import VGG16, ResNet50
from PIL import Image 
from random import shuffle
from sklearn.preprocessing import OneHotEncoder
import os
from time import gmtime, strftime
import matplotlib.pyplot as plt

trainCSV=pd.read_csv("train.csv")

train_path = 'train_images/train/'
valid_path = 'train_images/valid/'
test_path = 'train_images/test/'

test_path2 = 'test_images/'

#os.system('mv -r *.png .')

        
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(
      train_path,
      target_size=(224, 224),
      batch_size=32,
      class_mode='categorical')
test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(
        valid_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')
        
#conv_base = V(weights="imagenet",include_top=False,input_shape=(224,224,3))
conv_base = VGG16(weights='imagenet', include_top=False,input_shape=(224,224,3))

model = models.Sequential()


model.add(conv_base)

'''
model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(6,activation='softmax'))
model.summary();
'''
'''
model.add(Flatten())  
model.add(Dense(32))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(6,activation='softmax'))
'''
model.add(Conv2D(32, (1, 1),activation="relu"))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64,(3, 3),activation="relu"))
#model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(32))
#model.add(Dense(64))

#model.add(Dropout(0.2))
model.add(Dense(64))

#model.add(Dense(256))
#model.add(Dense(512))
model.add(Dense(6,activation='softmax'))

#conv_base.trainable = False #

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])
              
history = model.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.samples//train_generator.batch_size,
      epochs=15,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples//validation_generator.batch_size
)
model.save('model/VGG16_pretrain_all_'+strftime("%Y_%m_%d_%H_%M_%S", gmtime())+".model")
print('model/VGG16_pretrain_all_'+strftime("%Y_%m_%d_%H_%M_%S", gmtime())+".model")


test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')
 
test_loss, test_acc = model.evaluate_generator(
	test_generator, 
	steps=test_generator.samples//test_generator.batch_size)

print('test acc:', test_acc)
