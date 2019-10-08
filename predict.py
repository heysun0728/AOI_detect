import glob
import pandas as pd
import numpy as np
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
import csv

trainCSV=pd.read_csv("train.csv")

train_path = 'train_images/train/'
valid_path = 'train_images/valid/'
test_path = 'train_images/test/'
test_path2 = 'test_images/'

#os.system('mv -r *.png .')


model=models.load_model("model/VGG16_pretrain_all_2019_06_12_09_42_37.model")
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

test_generator2 = test_datagen.flow_from_directory(
        test_path2,
        target_size=(224, 224),
        class_mode='categorical',
	    batch_size=1,
	    shuffle=False)

predict = model.predict_generator(
    test_generator2,
    steps=test_generator2.samples//test_generator2.batch_size)



#print(predict)

i =0 
#f1 = open ("inFile","r") # open input file for reading
with open('output.csv', 'w') as csvfile:
	writer = csv.writer(csvfile)
	writer.writerow(['ID', 'Label'])
	#print(len(predict))
	while i < len(predict):
		#image, label = test_generator2._get_batches_of_transformed_samples(np.array([i]))
		image_name = test_generator2.filenames[i]
		result=np.where(predict[i] == np.amax(predict[i]))[0][0]
		writer.writerow([image_name, result])
		i=i+1
	print(i)
