from glob import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras import optimizers, regularizers, metrics, regularizers, models, layers, utils
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import ntpath
from keras.applications import VGG16
from PIL import Image 
from random import shuffle
from sklearn.preprocessing import OneHotEncoder
import os

trainCSV=pd.read_csv("train.csv")

train_path = 'train_images/train/'
valid_path = 'train_images/valid/'
test_path = 'train_images/test/'

all_training_files = glob("train_images/*.png")
print(len(all_training_files))

# ??????
shuffles = np.random.permutation(all_training_files)

if len(all_training_files) > 0:
    testInd = int(np.floor(len(all_training_files)*0.85))
    validInd = int(np.floor(len(all_training_files)*0.7))
    for i in range(0, validInd):
        file=ntpath.basename(shuffles[i])
        label=trainCSV.loc[trainCSV['ID'] == file, 'Label'].iloc[0]
        newpath=shuffles[i].replace('train_images', train_path+str(label))
        os.rename(shuffles[i], newpath)
    for i in range(validInd, testInd):
        file=ntpath.basename(shuffles[i])
        label=trainCSV.loc[trainCSV['ID'] == file, 'Label'].iloc[0]
        newpath=shuffles[i].replace('train_images', valid_path+str(label))
        os.rename(shuffles[i], newpath)
    for i in range(testInd,len(all_training_files)):
        file=ntpath.basename(shuffles[i])
        label=trainCSV.loc[trainCSV['ID'] == file, 'Label'].iloc[0]
        newpath=shuffles[i].replace('train_images', test_path+str(label))
        os.rename(shuffles[i], newpath)