import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2

import time
import os
import glob

import keras
from keras.callbacks import History 
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Input

class Compute_time:
  def __init__(self):
    self.start_time = time.time()
    self.born_time  = time.time()
    
  def show(self):
    end_time = time.time()
    interval = int(end_time - self.start_time)
    interval2 = int(end_time - self.born_time)
    print("[excution time :  %ih , %im , %is]" % (interval//3600, (interval//60)%60, (interval)%60))
    print("[total    time :  %ih , %im , %is]" % (interval2//3600, (interval2//60)%60, (interval2)%60))
    self.start_time = end_time
  
t = Compute_time()

file_list = glob.glob('./20bn-jester-v1/*')
print('Number of sequence :', len(file_list))
t.show()

#################################
### Download the 20bn dataset ###
#################################

local_download_path = os.path.expanduser('~/data')

notHaveSeq = False if os.path.isdir('./20bn-jester-v1') else True
if notHaveSeq:
  
  # 1. Authenticate and create the PyDrive client.
  auth.authenticate_user()
  gauth = GoogleAuth()
  gauth.credentials = GoogleCredentials.get_application_default()
  drive = GoogleDrive(gauth)
  # 2. Auto-iterate using the query syntax
  #    https://developers.google.com/drive/v2/web/search-parameters

  file_list = drive.ListFile({'q': "'14gTk-1VguHxKPq9hP_bFJRJg8E_4HEnM' in parents"}).GetList()
  # choose a local (colab) directory to store the data.
  
  # 3. Create & download by id.
  try:
    os.makedirs(local_download_path)
  except: 
    pass
  
  for f in file_list:
    # 3. Create & download by id.
#     print('title: %s, id: %s' % (f['title'], f['id']))
    fname = os.path.join(local_download_path, f['title'])
    print('downloading to {}'.format(fname))
    f_ = drive.CreateFile({'id': f['id']})
    f_.GetContentFile(fname)

  # Unzip
  t.show()

###########################
### Unzip 20bn data set ###
###########################
notHave20bn = False if os.path.isdir('./20bn-jester-v1') else True
if notHave20bn:
  ! cat /content/data/20bn-jester-v1-?? | tar zx
t.show()

##########################
### Download npy_data  ###
##########################
notHave_npy = False if os.path.isdir('./npy_data') else True

if notHave_npy :
  # 1. Authenticate and create the PyDrive client.
  auth.authenticate_user()
  gauth = GoogleAuth()
  gauth.credentials = GoogleCredentials.get_application_default()
  drive = GoogleDrive(gauth)
  # 2. Auto-iterate using the query syntax
  #    https://developers.google.com/drive/v2/web/search-parameters

  file_list = drive.ListFile({'q': "'1UXHKf8dR8e7-7otI2CJI-X_S2Kw6J0Xs' in parents"}).GetList()
  # choose a local (colab) directory to store the data.
 
  # 3. Create & download by id.

  for f in file_list:
    # 3. Create & download by id.
#     print('title: %s, id: %s' % (f['title'], f['id']))
    fname = os.path.join('./', f['title'])
    print('downloading to {}'.format(fname))
    f_ = drive.CreateFile({'id': f['id']})
    f_.GetContentFile(fname)
    ! ls

!unzip -o npy_data_3-0.zip
!unzip -o npy_data_3-1.zip
!unzip -o npy_data_3-2.zip

#

file_list = glob.glob('./20bn-jester-v1/*')
print('Number of sequence :', len(file_list))

file_list = glob.glob('./npy_data/*')
print('Number of sequence :', len(file_list))

######################################
### Download all .csv and .py file ###
######################################

notHave_CSV_PY = False if os.path.isfile('./Process_Data.py') else True

if notHave_CSV_PY :
  
  # 1. Authenticate and create the PyDrive client.
  auth.authenticate_user()
  gauth = GoogleAuth()
  gauth.credentials = GoogleCredentials.get_application_default()
  drive = GoogleDrive(gauth)
  # 2. Auto-iterate using the query syntax
  #    https://developers.google.com/drive/v2/web/search-parameters

  file_list = drive.ListFile({'q': "'18axPRqbSQwhZcdCtcVCAy0h4diqbtRIw' in parents"}).GetList()
  # choose a local (colab) directory to store the data.
 
  # 3. Create & download by id.

  for f in file_list:
    # 3. Create & download by id.
#     print('title: %s, id: %s' % (f['title'], f['id']))
    fname = os.path.join('./', f['title'])
    print('downloading to {}'.format(fname))
    f_ = drive.CreateFile({'id': f['id']})
    f_.GetContentFile(fname)

file_list = sorted(glob.glob('./20bn-jester-v1/57033/*.jpg'))
print(file_list)

class Extractor():
  
    def __init__(self, weights=None):
        """Either load pretrained from imagenet, or load our saved
        weights from our own training."""

        self.weights = weights  # so we can check elsewhere which model

        if weights is None:
            # Get model with pretrained weights.
            base_model = InceptionV3(
                weights='imagenet',
                include_top=True
            )

            # We'll extract features at the final pool layer.
            self.model = Model(
                inputs=base_model.input,
                outputs=base_model.get_layer('avg_pool').output
            )

        else:
            # Load the model first.
            self.model = load_model(weights)

            # Then remove the top so we get features not predictions.
            # From: https://github.com/fchollet/keras/issues/2371
            self.model.layers.pop()
            self.model.layers.pop()  # two pops to get to pool layer
            self.model.outputs = [self.model.layers[-1].output]
            self.model.output_layers = [self.model.layers[-1]]
            self.model.layers[-1].outbound_nodes = []
            
    def extract_seq(self, seq_path, downsample_size=0):
        seq = []
        for img_src in sorted(glob.glob(seq_path + '/*.jpg'),key=lambda x:int(os.path.split(x)[-1].split('.')[0])):
            img = image.load_img(img_src, target_size=(299, 299))
            x = image.img_to_array(img)
            seq.append(x)
        seq = np.array(seq)
        if downsample_size > 0:
            step_size = (len(seq)-1) // (downsample_size-1)
            sample_index = [i*step_size for i in range(downsample_size)]
            seq = [seq[i,] for i in sample_index]
        seq = np.array(seq).astype('float32')
        
        seq = preprocess_input(seq)
        # Get the prediction
#         print(seq.shape)
        features_seq = self.model.predict(seq)
        
        return np.array(features_seq)

    def extract(self, image_path):
        img = image.load_img(image_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Get the prediction.
        features = self.model.predict(x)

        if self.weights is None:
            # For imagenet/default network:
            features = features[0]
        else:
            # For loaded network:
            features = features[0]

        return features

class DataGenerator(keras.utils.Sequence):
    #'Generates data for Keras'
    def __init__(self, list_IDs, labels, num_batch_per_epoch=0, batch_size=32, 
                 dim=(100,100,30), n_channels=3, n_classes=27, shuffle=True,
                extractor=None, seq_depth=20):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.extractor = extractor
        self.seq_depth = seq_depth
        self.num_batch_per_epoch = num_batch_per_epoch
        self.current_path = './'
        self.data_dir = '20bn-jester-v1'
        
        ## 6/14 ##
        if not os.path.isdir('./npy_data'):
            os.makedirs('./npy_data')
        
    def set_path(self, current_path='./', data_dir='20bn-jester-v1'):
        self.current_path = current_path
        self.data_dir = data_dir
        
    def set_num_batch_per_epoch(self, num_batch_per_epoch):
        self.num_batch_per_epoch = num_batch_per_epoch
        
    def __len__(self):
        #'Denotes the number of batches per epoch'
        if self.num_batch_per_epoch == 0:
          return int(np.floor(len(self.list_IDs) / self.batch_size))
        else:
          return self.num_batch_per_epoch

    def __getitem__(self, index):
        #'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        #'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        #'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            ## 6/14 ##
            file_name = './npy_data/' + str(ID)+ '.npy'
            if not os.path.isfile(file_name):
              
                # Store sample
                data_dir = '20bn-jester-v1'
                dir_path = os.path.join(self.current_path, self.data_dir)
                dir_path = os.path.join(dir_path, str(ID))
                
#                 print(dir_path)
                
                feature_seq = self.extractor.extract_seq(dir_path, downsample_size=self.seq_depth)
                
                np.save(file_name, feature_seq)
                
                
            else :
                feature_seq = np.load(file_name)
         
            X[i,] = feature_seq
            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

class Preprocessing():
    
    def __init__(self, select_labels=None, isTest=False):
        
        self.train_df      = pd.read_csv('jester-v1-train.csv', sep=';', header=None, names = ["Index", "Label"])
        self.validation_df = pd.read_csv('jester-v1-validation.csv', sep=';', header=None, names = ["Index", "Label"])
        
        # 6/20
#         file_list = glob.glob('./npy_data/*')
#         file_list = [ file_list[i].split('/')[-1].split('.')[0] for i in range(len(file_list))]
#         self.train_df = self.train_df[self.train_df.Index.isin(file_list)]
#         self.validation_df = self.validation_df[self.validation_df.Index.isin(file_list)]
        
        self.label_df      = pd.read_csv('jester-v1-labels.csv',header=None, names = ["Label"])
        self.select_labels = self.label_df['Label'].values
        self.isTest = isTest
        
        if not select_labels == None:
            self.select_labels = select_labels
            self.train_df = self.train_df.loc[self.train_df['Label'].isin(select_labels)]
            self.validation_df = self.validation_df.loc[self.validation_df['Label'].isin(select_labels)]
            self.label_df = self.label_df.loc[self.label_df['Label'].isin(select_labels)]
    
    def partition_labels(self):
        train_df = self.train_df
        validation_df = self.validation_df
        label = self.label_df
        partition = { 'train':np.sort(train_df['Index'].values), 'validation': np.sort(validation_df['Index'].values)}  
        label_onehot= {}
        label_onehot_reverse = {}
        labels = {}
        for s in label['Label'].values:
            ## Add up ##
            value = 0 if len(label_onehot) == 0 else max(list(label_onehot.values()))+1
            label_onehot[s] = value
            label_onehot_reverse[value] = s
        for index in partition['train']:
            i = train_df.loc[train_df['Index']==index].index[0]
            lab = train_df.at[i,'Label']
            labels[index] = label_onehot[lab]
        for index in partition['validation']:
            i = validation_df.loc[validation_df['Index']==index].index[0]
            lab = validation_df.at[i,'Label']
            labels[index] = label_onehot[lab]
            
        self.label_onehot = label_onehot_reverse
        
        return (partition, labels)
    
    def preprocess_input(self, img_seq, downsample_size=0):
        img_seq = img_seq.astype('float32') / 255.0
        if not downsample_size==0:
            img_seq = self.downsample(img_seq, num_imgs_per_seq=downsample_size, from_end=False)
        return img_seq
    
    def prediction_to_gesture(self, prediction):
        value = np.argmax(prediction)
        return self.label_onehot[value]
    
    def get_label_onehot(self):
        return self.label_onehot
        
    ### Functions for image sequence downsample/read/wrtie/ ###
    def downsample(self, seq, num_imgs_per_seq, from_end=False):
        step_size = (len(seq)-1)// (num_imgs_per_seq-1)
        sample_index = [i*step_size for i in range(num_imgs_per_seq)]
        if from_end: 
            sample_index.reverse()
        seq_downsample = [seq[i,] for i in sample_index]
        return np.array(seq_downsample)
    
    def read_img_seq(self, dir_path, range_0_1=False):
        seq = []
        for img_src in sorted(glob.glob(dir_path + '/*.jpg'),key=lambda x:int(os.path.split(x)[-1].split('.')[0])):
            img = cv2.imread(img_src)
            img = cv2.resize(img, (100,100))
            if range_0_1 == True: 
                img = img.astype('float32') / 255.0
            seq.append(img)
        return np.array(seq)
    
    def img_seq_to_npy(self, save_path, img_seq, ID, depth_after=False):
        if depth_after == True:
            img_seq = np.rollaxis(np.array(img_seq), 0, 3)
        file_name = os.path.join(save_path, str(ID)+'.npy')
        np.save(file_name, img_seq)
        
    # Compute Optical Flow
    def computeOpticalFlow(self, seq):
        prev, next = [], []
        flow_seq = []
        for img in seq:
            next = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if not len(prev) == 0:
                flow = cv2.calcOpticalFlowFarneback(prev,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)
                vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
                horz = horz.astype('uint8')
                vert = vert.astype('uint8')
                max_ = np.maximum(horz, vert)
                flow_seq.append([horz, vert, max_])
            prev = next
        return np.array(flow_seq)
        
    
    def seq_to_npy(self, seq_path='./data', save_path='./npy_data', num_imgs_per_seq = 30):
        indexs_training   = self.train_df['Index'].values
        indexs_validation = self.validation_df['Index'].values
            
        def data_to_npy(indexs, d, df, save_path, num_imgs_per_seq):
            total, progress, current = len(indexs), len(indexs)//10, 0
            if self.isTest:
                total, progress = 10, 1
            print('--Total %i img_seqs are going to be transformed to npy--' %total)
            for index in indexs:
                # print 
                if (current % progress == 0):
                    print("Start seq #%i, save to %s/%s.npy" %(current+1, save_path, str(index)))
                current += 1
                # test 
                if self.isTest and current >= 10:
                    break
                dir_path = os.path.join(d, str(index))
                seq = self.read_img_seq(dir_path, range_0_1=True)
                seq_downsample = self.downsample(seq, num_imgs_per_seq, from_end=False)
                self.img_seq_to_npy(save_path, seq_downsample, index, depth_after=True)
            print('Finish transfroming img_seq to npy\n')
                                  
        print("## Train ##")
        data_to_npy(indexs_training, seq_path, self.train_df, save_path, num_imgs_per_seq)
        print("## Validation ##")
        data_to_npy(indexs_validation, seq_path, self.validation_df, save_path, num_imgs_per_seq)

extractor = Extractor()
t.show()

# import Process_Data
# from Process_Data import DataGenerator
# from Process_Data import Preprocessing

# Use select label 

# 0~9
# select_labels_1 = ['Drumming Fingers', 'No gesture', 'Pulling Hand In', 'Shaking Hand', 'Sliding Two Fingers Down', 
#        'Sliding Two Fingers Up', 'Thumb Down', 'Thumb Up', 'Zooming In With Two Fingers', 'Zooming Out With Two Fingers']

# # 10 ~ 19
# select_labels_2 = ['Doing other things', 'Stop Sign', 'Zooming Out With Full Hand', 'Zooming In With Full Hand', 'Turning Hand Counterclockwise', 
#        'Turning Hand Clockwise', 'Rolling Hand Backward', 'Rolling Hand Forward', 'Pulling Two Fingers In', 'Pushing Two Fingers Away']

# ## 20 ~ 26
# select_labels_3 = ['Sliding Two Fingers Right', 'Sliding Two Fingers Left', 'Pushing Hand Away', 'Swiping Up', 'Swiping Down', 'Swiping Right', 'Swiping Left' ]

preprocessor = Preprocessing(select_labels=None)
partition, labels = preprocessor.partition_labels()

seq_path = './20bn-jester-v1'

seq_depth, features_length = 30, 2048
input_shape = (seq_depth, features_length)
num_classes  = len(set(labels.values()))

# Parameters
params = {'dim': (seq_depth, features_length), 'batch_size': 32,'n_classes': len(set(labels.values())),'n_channels': 1, 'shuffle': False,
         'extractor':extractor, 'seq_depth':seq_depth}
      
# # Data batch Generators
current_path = './'
training_generator   = DataGenerator(partition['train'], labels, **params)
training_generator.set_path(current_path)
validation_generator = DataGenerator(partition['validation'], labels, **params)
validation_generator.set_path(current_path)
t.show()

print(set(labels.values()))
print(len(labels))

# for i in range(len(training_generator)):
#     training_generator.__getitem__(i)
for i in range(len(validation_generator)):
    validation_generator.__getitem__(i)

# def upload_npy(file_name):
#   # Get Auth
#   auth.authenticate_user()
#   gauth = GoogleAuth()
#   gauth.credentials = GoogleCredentials.get_application_default()
#   drive = GoogleDrive(gauth)
#   fid = '1UXHKf8dR8e7-7otI2CJI-X_S2Kw6J0Xs'
#   t.show()

#   # Upload npy_data
#   f = drive.CreateFile({"parents": [{"kind": "drive#fileLink", "id": fid}]})
#   f.SetContentFile(file_name+'.zip')
#   f.Upload()
#   print('Uploaded file {}'.format(f.get('title')))
#   for file in glob.glob('npy_data/*.npy'):
#       os.remove(file)

# def to_npy(training_generator, validation_generator, split=3, s=0, label=1):
#     print('train # batches :\t', len(training_generator))
#     print('validation # batches :\t', len(validation_generator))
#     file_name = 'npy_data_'+ str(label) + '-' + str(s)
#     print(file_name)
#     train_start, train_end = (len(training_generator)//split) * s,   (len(training_generator)//split) * (s+1)
#     val_start, val_end     = (len(validation_generator)//split) * s, (len(validation_generator)//split) * (s+1)
#     for i in range(train_start, train_end):
#         training_generator.__getitem__(i)
#         if i % 100 == 0: 
#           print('start :', train_start, '\tnow :', i, ' \tend:', train_end )
      
#     for i in range(val_start, val_end):
#         validation_generator.__getitem__(i)
#         if i % 100 == 0: 
#           print('start :', val_start, '\tnow :', i, ' \tend:', val_end )
#     return file_name

# preprocessor_1 = Preprocessing(select_labels=select_labels_1)
# partition_1, _ = preprocessor_1.partition_labels()
# training_generator   = DataGenerator(partition_1['train'], labels, **params)
# validation_generator = DataGenerator(partition_1['validation'], labels, **params)
# t.show()

# label = 1
# split=3

# s = 0
# to_npy(training_generator, validation_generator, split=split, s=s, label=label)
# file_name = 'npy_data_1-0'
# !zip npy_data_1-0.zip npy_data/*
# upload_npy(file_name)

# s = 1
# to_npy(training_generator, validation_generator, split=split, s=s, label=label)
# file_name = 'npy_data_1-1'
# !zip npy_data_1-1.zip npy_data/*
# upload_npy(file_name)

# s = 2
# to_npy(training_generator, validation_generator, split=split, s=s, label=label)
# file_name = 'npy_data_1-2'
# !zip npy_data_1-2.zip npy_data/*
# upload_npy(file_name)

# t.show()

# preprocessor_3 = Preprocessing(select_labels=select_labels_3)
# partition_3 , _ = preprocessor_3.partition_labels()
# training_generator_3   = DataGenerator(partition_3['train'], labels, **params)
# validation_generator_3 = DataGenerator(partition_3['validation'], labels, **params)
# print(len(training_generator_3 ))

# ## Peek the shape of input data
# x_batch, y_batch = training_generator.__getitem__(1)
# print(x_batch.shape)
# print(y_batch.shape)

# index = 14
# prediction = y_batch[index]
# print( preprocessor.prediction_to_gesture(prediction) )

# t.show()

# for i in range(len(training_generator_3)):
#   x_batch, y_batch = training_generator.__getitem__(i)
  
  
# for i in range(len(validation_generator_3)):
#     x_batch, y_batch = training_generator.__getitem__(i)

# # zip npy_data
# !zip npy_data_3.zip npy_data/*

# # Get Auth
# auth.authenticate_user()
# gauth = GoogleAuth()
# gauth.credentials = GoogleCredentials.get_application_default()
# drive = GoogleDrive(gauth)
# fid = '10EER9B9k9CzyEdt1fjdt9caiBtGJ4r46'
# t.show()

# # Upload npy_data
# f = drive.CreateFile({"parents": [{"kind": "drive#fileLink", "id": fid}]})
# f.SetContentFile('npy_data_3.zip')
# f.Upload()
# print('Uploaded file {}'.format(f.get('title')))

# t.show()

#################################
### Save & Load label mapping ###
#################################

import pickle

## Save label_onehot result
def save_obj(obj, name ):
    if not os.path.isdir('./obj'):
      os.makedirs('./obj')
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
label_onehot = preprocessor.get_label_onehot()
print(label_onehot)

save_obj(label_onehot, 'CNN_RNN_10_onehot')

from google.colab import files
files.download('./obj/CNN_RNN_10_onehot.pkl')

# def load_obj(name):
#     with open('obj/' + name + '.pkl', 'rb') as f:
#         return pickle.load(f)
# label_onehot = load_obj('./C3D_10_onehot')

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras import backend as K
from keras.utils.vis_utils import plot_model

# Plot history
import matplotlib.pyplot as plt
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel('train')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='center right')
    plt.show()

from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D, MaxPooling2D)
from collections import deque
import sys


"""Build a simple LSTM network. We pass the extracted features from our CNN to this model predomenently."""

#########################
### Build LSTM model  ###
#########################

BATCH_SIZE = 32
TIME_STEPS = seq_depth
INPUT_SIZE = 2048

model = Sequential()
model.add(LSTM(2048, return_sequences=False, batch_input_shape=(None, TIME_STEPS, INPUT_SIZE) ,dropout=0.25))
model.add(Dense(4096, activation='relu'))
model.add(Dense(4096, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

t.show()

##############################
### Download trained model ###
##############################

to_load_model = False

if to_load_model :
  local_download_path = './'
  auth.authenticate_user()
  gauth = GoogleAuth()
  gauth.credentials = GoogleCredentials.get_application_default()
  drive = GoogleDrive(gauth)
  # 2. Auto-iterate using the query syntax
  #    https://developers.google.com/drive/v2/web/search-parameters

  file_list = drive.ListFile({'q': "'1CCSn7XB9QAUcm4aGJ58xI54b6fZB4Mrv' in parents"}).GetList()
  # choose a local (colab) directory to store the data.
  
  # 3. Create & download by id.
  try:
    os.makedirs(local_download_path)
  except: 
    pass
  
  for f in file_list:
    # 3. Create & download by id.
#     print('title: %s, id: %s' % (f['title'], f['id']))
    fname = os.path.join(local_download_path, f['title'])
    print('downloading to {}'.format(fname))
    f_ = drive.CreateFile({'id': f['id']})
    f_.GetContentFile(fname)
    
t.show()

!ls 
# for file in glob.glob('./*.h5'):
#     os.remove(file)

##################
### Load model ###
##################

from keras.models import load_model

model_name = "CNN-RNN_D10_C10.h5"
if to_load_model:
  loaded_model = load_model(model_name)
  model = loaded_model
  model.summary()

##########################################
### Train & Save model to google drive ###
##########################################

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
fid = '18arQkXeiMF9_NNuN6SnYIMwL7u3bHe-d'

def save_model(model, model_name="model",label_onehot={},history={}):
    # Save model to disk
    model_name = model_name + '.h5'
    model.save( model_name )

    # Get Auth
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)
    fid = '18arQkXeiMF9_NNuN6SnYIMwL7u3bHe-d'

    # Upload model h5 file
    f = drive.CreateFile({"parents": [{"kind": "drive#fileLink", "id": fid}]})
    f.SetContentFile(model_name)
    f.Upload()
    print('Uploaded file {}'.format(f.get('title')))

    if not len(label_onehot) == {}:
        save_obj(label_onehot, 'CNN-RNN_10_onehot')
        f = drive.CreateFile({"parents": [{"kind": "drive#fileLink", "id": fid}]})
        f.SetContentFile('./obj/CNN-RNN_10_onehot.pkl')
        f.Upload()
        print('Uploaded file {}'.format(f.get('title')))

    if not history == {}:
        save_obj(history, 'CNN-RNN-10_history')
        f = drive.CreateFile({"parents": [{"kind": "drive#fileLink", "id": fid}]})
        f.SetContentFile('./obj/CNN-RNN-10_history.pkl')
        f.Upload()
        print('Uploaded file {}'.format(f.get('title')))


def update_history(history_old, history_new):
    history_old.history['val_loss'] += history_new.history['val_loss']
    history_old.history['val_acc'] += history_new.history['val_acc']
    history_old.history['loss'] += history_new.history['loss']
    history_old.history['acc'] += history_new.history['acc']
    return history_old
  

def train_model(model=None, training_generator=None, validation_generator=None, optimizer=None, num_epochs=1, time_to_save=60, save_name='model', save_per_epoch=True):
    ## 6/15
    history = History()
    history.history={'val_loss':[], 'val_acc':[], 'loss':[], 'acc':[]}
    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999) if optimizer == None else optimizer
    for i in range(num_epochs):
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        _ = model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=1, verbose=1)
        
        history.history = {'val_loss': history.history['val_loss'] + _.history['val_loss'], 
                           'val_acc':  history.history['val_acc'] + _.history['val_acc'],
                           'loss': history.history['loss'] + _.history['loss'],
                           'acc': history.history['acc'] + _.history['acc'],}
        if save_per_epoch:
          
          
            model_name = save_name + '_epoch_' + str(i) + '.h5' 
            model.save( model_name )
            
            # Get Auth
            auth.authenticate_user()
            gauth = GoogleAuth()
            gauth.credentials = GoogleCredentials.get_application_default()
            drive = GoogleDrive(gauth)
            fid = '18arQkXeiMF9_NNuN6SnYIMwL7u3bHe-d'
            
            # Upload model epoch .h5 file
            f = drive.CreateFile({"parents": [{"kind": "drive#fileLink", "id": fid}]})
            f.SetContentFile(model_name)
            f.Upload()
            
            print('Uploaded file {}'.format(f.get('title')))
            
    # Upload model .h5 file
    model_name = save_name + '.h5'
    model.save( model_name )

    # Get Auth
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)
    fid = '18arQkXeiMF9_NNuN6SnYIMwL7u3bHe-d'
    f = drive.CreateFile({"parents": [{"kind": "drive#fileLink", "id": fid}]})
    f.SetContentFile(model_name)
    f.Upload()
    print('Uploaded file {}'.format(f.get('title')))
    
    # Upload model history pkl file
    save_obj(history.history, save_name + '_history')
    f = drive.CreateFile({"parents": [{"kind": "drive#fileLink", "id": fid}]})
    f.SetContentFile('./obj/' + save_name + '_history.pkl')
    f.Upload()
    print('Uploaded file {}'.format(f.get('title')))
    
    show_train_history(history, 'acc','val_acc')
    show_train_history(history, 'loss','val_loss')

    return history

history = History()
history.history={ 'val_loss':[], 'val_acc':[], 'loss':[], 'acc':[] }
t.show()

print('Total # img_seq : %i' %len(labels))
print('# Training img_seq : %i,\t # batchs_32 : %i' %(len(partition['train']), len(partition['train'])// 32))
print('# Validation img_seq : %i,\t # batchs_32 : %i' %(len(partition['validation']), len(partition['validation'])// 32))

from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath="model_weights.h5", verbose=1, save_best_only=True)

# num_batch_per_epoch = 0 represent whole set
training_generator   = DataGenerator(partition['train'], labels, num_batch_per_epoch = 0, **params)
validation_generator = DataGenerator(partition['validation'], labels, num_batch_per_epoch = 0, **params)

adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
sgd =  keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)


model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
_ = model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=10, verbose=1, callbacks=[checkpointer])

history = update_history(history, _)

show_train_history(history, 'acc','val_acc')
show_train_history(history, 'loss','val_loss')

t.show()

lr = 0.001
for i in range(3):
    sgd =  keras.optimizers.SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    _ = model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=10, verbose=1, callbacks=[checkpointer])
    history = update_history(history, _)
    lr *= 0.1
show_train_history(history, 'acc','val_acc')
show_train_history(history, 'loss','val_loss')

t.show()

save_model(model, model_name='CNN-RNN_D30_C27.h5', history=history)


# save best weight
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
fid = '18arQkXeiMF9_NNuN6SnYIMwL7u3bHe-d'
# Upload model h5 file
f = drive.CreateFile({"parents": [{"kind": "drive#fileLink", "id": fid}]})
f.SetContentFile('model_weights.h5') # best_weight
f.Upload()
print('Uploaded file {}'.format(f.get('title')))

# memory footprint support libraries/code
!ln -sf /opt/bin/nvidia-smi /usr/bin/nvidia-smi
!pip install gputil
!pip install psutil
!pip install humanize
import psutil
import humanize
import GPUtil as GPU
GPUs = GPU.getGPUs()
# XXX: only one GPU on Colab and isn’t guaranteed
gpu = GPUs[0]

def printm():
  process = psutil.Process(os.getpid())
  print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " I Proc size: " + humanize.naturalsize( process.memory_info().rss))
  print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
print('-'*100)
printm()
# Disk
print('-'*100)
! df -h
print('-'*100)
! ls

print(len(glob.glob('./npy_data/*.npy')))

# zip npy_data
# !zip npy_data.zip npy_data/*

# # Get Auth
# auth.authenticate_user()
# gauth = GoogleAuth()
# gauth.credentials = GoogleCredentials.get_application_default()
# drive = GoogleDrive(gauth)
# fid = '1XVaGqGqjRW32ah4iSdPNTGjGmNwBrPkK'
# t.show()

# # Upload npy_data
# f = drive.CreateFile({"parents": [{"kind": "drive#fileLink", "id": fid}]})
# f.SetContentFile('npy_data.zip')
# f.Upload()
# print('Uploaded file {}'.format(f.get('title')))

# t.show()

!ls

