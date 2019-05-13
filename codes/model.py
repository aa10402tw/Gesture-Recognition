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



