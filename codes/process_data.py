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

from model import *

t = Compute_time()

file_list = glob.glob('./20bn-jester-v1/*')
print('Number of sequence :', len(file_list))
t.show()

extractor = Extractor()
t.show()

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

for i in range(len(validation_generator)):
    validation_generator.__getitem__(i)

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









