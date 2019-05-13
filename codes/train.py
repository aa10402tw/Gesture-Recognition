from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras import backend as K
from keras.utils.vis_utils import plot_model

import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D, MaxPooling2D)
from collections import deque
import sys
from keras.models import load_model

from model import *

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel('train')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='center right')
    plt.show()

def save_model(model, model_name="model",label_onehot={},history={}):
    # Save model to disk
    model_name = model_name + '.h5'
    model.save( model_name )

    if not label_onehot == {}:
        save_obj(label_onehot, 'CNN-RNN_10_onehot')

    if not history == {}:
        save_obj(history, 'CNN-RNN-10_history')


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

    # Upload model .h5 file
    model_name = save_name + '.h5'
    model.save( model_name )

    return history

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

##################
### Load model ###
##################

from keras.models import load_model
to_load_model = False

model_name = "CNN-RNN_D10_C10.h5"
if to_load_model:
  loaded_model = load_model(model_name)
  model = loaded_model
  model.summary()


t = Compute_time()
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