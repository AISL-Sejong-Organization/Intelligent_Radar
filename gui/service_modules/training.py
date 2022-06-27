import numpy as np
import tensorflow as tf
import os
import copy
from math import pi
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from pandas import DataFrame as DF

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from classifier import preprocessing as prep

class_names = ['empty','can','paper','glass','plastic']

label2idx_Dict = {
                'empty' : 0,
                'can' : 1,
                'paper' : 2,
                'glass' : 3,
                'plastic' : 4,
            }

idx2label_Dict = {
    0 : 'empty',
    1 : 'can',
    2 : 'paper',
    3 : 'glass',
    4 : 'plastic',
}

dir_path = './data/train'
test_dir_path = './data/test'

def readNpy(dir_path):
    class_num = len(idx2label_Dict)

    Empty = list()
    Can = list()
    Paper = list()
    Glass = list()
    Plastic = list()
    Empty = np.array(Empty)
    Can = np.array(Can)
    Paper = np.array(Paper)
    Glass = np.array(Glass)
    Plastic = np.array(Plastic)

    for dir in os.listdir(dir_path):
        d_path = os.path.join(dir_path, dir)
        file_list = os.listdir(d_path)
        for file in file_list:
            file_path = os.path.join(d_path, file)
            if dir == 'can':
                if len(Can) == 0:
                    Can = np.load(file_path)
                else :
                    Can = np.append(Can, np.load(file_path), axis = 0)
            elif dir == 'paper':
                if len(Paper) == 0:
                    Paper = np.load(file_path)
                else :
                    Paper = np.append(Paper, np.load(file_path), axis = 0)
            elif dir == 'glass':
                if len(Glass) == 0:
                    Glass = np.load(file_path)
                else:
                    Glass = np.append(Glass, np.load(file_path), axis = 0)
            elif dir == 'plastic':
                if len(Plastic) == 0:
                    Plastic = np.load(file_path)
                else:
                    Plastic = np.append(Plastic, np.load(file_path), axis = 0)
            elif dir == 'empty':
                if len(Empty) == 0:
                    Empty = np.load(file_path)
                else:
                    Empty = np.append(Empty, np.load(file_path), axis = 0)
    bound = Can.shape[1]
    Empty_label = np.full((Empty.shape[0], class_num), np.eye(len(label2idx_Dict))[label2idx_Dict['empty']])
    Can_label = np.full((Can.shape[0],class_num), np.eye(len(label2idx_Dict))[label2idx_Dict['can']])
    Paper_label = np.full((Paper.shape[0],class_num), np.eye(len(label2idx_Dict))[label2idx_Dict['paper']])
    Glass_label = np.full((Glass.shape[0],class_num), np.eye(len(label2idx_Dict))[label2idx_Dict['glass']])
    Plastic_label = np.full((Plastic.shape[0],class_num), np.eye(len(label2idx_Dict))[label2idx_Dict['plastic']])

    Empty = np.concatenate((Empty, Empty_label), axis=1)
    Can = np.concatenate((Can, Can_label), axis=1)
    Paper = np.concatenate((Paper, Paper_label), axis=1)
    Glass = np.concatenate((Glass, Glass_label), axis=1)
    Plastic = np.concatenate((Plastic, Plastic_label), axis=1)
    array = Can
    array = np.append(array, Paper, axis = 0)
    array = np.append(array, Glass, axis = 0)
    array = np.append(array, Plastic, axis = 0)
    array = np.append(array, Empty, axis = 0)
    s = np.arange(array.shape[0])
    np.random.shuffle(s)
    array_s = array[s]

    X = array_s[:,:bound]
    Y = np.real(array_s[:,bound:])
    return copy.deepcopy(X), copy.deepcopy(Y)

def seperater(arr):
    global maximum
    pre_data = arr
    amp = np.abs(pre_data)
    amp = amp / maximum
    phs = np.angle(pre_data)
    # phs = (phs - (- pi)) / (pi - (- pi))
    sin = np.sin(phs)
    sin = (sin + 1) / 2
    seperated_data = np.stack((amp.T,sin.T), axis=0)
    seperated_data = np.expand_dims(seperated_data, axis=0)
    return np.array(seperated_data)

def dataSeperator(arr):
    temp = copy.deepcopy(seperater(arr[0]))
    for i in range(1, len(arr)):
        temp = np.concatenate((temp, seperater(arr[i])), axis=0)
    return temp
    
def VGG_branch(X, Y, test_X, test_Y, cp_filepath, EPOCH=100):
    
    amp_input = keras.Input(shape=(X.shape[2],1), name="amplitude")
    phs_input = keras.Input(shape=(X.shape[2],1), name='phase')
    
    amp_features = layers.Conv1D(64, (3), activation = 'relu', input_shape = (X.shape[2], 1), padding = 'same')(amp_input)
    amp_features = layers.Conv1D(64, (3), activation = 'relu', padding = 'same')(amp_features)
    amp_features = layers.MaxPool1D(2)(amp_features)
    amp_features = layers.Conv1D(128, (3), activation = 'relu', padding = 'same')(amp_features)
    amp_features = layers.Conv1D(128, (3), activation = 'relu', padding = 'same')(amp_features)
    amp_features = layers.MaxPool1D(2)(amp_features)
    amp_features = layers.Conv1D(256,(3), activation = 'relu', padding = 'same')(amp_features)
    amp_features = layers.Conv1D(256,(3), activation = 'relu', padding = 'same')(amp_features)
    amp_features = layers.Conv1D(256,(3), activation = 'relu', padding = 'same')(amp_features)
    amp_features = layers.Conv1D(256,(3), activation = 'relu', padding = 'same')(amp_features)
    amp_features = layers.MaxPool1D(2)(amp_features)
    amp_features = layers.Flatten()(amp_features)

    phs_features = layers.Conv1D(64, (3), activation = 'relu', input_shape = (X.shape[2], 1), padding = 'same')(phs_input)
    phs_features = layers.Conv1D(64, (3), activation = 'relu', padding = 'same')(phs_features)
    phs_features = layers.MaxPool1D(2)(phs_features)
    phs_features = layers.Conv1D(128, (3), activation = 'relu', padding = 'same')(phs_features)
    phs_features = layers.Conv1D(128, (3), activation = 'relu', padding = 'same')(phs_features)
    phs_features = layers.MaxPool1D(2)(phs_features)
    phs_features = layers.Conv1D(256,(3), activation = 'relu', padding = 'same')(phs_features)
    phs_features = layers.Conv1D(256,(3), activation = 'relu', padding = 'same')(phs_features)
    phs_features = layers.Conv1D(256,(3), activation = 'relu', padding = 'same')(phs_features)
    phs_features = layers.Conv1D(256,(3), activation = 'relu', padding = 'same')(phs_features)
    phs_features = layers.MaxPool1D(2)(phs_features)
    phs_features = layers.Flatten()(phs_features)

    x = layers.concatenate([amp_features, phs_features], axis = -1)
    x = layers.Dropout(0.8)(x)
    x = layers.Dense(4096, activation = 'relu')(x)
    x = layers.Dropout(0.8)(x)
    x = layers.Dense(4096, activation = 'relu')(x)

    material_output = layers.Dense(5, activation = 'softmax', name = 'material_output')(x)

    model = keras.Model(inputs = [amp_input, phs_input],
                        outputs = [material_output],)

    model.summary()
    keras.utils.plot_model(model, "./branced_model.png", show_shapes=True)
    model.compile(optimizer = 'adam', loss = tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    checkpoint_filepath = cp_filepath
    callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = './model/' + checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only = True,
        save_weigths_only = False,
    )
    log_dir = './logs/fit/'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir, histogram_freq = 1)

    return model.fit(
    {"amplitude": X[:,0,:], "phase": X[:,1,:]},
    {"material_output": Y},
    epochs=EPOCH,
    validation_data = ({"amplitude" : test_X[:,0,:], "phase": test_X[:,1,:]}, {"material_output" : test_Y}),
    callbacks = [callback, tensorboard_callback]
    )

def main():

    dir_path = './data/train'
    test_dir_path = './data/test'
    global maximum


    X, Y = readNpy(dir_path)

    test_X, test_Y= readNpy(test_dir_path)

    maximum = np.max(np.abs(X))

    Split_X_n = dataSeperator(X)
    test_Split_X_n = dataSeperator(test_X)

    VGG_branch_hist = VGG_branch(Split_X_n, Y, test_X = test_Split_X_n, test_Y = test_Y, EPOCH=100, cp_filepath='./model/branch_model')

    print(max(VGG_branch_hist.history['val_accuracy']))

if __name__ == '__main__':
    main()
