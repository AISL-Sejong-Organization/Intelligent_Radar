import numpy as np
import tensorflow as tf
import os
import copy
from math import pi
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
from pandas import DataFrame as DF
from models import AP_ResNet, ResNet, VGG_branch, ResNetLSTM, ConvLSTM, ConvLSTM_dropout

from tensorflow import keras
import matplotlib.pyplot as plt
from classifier import preprocessing as prep
import os
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # 텐서플로가 첫 번째 GPU만 사용하도록 제한
  try:
    tf.config.experimental.set_visible_devices(gpus[2], 'GPU')
  except RuntimeError as e:
    # 프로그램 시작시에 접근 가능한 장치가 설정되어야만 합니다
    print(e)

class_names = ['asphalt','bicycle','block','floor','ground']
bound = 100

label2idx_Dict = {
                'asphalt' : 0,
                'bicycle' : 1,
                'block' : 2,
                'floor' : 3,
                'ground' : 4,
            }

idx2label_Dict = {
    0 : 'asphalt',
    1 : 'bicycle',
    2 : 'block',
    3 : 'floor',
    4 : 'ground',
}

dir_path = './road_data'
def readNpy(dir_path):
    class_num = len(idx2label_Dict)

    Asphalt = list()
    Bicycle = list()
    Block = list()
    Floor = list()
    Ground = list()
    Asphalt = np.array(Asphalt)
    Bicycle = np.array(Bicycle)
    Block = np.array(Block)
    Floor = np.array(Floor)
    Ground = np.array(Ground)
    

    for dir in os.listdir(dir_path):
        d_path = os.path.join(dir_path, dir)
        file_list = os.listdir(d_path)
        for file in file_list:
            file_path = os.path.join(d_path, file)
            if dir == idx2label_Dict[0] :
                if len(Asphalt) == 0:
                    Asphalt = np.load(file_path, allow_pickle=True)
                else :
                    Asphalt = np.append(Asphalt, np.load(file_path), axis = 0)
            elif dir == idx2label_Dict[1]:
                if len(Bicycle) == 0:
                    Bicycle = np.load(file_path, allow_pickle=True)
                else :
                    Bicycle = np.append(Bicycle, np.load(file_path), axis = 0)
            elif dir == idx2label_Dict[2]:
                if len(Block) == 0:
                    Block = np.load(file_path, allow_pickle=True)
                else:
                    Block = np.append(Block, np.load(file_path), axis = 0)
            elif dir == idx2label_Dict[3]:
                if len(Floor) == 0:
                    Floor = np.load(file_path, allow_pickle=True)
                else:
                    Floor = np.append(Floor, np.load(file_path), axis = 0)
            elif dir == idx2label_Dict[4]:
                if len(Ground) == 0:
                    Ground = np.load(file_path, allow_pickle=True)
                else:
                    Ground = np.append(Ground, np.load(file_path), axis = 0)

    bound = Asphalt.shape[1]

    Ground_label = np.full((Ground.shape[0], class_num), np.eye(len(label2idx_Dict))[label2idx_Dict['ground']])
    Asphalt_label = np.full((Asphalt.shape[0],class_num), np.eye(len(label2idx_Dict))[label2idx_Dict['asphalt']])
    Bicycle_label = np.full((Bicycle.shape[0],class_num), np.eye(len(label2idx_Dict))[label2idx_Dict['bicycle']])
    Block_label = np.full((Block.shape[0],class_num), np.eye(len(label2idx_Dict))[label2idx_Dict['block']])
    Floor_label = np.full((Floor.shape[0],class_num), np.eye(len(label2idx_Dict))[label2idx_Dict['floor']])

    Ground = np.concatenate((Ground, Ground_label), axis=1)
    Asphalt = np.concatenate((Asphalt, Asphalt_label), axis=1)
    Bicycle = np.concatenate((Bicycle, Bicycle_label), axis=1)
    Block = np.concatenate((Block, Block_label), axis=1)
    Floor = np.concatenate((Floor, Floor_label), axis=1)
    
    array = Asphalt
    array = np.append(array, Bicycle, axis = 0)
    array = np.append(array, Block, axis = 0)
    array = np.append(array, Floor, axis = 0)
    array = np.append(array, Ground, axis = 0)
    s = np.arange(array.shape[0])
    np.random.shuffle(s)
    array_s = array[s]

    X = array_s[:,:bound]
    Y = np.real(array_s[:,bound:])
    return copy.deepcopy(X), copy.deepcopy(Y)

X, y = readNpy(dir_path)

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

maximum = np.max(np.abs(X))
Split_X = dataSeperator(X)

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data[0])-seq_length-1):
        x = data[0][:][i:(i+seq_length)]
        y = data[1][i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 20
X, Y = create_sequences((Split_X, y), seq_length = seq_length)

def train(
    model, X, Y,
    test_X, test_Y, 
    batch_size = 64, history_dict = None
    ):
    Epoch = 300
    callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = './model/' + 'Resnet_LSTM',
        monitor='val_accuracy',
        mode='max',
        save_best_only = True,
        save_weigths_only = False,
    )
    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer = optimizer , loss = loss, 
                metrics = ['accuracy', 'categorical_crossentropy'])
    history = model.fit(X, Y,  batch_size = batch_size, epochs = Epoch,
                # callbacks = callback, 
                validation_data = (test_X, test_Y)
                # validation_split = 0.3
                )
    return history

X = X[:, :, 0, :]
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size = 0.3
)
# test_len = 100
# train_X, test_X = X[:-100], X[-100:]
# train_Y, test_Y = Y[:-100], Y[-100:]
# print(train_X.shape, test_X.shape, train_Y.shape, test_Y.shape)

cl_model = ConvLSTM_dropout()
history = train(cl_model, X_train, Y_train, X_test, Y_test, batch_size = 64)
print(np.max(history.history['val_accuracy']))

