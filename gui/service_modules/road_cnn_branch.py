import numpy as np
import tensorflow as tf
import os
import copy
from math import pi
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import argparse
from pandas import DataFrame as DF
from models import AP_ResNet, ResNet, VGG_branch, ResNetLSTM, ConvLSTM, ConvLSTM_dropout, ResNet_CNN, VGG_CNN
from tensorflow.keras.applications.vgg16 import VGG16
from branchmodels import VGG_branch

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from classifier import preprocessing as prep
import os
import gc
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"



# label2idx_Dict = {
#                 'asphalt' : 0,
#                 'bicycle' : 1,
#                 'block' : 2,
#                 'floor' : 3,
#                 'ground' : 4,
#             }

# idx2label_Dict = {
#     0 : 'asphalt',
#     1 : 'bicycle',
#     2 : 'block',
#     3 : 'floor',
#     4 : 'ground',
# }

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
                    Asphalt = np.append(Asphalt, np.load(file_path, allow_pickle=True), axis = 0)
            # elif dir == idx2label_Dict[1]:
            #     if len(Bicycle) == 0:
            #         Bicycle = np.load(file_path, allow_pickle=True)
            #     else :
            #         Bicycle = np.append(Bicycle, np.load(file_path, allow_pickle=True), axis = 0)
            elif dir == idx2label_Dict[1]:
                if len(Block) == 0:
                    Block = np.load(file_path, allow_pickle=True)
                else:
                    Block = np.append(Block, np.load(file_path, allow_pickle=True), axis = 0)
            elif dir == idx2label_Dict[2]:
                if len(Floor) == 0:
                    Floor = np.load(file_path, allow_pickle=True)
                else:
                    Floor = np.append(Floor, np.load(file_path, allow_pickle=True), axis = 0)
            elif dir == idx2label_Dict[3]:
                if len(Ground) == 0:
                    Ground = np.load(file_path, allow_pickle=True)
                else:
                    Ground = np.append(Ground, np.load(file_path, allow_pickle=True), axis = 0)

    bound = Asphalt.shape[1]

    Ground_label = np.full((Ground.shape[0], class_num), np.eye(len(label2idx_Dict))[label2idx_Dict['ground']])
    Asphalt_label = np.full((Asphalt.shape[0],class_num), np.eye(len(label2idx_Dict))[label2idx_Dict['asphalt']])
    # Bicycle_label = np.full((Bicycle.shape[0],class_num), np.eye(len(label2idx_Dict))[label2idx_Dict['bicycle']])
    Block_label = np.full((Block.shape[0],class_num), np.eye(len(label2idx_Dict))[label2idx_Dict['block']])
    Floor_label = np.full((Floor.shape[0],class_num), np.eye(len(label2idx_Dict))[label2idx_Dict['floor']])

    Ground = np.concatenate((Ground, Ground_label), axis=1)
    Asphalt = np.concatenate((Asphalt, Asphalt_label), axis=1)
    # Bicycle = np.concatenate((Bicycle, Bicycle_label), axis=1)
    Block = np.concatenate((Block, Block_label), axis=1)
    Floor = np.concatenate((Floor, Floor_label), axis=1)
    
    array = Asphalt
    # array = np.append(array, Bicycle, axis = 0)
    array = np.append(array, Block, axis = 0)
    array = np.append(array, Floor, axis = 0)
    array = np.append(array, Ground, axis = 0)
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

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data[0])-seq_length-1):
        x = data[0][:][i:(i+seq_length)]
        y = data[1][i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def train(
    model, X, Y,
    test_X, test_Y, 
    callback,
    batch_size = 16, history_dict = None,
    Epoch = 50,
    learning_rate = 1e-3,
    # devices = ['/gpu:0', '/gpu:1']
    ):

    # strategy = tf.distribute.MirroredStrategy(devices=devices)
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy()
    # with strategy.scope():
    model.compile(optimizer = optimizer , loss = loss, 
                metrics = ['accuracy', 'categorical_crossentropy'])
    history = model.fit(X, Y,  batch_size = batch_size, epochs = Epoch,
                # callbacks = callback, 
                validation_data = (test_X, test_Y),
                # validation_split = 0.3
                )
    return history

def main(model, file_name, devices = ['/gpu:0', '/gpu:1']):
    dir_path = './road_data'
    X, y = readNpy(dir_path)
    seq_length = 90
    class_names = ['asphalt','bicycle','block','floor','ground']
    bound = 208
    maximum = np.max(np.abs(X))
    Split_X = dataSeperator(X)
    # Amp_X = np.abs(X) / maximum
    X, Y = create_sequences((Split_X, y), seq_length = seq_length)


    x_train, x_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

    X_train = np.expand_dims(x_train, axis = -1)
    X_test = np.expand_dims(x_test, axis = -1)

    checkpoint_filepath = file_name
    callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = './model/' + checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only = True,
        save_weigths_only = False,
    )
    log_dir = './logs/fit/'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir, histogram_freq = 1)
    # model_resnet = tf.keras.applications.ResNet152V2(
    #     include_top = True,
    #     weights = None,
    #     classes = 4,
    #     classifier_activation = 'softmax',
    #     input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    # )
    

    resnet_history = train(model, X_train, Y_train, X_test, Y_test, 
                callback = callback, 
                # devices = devices
                )
    plot_name = './plot/' + str(file_name) + '.png'
    plt.subplot(4,1,1)
    plt.plot(resnet_history.history['val_accuracy'])
    plt.title('val_accuracy')
    plt.subplot(4,1,2)
    plt.plot(resnet_history.history['val_loss'])
    plt.title('val_loss')
    plt.subplot(4,1,3)
    plt.plot(resnet_history.history['accuracy'])
    plt.title('accuracy')
    plt.subplot(4,1,4)
    plt.plot(resnet_history.history['loss'])
    plt.title('loss')
    plt.savefig(plot_name, dpi = 300)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Radar data trainer')
    # parser.add_argument('-m','--model', action = 'store')
    parser.add_argument('-p','--pre', action = 'store_true')
    parser.add_argument('-n','--name', action = 'store')
    parser.add_argument('-g', '--gpu', action = 'store')
    parser.add_argumetn('-b', '--branch', action = 'store_true')
    args = parser.parse_args()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if gpus:
    # 텐서플로가 첫 번째 GPU에 1GB 메모리만 할당하도록 제한
        try:
            tf.config.experimental.set_visible_devices(gpus[int(args.gpu)], 'GPU')
        except RuntimeError as e:
            # 프로그램 시작시에 가상 장치가 설정되어야만 합니다
            print(e)
    label2idx_Dict = {
                'asphalt' : 0,
                # 'bicycle' : 1,
                'block' : 1,
                'floor' : 2,
                'ground' : 3,
            }

    idx2label_Dict = {
        0 : 'asphalt',
        # 1 : 'bicycle',
        1 : 'block',
        2 : 'floor',
        3 : 'ground',
    }

    # if args.model == 'resnet':
    #     if args.pre:
    #         model = ResNet_CNN(weight = 'imagenet')
    #     else:
    #         model = ResNet_CNN()
    # elif args.model == 'vgg': 
    #     if args.pre:
    #         model = VGG_CNN(weight = 'imagenet')
    #     else:
    #         model = VGG_CNN()

    file_name = args.name
    model = VGG_branch()
    # if args.gpu == '01':
    #     devices = ['/gpu:0', '/gpu:1']
    # if args.gpu == '23':
    #     devices = ['/gpu:2', '/gpu:3']
    main(
        # devices = devices, 
        model = model, file_name = file_name)




