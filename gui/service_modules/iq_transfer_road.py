import socket, threading
import numpy as np
from collections import deque
import argparse
from math import pi
import tensorflow as tf
# import matplotlib.pyplot as plt
import copy
import os
import time
from datetime import timedelta
from models import VGG_branch

# HOST = 'localhost'
PORT = 19960
DATALENGTH = 600
FRAMENUM = 1
INITIALDATA = None
FLAG = True
DATASHAPE = 167
DATA_CNT = 0

# label2idx_Dict = {
#                 'empty' : 0,
#                 'can' : 1,
#                 'paper' : 2,
#                 'glass' : 3,
#                 'plastic' : 4,
#             }

# idx2label_Dict = {
#     0 : 'empty',
#     1 : 'can',
#     2 : 'paper',
#     3 : 'glass',
#     4 : 'plastic',
# }
label2idx_Dict = {
                # 'empty' : 0,
                'can' : 0,
                'paper' : 1,
                'glass' : 2,
                'plastic' : 3,
            }

idx2label_Dict = {
    # 0 : 'empty',
    0 : 'can',
    1 : 'paper',
    2 : 'glass',
    3 : 'plastic',
}

def setInitialData(data_queue):
    global FLAG
    data = list()
    cnt = 0
    global INITIALDATA
    while cnt < FRAMENUM:
        if data_queue:
            temp = data_queue.pop()
            if temp is not None:
                data.append(temp)
                cnt += 1
    data = np.array(data)
    INITIALDATA = np.mean(data, axis = 0)
    print('Create Initial Data')
    FLAG = False

def preprocess_data(data_queue, feature_queue):
    try:
        global INITIALDATA
        # global DATASHAPE
        print('preprocess_data')
        while True:
            if INITIALDATA is not None:
                feature_data = list()
                cnt = 0
                while cnt < FRAMENUM:
                    if data_queue:
                        # print('cnt', cnt)
                        temp = data_queue.pop()
                        if temp is not None:
                            feature_data.append(temp)
                            cnt += 1
                feature_mean_data = np.mean(feature_data, axis=0)
                # DATASHAPE = feature_mean_data.shape[2]
                # print(feature_mean_data.shape)
                feature_queue.appendleft(feature_mean_data - INITIALDATA)
    except:
        pass

def binder(client_socket, addr, queue):
    print('Connected by', addr)
    msg = 'OK'
    msg = msg.encode()
    msg_len = len(msg)
    global FLAG
    try:
        while True:
            leng = client_socket.recv(4)
            length = int.from_bytes(leng, "little")
            stringData = client_socket.recv(length)
            data = np.frombuffer(stringData, dtype=np.cdouble)
            if queue:
                pass
            else:
                if FLAG:
                    queue.appendleft(data)
                else:
                    queue.clear()
            client_socket.sendall(msg_len.to_bytes(4, byteorder="little"))
            client_socket.sendall(msg)
            # print(data)
    except:
        queue.appendleft('end')
        print('except : ', addr)
    finally:
        client_socket.close()

def saver(data_len, queue):
    global FLAG
    global DATA_CNT
    
    while True:
        if INITIALDATA is not None:
            data = list()
            file_num = 0
            # file_name = input('file name (ex : train/can/can1)')
            file_name = str(input('File name : '))
            print(file_name)
            start = time.process_time()
            FLAG = True
            DATA_CNT = 0
            while DATA_CNT < data_len:
                if queue:
                    temp = queue.pop()
                    data.append(temp)
                    DATA_CNT += 1
                    print(len(data))
                    if len(data) % 200 == 0:
                        data = np.array(data)
                        data_path = os.path.join('./road_data', (file_name + '.npy'))
                        if not os.path.exists(os.path.dirname(data_path)):
                            os.makedirs(os.path.dirname(data_path))
                        np.save('./road_data/{}{}.npy'.format(file_name, file_num), data)
                        print('save : {}{}.npy'.format(file_name, file_num))
                        data = list()
                        file_num += 1
                else:
                    pass
            FLAG = False
            end = time.process_time()
            print("Time elapsed : ", end - start)
            print("Hz : ", (600/(end - start)))
            # data = np.array(data)
            # data_path =os.path.join('./road_data', (file_name + '.npy'))
            # if not os.path.exists(os.path.dirname(data_path)):
            #     os.makedirs(os.path.dirname(data_path))
            # np.save('./data/{}.npy'.format(file_name), data)
            # print('save : ' + file_name)
            break

def seperater(data):
    pre_data = np.array(copy.deepcopy(data))
    amp = np.abs(pre_data)
    phs = np.angle(pre_data)
    sin = np.sin(phs)
    sin = (sin + 1) / 2
    amp = np.expand_dims(amp, axis = 0)
    sin = np.expand_dims(sin ,axis = 0)
    seperated_data = np.concatenate([amp, sin], axis = 0)
    return np.array(seperated_data)

def prediction(queue, model_path):
    global FLAG
    global DATASHAPE
    print("model load start")
    model = VGG_branch(DATASHAPE)
    model.load_weights(model_path)
    print("complete model load")
    # try:
    while True:
        # try:
        FLAG = True
        if queue:
            print('predict')
            pre_data = queue.pop()
            data = seperater(pre_data)
            FLAG = False
            data = np.expand_dims(data, axis=0)
            print(data.shape)
            start = time.process_time()
            predict = model.predict(data, verbose = 0)
            end = time.process_time()
            predict_dec = np.argmax(predict)
            text = idx2label_Dict[predict_dec]
            print('object : ',text)
            print('spend time : ', end - start)
        # except:
        #     print('error')

def main(data_queue, feature_queue, save, model_path, predict):

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('', PORT))
    print('wating...')
    try:
        th4 = threading.Thread(target = preprocess_data, args = (data_queue, feature_queue))
        th4.start()
        th5 = threading.Thread(target = setInitialData, args = (data_queue,))
        th5.start()
        if predict:
            print('start predict')
            # model = tf.keras.models.load_model(model_path)
            # print('model load complete')
            th3 = threading.Thread(target = prediction, args = (feature_queue, model_path))
            th3.start()
        if save:
            print('start save')
            # th2 = threading.Thread(target = saver,args = (DATALENGTH, feature_queue, save[1]) )
            th2 = threading.Thread(target = saver,args = (DATALENGTH, data_queue))
            th2.start()
        while True:
            server_socket.listen()
            client_socket, addr = None, None
            client_socket, addr = server_socket.accept()
            th1 = threading.Thread(target = binder, args = (client_socket, addr, data_queue))
            th1.start()
    except:
        print('error')
    finally:
        server_socket.close()
    try :
        th1.join()
        th2.join()
        th3.join()
        th4.join()
        th5.join()
    except:
        pass

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Radar data saver and predictor')
    parser.add_argument('-s','--save', action = 'store_true')
    # parser.add_argument('-n','--name', action = 'store')
    parser.add_argument('-p', '--predict', action='store_true')
    args = parser.parse_args()
    data_queue = deque()
    feature_queue = deque()
    model_path = '/home/pi/Intelligent_Radar-master/gui/service_modules/model/branchweight.h5'
    main(
        data_queue, 
        feature_queue, 
        # save = [args.save, args.name], 
        save = args.save,
        model_path = model_path,
        predict = args.predict
    )
    
