import socket, threading
import numpy as np
from collections import deque
import argparse
from math import pi
import tensorflow as tf
# import matplotlib.pyplot as plt
import copy
import os
import json
import requests

# HOST = 'localhost'
PORT = 19960
DATALENGTH = 10
FRAMENUM = 5
INITIALDATA = None
FLAG = True

TF_IP = '203.250.148.120'
TF_PORT = '20529'
MODEL_NAME = 'AGV_test'


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
                feature_queue.appendleft(np.abs(feature_mean_data - INITIALDATA))
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
    while True:
        if INITIALDATA is not None:
            data = list()
            file_name = input('file name (ex : train/can/can1)')
            print(file_name)
            FLAG = True
            while len(data) < data_len:
                if queue:
                    temp = queue.pop()
                    # print(temp)
                    if temp is not None:
                        data.append(temp)
                    elif temp == 'end':
                        break
                    else:
                        pass
                else:
                    pass
            FLAG = False
            data = np.array(data)
            data_path =os.path.join('./data', (file_name + '.npy'))
            if os.path.exists(os.path.dirname(data_path)):
                np.save('./data/{}.npy'.format(file_name), data)
                print('save : ' + file_name)
            else:
                print('not save')

def Predict(queue, model):
    global FLAG
    # try:
    while True:
        try:
            FLAG = True
            if queue:
                print('predict')
                pre_data = queue.pop()
                data = np.abs(pre_data)
                FLAG = False
                data = np.expand_dims(data, axis=0)
                predict = model.predict({"amplitude" : data[:,0,:], "phase": data[:,1,:]}, verbose = 0)
                predict_dec = np.argmax(predict)
                text = idx2label_Dict[predict_dec]
                print('object : ',text)
        except:
            print('error')


def Transfer(queue):
    global FLAG
    while True:
        try:
            FLAG = True
            if queue:
                print('Transfer')

                send_data = queue.pop()

                print(send_data.shape)
                data = {
                'signature_name': "serving_default",
                'instances': send_data.tolist()
                }
                payload = json.dumps(data)
                model_name = MODEL_NAME
                version = '2'
                url = "http://{0}:{1}/v1/models/{2}/versions/{3}:predict".format(TF_IP, TF_PORT,  model_name, version)
                headers = {"content-type": "application/json"}
                json_response = requests.post(url, data=payload, headers=headers)
                # predictions = json.loads(json_response.text)['predictions']

        except:
            print('transfer error')


def main(data_queue, feature_queue, save, transfer, model_path, predict):

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('', PORT))
    print('wating...')
    try:
        th4 = threading.Thread(target = preprocess_data, args = (data_queue, feature_queue))
        th4.start()
        th5 = threading.Thread(target = setInitialData, args = (data_queue,))
        th5.start()
        # print(save)
        if predict:
            print('start predict')
            model = tf.keras.models.load_model(model_path)
            th3 = threading.Thread(target = Predict, args = (feature_queue, model))
            th3.start()
        if save:
            print('start save')
            th2 = threading.Thread(target = saver,args = (DATALENGTH, feature_queue) )
            th2.start()
        if transfer:
            print('start transfer')
            th6 = threading.Thread(target = Transfer, args = (feature_queue,))
            th6.start()
        while True:
            server_socket.listen()
            client_socket, addr = None, None
            client_socket, addr = server_socket.accept()
            th1 = threading.Thread(target = binder, args = (client_socket, addr, data_queue))
            th1.start()
    except:
        print('server')
    finally:
        server_socket.close()
    try :
        th1.join()
        th2.join()
        th3.join()
        th4.join()
        th5.join()
        th6.join()
    except:
        pass

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Radar data saver and predictor')
    parser.add_argument('-s','--save', action = 'store_true', default=False)
    parser.add_argument('-p', '--predict', action = 'store_true', default=False)
    parser.add_argument('-t', '--transfer', action = 'store_true', default=False)
    args = parser.parse_args()
    data_queue = deque()
    feature_queue = deque()
    model_path = '/Users/joonghocho/Radar/Intelligent_Radar/gui/service_modules/model/branch_model'
    main(data_queue, feature_queue, save = args.save, transfer = args.transfer, model_path = model_path, predict = args.predict)
    