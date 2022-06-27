import socket, threading
import numpy as np
from collections import deque
import argparse
from math import pi
import tensorflow as tf
import matplotlib.pyplot as plt

# HOST = 'localhost'
PORT = 19960
DATALENGTH = 1000
FRAMENUM = 5
INITIALDATA = None
FLAG = True

label2idx_Dict = {
                'can' : 0,
                'paper' : 1,
                'glass' : 2,
                'plastic' : 3,
            }

idx2label_Dict = {
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
                INITIALDATA = 0
                feature_data = list()
                cnt = 0
                while cnt < FRAMENUM:
                    if data_queue:
                        print('cnt', cnt)
                        temp = data_queue.pop()
                        if temp is not None:
                            feature_data.append(temp)
                            cnt += 1
                feature_mean_data = np.mean(feature_data, axis=0)
                print(feature_mean_data.shape)
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
    while True:
        if INITIALDATA is not None:
            INITIALDATA = 0
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
            # print(data)
            np.save('./load_data/{}.npy'.format(file_name), data)
            print('save' + file_name)

def seperater(queue):
    data = list()
    data = np.arrya(data)
    while True:
        pre_data = queue.pop
        amp = np.abs(pre_data)
        amp = amp / 1720.8725345010303
        phs = np.angle(pre_data)
        # phs = (phs - (- pi)) / (pi - (- pi))
        sin = np.sin(phs)
        sin = (sin + 1) / 2
        seperated_data = np.append(amp, sin, axis = 0)
        return np.array(seperated_data)

def prediction(model, data):
    return model.predict(data, verbose = 0)

def process(queue, model_path):
    try:
        while True:
            if FLAG:
                if queue:
                    data = seperater(queue)
                    data = np.expand_dims(data, axis=0)
                    # print(data)
                    model = tf.keras.models.load_model(model_path)
                    # print(1)
                    predict = prediction(model, data)
                    predict_dec = np.argmax(predict)
                    text = idx2label_Dict[predict_dec]
                    # if predict_dec == 0:
                    #     text = 'can'
                    # elif predict_dec == 1:
                    #     text = 'paper'
                    # elif predict_dec == 2:
                    #     text = 'glass'
                    # elif predict_dec == 3:
                    #     text = 'plastic'
                    # else :
                    #     text = 'error'
                    print(text)
                else:
                    print('queue is empty')
    except:
        pass

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
            th3 = threading.Thread(target = process, args = (feature_queue, model_path))
            th3.start()
        if save:
            print('start save')
            th2 = threading.Thread(target = saver,args = (DATALENGTH, feature_queue) )
            th2.start()
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
    th1.join()
    th2.join()
    th3.join()
    th4.join()
    th5.join()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Radar data saver and predictor')
    parser.add_argument('-s','--save', action = 'store_true')
    parser.add_argument('-p', '--predict', action='store_true')
    args = parser.parse_args()
    data_queue = deque()
    feature_queue = deque()
    model_path = '/Users/joonghocho/Radar/Intelligent_Radar/gui/service_modules/model/model/branch_model'
    main(data_queue, feature_queue, save = args.save, model_path = model_path, predict = args.predict)
    