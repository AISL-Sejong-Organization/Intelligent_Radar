import os
import sys
import traceback

import numpy as np

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../")))
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__),"./ml")))
print(sys.path)
# from acconeer.exptool import configs
try:
    # from acconeer.exptool import imock, utils
    from gui.ml import feature_processing as feature_proc  
    from gui.ml import keras_processing as kp
except Exception:
    print("Failed to import deeplearning libraries, please specify acconeer-exploration-folder!")
    exit(1)

ML = kp.MachineLearning()
data = ML.load_train_data(['C:\\acconeer\\gui_tool\\acconeer-python-exploration\\data\\20211015\\data\\Can.npy',
'C:\\acconeer\\gui_tool\\acconeer-python-exploration\\data\\20211015\\data\\Can2.npy',
'C:\\acconeer\\gui_tool\\acconeer-python-exploration\\data\\20211015\\data\\Glass.npy',
'C:\\acconeer\\gui_tool\\acconeer-python-exploration\\data\\20211015\\data\\Glass2.npy',
'C:\\acconeer\\gui_tool\\acconeer-python-exploration\\data\\20211015\\data\\Plastic.npy'])
training_data = ML.training_data

