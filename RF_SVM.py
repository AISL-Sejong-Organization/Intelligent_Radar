import os
import sys
import traceback

import numpy as np

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../")))
print(sys.path)
# from acconeer.exptool import configs
try:
    # from acconeer.exptool import imock, utils
    import gui.ml.feature_processing as feature_proc
    import gui.ml.keras_processing as kp
except Exception:
    print("Failed to import deeplearning libraries, please specify acconeer-exploration-folder!")
    exit(1)

ML = kp.MachineLearning()
data = ML.load_train_data(['data\20211015\data\Can.npy','data\20211015\data\Can2.npy','data\20211015\data\Glass.npy',
'data\20211015\data\Glass2.npy',
'data\20211015\data\Plastic.npy'])

print(data)
