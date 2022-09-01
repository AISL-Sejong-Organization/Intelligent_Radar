from lib2to3.pgen2 import driver
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from classifier import preprocessing as prep
import numpy as np

def VGG_branch(X):
    
    # amp_input = keras.Input(shape=(X,1), name="amplitude")
    # phs_input = keras.Input(shape=(X,1), name='phase')
    inputs = tf.keras.Input(shape=(2, X))
    amp = inputs[:,0,:]
    phs = inputs[:,1,:]
    amp = tf.expand_dims(amp, axis=2)
    phs = tf.expand_dims(phs, axis=2)
    print(phs.shape)
    print(X)
    
    # amp_features = layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (X, 1), padding = 'same')(amp_input)
    amp_features = layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (90, 208, 3), padding = 'same')(amp)
    amp_features = layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same')(amp_features)
    amp_features = layers.MaxPool2D((2,2))(amp_features)
    amp_features = layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same')(amp_features)
    amp_features = layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same')(amp_features)
    amp_features = layers.MaxPool2D((2,2))(amp_features)
    amp_features = layers.Conv2D(256,(3,3), activation = 'relu', padding = 'same')(amp_features)
    amp_features = layers.Conv2D(256,(3,3), activation = 'relu', padding = 'same')(amp_features)
    amp_features = layers.Conv2D(256,(3,3), activation = 'relu', padding = 'same')(amp_features)
    amp_features = layers.Conv2D(256,(3,3), activation = 'relu', padding = 'same')(amp_features)
    amp_features = layers.MaxPool2D((2,2))(amp_features)
    amp_features = layers.Flatten()(amp_features)

    # phs_features = layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (X, 1), padding = 'same')(phs_input)
    phs_features = layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (90, 208, 3), padding = 'same')(phs)
    phs_features = layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same')(phs_features)
    phs_features = layers.MaxPool2D((2,2))(phs_features)
    phs_features = layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same')(phs_features)
    phs_features = layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same')(phs_features)
    phs_features = layers.MaxPool2D((2,2))(phs_features)
    phs_features = layers.Conv2D(256,(3,3), activation = 'relu', padding = 'same')(phs_features)
    phs_features = layers.Conv2D(256,(3,3), activation = 'relu', padding = 'same')(phs_features)
    phs_features = layers.Conv2D(256,(3,3), activation = 'relu', padding = 'same')(phs_features)
    phs_features = layers.Conv2D(256,(3,3), activation = 'relu', padding = 'same')(phs_features)
    phs_features = layers.MaxPool2D((2,2))(phs_features)
    phs_features = layers.Flatten()(phs_features)

    x = layers.concatenate([amp_features, phs_features], axis = -1)
    x = layers.Dense(4096, activation = 'relu')(x)
    x = layers.Dropout(0.9)(x)
    x = layers.Dense(4096, activation = 'relu')(x)
    x = layers.Dropout(0.9)(x)
    # material_output = layers.Dense(5, activation = 'softmax', name = 'material_output')(x)
    material_output = layers.Dense(4, activation = 'softmax', name = 'material_output')(x)

    # model = keras.Model(inputs = [amp_input, phs_input],
    #                     outputs = [material_output],)

    model = keras.Model(inputs = inputs, outputs = material_output)
    model.summary()
    keras.utils.plot_model(model, "./branced_model_2D.png", show_shapes=True)
    model.compile(optimizer = 'adam', loss = tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    return model

class VGG_branch(tf.keras.Model):
    def __init__(self):
        super(VGG_branch, self).__init__()
        self.amp_conv_1 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')
        self.amp_conv_2 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')
        self.amp_maxpool_1 = MaxPool2D((2,2))
        self.amp_conv_3 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')
        self.amp_conv_4 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')
        self.amp_maxpool_2 = MaxPool2D((2,2))
        self.amp_conv_5 = Conv2D(256, (3,3), activation = 'relu', padding = 'same')
        self.amp_conv_6 = Conv2D(256, (3,3), activation = 'relu', padding = 'same')
        self.amp_conv_7 = Conv2D(256, (3,3), activation = 'relu', padding = 'same')
        self.amp_conv_8 = Conv2D(256, (3,3), activation = 'relu', padding = 'same')
        self.amp_maxpool_3 = MaxPool2D((2,2))
        self.amp_flatten = Flatten()

        self.phs_conv_1 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')
        self.phs_conv_2 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')
        self.phs_maxpool_1 = MaxPool2D((2,2))
        self.phs_conv_3 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')
        self.phs_conv_4 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')
        self.phs_maxpool_2 = MaxPool2D((2,2))
        self.phs_conv_5 = Conv2D(256, (3,3), activation = 'relu', padding = 'same')
        self.phs_conv_6 = Conv2D(256, (3,3), activation = 'relu', padding = 'same')
        self.phs_conv_7 = Conv2D(256, (3,3), activation = 'relu', padding = 'same')
        self.phs_conv_8 = Conv2D(256, (3,3), activation = 'relu', padding = 'same')
        self.phs_maxpool_3 = MaxPool2D((2,2))
        self.phs_flatten = Flatten()

        self.fc1 = Dense(4096, activation = 'relu')
        self.dr1 = layers.Dropout(0.8)
        self.fc2 = Dense(4096, activation = 'relu')
        self.dr2 = layers.Dropout(0.8)
        self.fc3 = Dense(4, activation = 'softmax')

    def call(self, x):
        amp = x[:,0,:]
        phs = x[:,1,:]
        amp = tf.expand_dims(amp, axis=2)
        phs = tf.expand_dims(phs, axis=2)
        
        amp = self.amp_conv_1(amp)
        amp = self.amp_conv_2(amp)
        amp = self.amp_maxpool_1(amp)
        amp = self.amp_conv_3(amp)
        amp = self.amp_conv_4(amp)
        amp = self.amp_maxpool_2(amp)
        amp = self.amp_conv_5(amp)
        amp = self.amp_conv_6(amp)
        amp = self.amp_conv_7(amp)
        amp = self.amp_conv_8(amp)
        amp = self.amp_maxpool_3(amp)
        amp = self.amp_flatten(amp)

        phs = self.phs_conv_1(phs)
        phs = self.phs_conv_2(phs)
        phs = self.phs_maxpool_1(phs)
        phs = self.phs_conv_3(phs)
        phs = self.phs_conv_4(phs)
        phs = self.phs_maxpool_2(phs)
        phs = self.phs_conv_5(phs)
        phs = self.phs_conv_6(phs)
        phs = self.phs_conv_7(phs)
        phs = self.phs_conv_8(phs)
        phs = self.phs_maxpool_3(phs)
        phs = self.phs_flatten(phs)

        out = layers.concatenate([amp, phs], axis = -1)
        out = self.fc1(out)
        out = self.dr1(out)
        out = self.fc2(out)
        out = self.dr2(out)
        out = self.fc3(out)

        return out