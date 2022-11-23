from lib2to3.pgen2 import driver
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import layers, models
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
    
    # amp_features = layers.Conv1D(64, (3), activation = 'relu', input_shape = (X, 1), padding = 'same')(amp_input)
    amp_features = layers.Conv1D(64, (3), activation = 'relu', input_shape = (1, X), padding = 'same')(amp)
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

    # phs_features = layers.Conv1D(64, (3), activation = 'relu', input_shape = (X, 1), padding = 'same')(phs_input)
    phs_features = layers.Conv1D(64, (3), activation = 'relu', input_shape = (1, X), padding = 'same')(phs)
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
    keras.utils.plot_model(model, "./branced_model.png", show_shapes=True)
    model.compile(optimizer = 'adam', loss = tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    return model

class ResidualUnit(tf.keras.Model):
    def __init__(self, filter_in, filter_out, kernel_size):
        super(ResidualUnit, self).__init__()

        self.bn1 = keras.layers.BatchNormalization()
        self.conv1 = keras.layers.Conv1D(filter_out, kernel_size, padding='same')
        self.bn2 = keras.layers.BatchNormalization()
        self.conv2 = keras.layers.Conv1D(filter_out, kernel_size, padding='same')

        if filter_in == filter_out:
            self.identity = lambda x: x
        else:
            self.identity = keras.layers.Conv1D(filter_out, 1, padding = 'same')
        
    def call(self, x, training=False, mask = None):
        h = self.bn1(x, training=training)
        h = tf.nn.relu(h)
        h = self.conv1(h)

        h = self.bn2(h, training=training)
        h = tf.nn.relu(h)
        h = self.conv2(h)
        return self.identity(x) + h

class ResnetLayer(tf.keras.Model):
    def __init__(self, filter_in, filters, kernel_size):
        super(ResnetLayer, self).__init__()
        self.sequence = list()

        for f_in, f_out in zip([filter_in] + list(filters), filters):
            self.sequence.append(ResidualUnit(f_in, f_out, kernel_size))

    def call(self, x, training=False , mask=None):
        for unit in self.sequence:
            x = unit(x, training=training)
        return x

class AP_ResNet(tf.keras.Model):
    def __init__(self):
        super(AP_ResNet, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(8, 3, padding='same', activation='relu') #28x28x8
        self.res1 = ResnetLayer(8, (16, 16), 3) # 28X28X16
        self.pool1 = tf.keras.layers.MaxPool1D(2)

        self.res2 = ResnetLayer(16, (32, 32), 3)
        self.pool2 = tf.keras.layers.MaxPool1D(2)

        self.res3 = ResnetLayer(32, (64, 64), 1)
        
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(5, activation = 'softmax')

    def call(self, x, training=False, mask=None):
        amp = x[:, 0, :]
        phs = x[:, 1, :]
        amp = tf.expand_dims(amp, axis = 2)
        phs = tf.expand_dims(phs, axis = 2)
        amp = self.conv1(amp)
        amp = self.res1(amp)
        amp = self.pool1(amp)
        amp = self.res2(amp)
        amp = self.pool2(amp)
        amp = self.res3(amp)
        amp = self.flatten(amp)

        phs = self.conv1(phs)
        phs = self.res1(phs)
        phs = self.pool1(phs)
        phs = self.res2(phs)
        phs = self.pool2(phs)
        phs = self.res3(phs)
        phs = self.flatten(phs)

        out = tf.keras.layers.concatenate([amp, phs], axis = -1)
        out = self.dense1(out)
        out = self.dense2(out)

        return out

class ResNet(tf.keras.Model):
    def __init__(self, dropout):
        super(ResNet, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(8, 3, padding='same', activation='relu') #28x28x8
        self.res1 = ResnetLayer(8, (16, 16), 3) # 28X28X16
        self.pool1 = tf.keras.layers.MaxPool1D(2)

        self.res2 = ResnetLayer(16, (32, 32), 3)
        self.pool2 = tf.keras.layers.MaxPool1D(2)

        self.res3 = ResnetLayer(32, (64, 64), 3)
        
        self.flatten = tf.keras.layers.Flatten()
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(0.8)
        self.dense1 = tf.keras.layers.Dense(128, activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(5, activation = 'softmax')

    def call(self, x, training=False, mask=None):
        amp = x[:, 0, :]
        # phs = x[:, 1, :]
        amp = tf.expand_dims(amp, axis = 2)
        amp = self.conv1(amp)
        amp = self.res1(amp)
        amp = self.dropout1(amp)
        amp = self.pool1(amp)
        amp = self.res2(amp)
        amp = self.dropout1(amp)
        amp = self.pool2(amp)
        amp = self.res3(amp)
        amp = self.flatten(amp)
        amp = self.dropout1(amp)
        amp = self.dense1(amp)
        amp = self.dropout1(amp)
        amp = self.dense2(amp)

        return amp

class ResidualUnit2D(tf.keras.Model):
    def __init__(self, filter_in, filter_out, kernel_size):
        super(ResidualUnit2D, self).__init__()

        self.bn1 = keras.layers.BatchNormalization()
        self.conv1 = keras.layers.Conv2D(filter_out, kernel_size, padding='same')
        self.bn2 = keras.layers.BatchNormalization()
        self.conv2 = keras.layers.Conv2D(filter_out, kernel_size, padding='same')

        if filter_in == filter_out:
            self.identity = lambda x: x
        else:
            self.identity = keras.layers.Conv2D(filter_out, 1, padding = 'same')
        
    def call(self, x, training=False, mask = None):
        h = self.bn1(x, training=training)
        h = tf.nn.relu(h)
        h = self.conv1(h)

        h = self.bn2(h, training=training)
        h = tf.nn.relu(h)
        h = self.conv2(h)
        return self.identity(x) + h

class ResnetLayer2D(tf.keras.Model):
    def __init__(self, filter_in, filters, kernel_size):
        super(ResnetLayer2D, self).__init__()
        self.sequence = list()

        for f_in, f_out in zip([filter_in] + list(filters), filters):
            self.sequence.append(ResidualUnit2D(f_in, f_out, kernel_size))

    def call(self, x, training=False , mask=None):
        for unit in self.sequence:
            x = unit(x, training=training)
        return x

class ResNetLSTM(tf.keras.Model):
    def __init__(self, dropout = 0):
        super(ResNetLSTM, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(8, (3,3), padding='same', activation='relu') #28x28x8
        self.res1 = ResnetLayer2D(8, (16, 16), (3,3)) # 28X28X16
        self.pool1 = tf.keras.layers.MaxPool2D((2,2))

        self.res2 = ResnetLayer2D(16, (32, 32), (3,3))
        self.pool2 = tf.keras.layers.MaxPool2D((2,2))

        self.res3 = ResnetLayer2D(32, (64, 64), (3,3))
        
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(1024, activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(512, activation = 'relu')
        self.dense3 = tf.keras.layers.Dense(5, activation = 'softmax')
        
        self.lstm1 = tf.keras.layers.LSTM(256, return_sequences=False , dropout = 0.8)
        
        self.reshape = tf.keras.layers.Reshape((-1, 64))

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(0.8)

    def call(self, x, training=False, mask=None):
        # print('x shape',x.shape)
        amp = x[:, :, 0, :]
        # phs = x[:, 1, :]
        # print(amp.shape)
        amp = tf.expand_dims(amp, axis = -1)
        amp = self.conv1(amp)
        amp = self.res1(amp)
        amp = self.dropout1(amp)
        amp = self.pool1(amp)
        amp = self.res2(amp)
        amp = self.dropout1(amp)
        amp = self.pool2(amp)
        amp = self.res3(amp)
        
        amp = self.dropout1(amp)
        amp = self.reshape(amp)
        amp = self.lstm1(amp)
        amp = self.flatten(amp)
        amp = self.dropout1(amp)
        
        
        out = self.dense1(amp)
        out = self.dropout1(out)
        out = self.dense2(out)
        out = self.dense3(out)
        return out

class ConvLSTM(tf.keras.Model):
    def __init__(self):
        super(ConvLSTM, self).__init__()
        self.convlstm1 = tf.keras.layers.ConvLSTM1D(64, 3, activation = 'relu', padding = 'same', return_sequences = True)
        self.convlstm2 = tf.keras.layers.ConvLSTM1D(128, 3, activation = 'relu', padding = 'same', return_sequences = True)
        self.convlstm3 = tf.keras.layers.ConvLSTM1D(256, 3, activation = 'relu', padding = 'same')

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.flatten = tf.keras.layers.Flatten()

        self.dense1 = tf.keras.layers.Dense(1024, activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(512, activation = 'relu')
        self.dense3 = tf.keras.layers.Dense(5, activation = 'softmax')

        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.dropout3 = tf.keras.layers.Dropout(0.5)

    def call(self, x, training=False, mask=None):
        # amp = x[:, :, 0, :]
        # amp = tf.expand_dims(x, axis = -1)
        amp = x
        amp = tf.expand_dims(amp, axis = -1)
        # print(amp.shape)
        amp = self.convlstm1(amp)
        amp = self.bn1(amp)
        amp = self.convlstm2(amp)
        amp = self.bn2(amp)
        amp = self.convlstm3(amp)
        amp = self.bn3(amp)
        amp = self.flatten(amp)
        # amp = self.dropout1(amp)
        amp = self.dense1(amp)
        # amp = self.dropout2(amp)
        amp = self.dense2(amp)
        # amp = self.dropout3(amp)
        amp = self.dense3(amp)

        return amp

class ConvLSTM_dropout(tf.keras.Model):
    def __init__(self):
        super(ConvLSTM_dropout, self).__init__()
        self.convlstm1 = tf.keras.layers.ConvLSTM1D(64, 3, activation = 'relu', padding = 'same', return_sequences = True)
        self.convlstm2 = tf.keras.layers.ConvLSTM1D(128, 3, activation = 'relu', padding = 'same', return_sequences = True)
        self.convlstm3 = tf.keras.layers.ConvLSTM1D(256, 3, activation = 'relu', padding = 'same')

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.flatten = tf.keras.layers.Flatten()

        self.dense1 = tf.keras.layers.Dense(2048, activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(2048, activation = 'relu')
        self.dense3 = tf.keras.layers.Dense(5, activation = 'softmax')

        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.dropout3 = tf.keras.layers.Dropout(0.5)
        self.dropout4 = tf.keras.layers.Dropout(0.5)

    def call(self, x, training=False, mask=None):
        # amp = x[:, :, 0, :]
        # amp = tf.expand_dims(x, axis = -1)
        amp = x
        amp = tf.expand_dims(amp, axis = -1)
        # print(amp.shape)
        amp = self.convlstm1(amp)
        amp = self.bn1(amp)
        # amp = self.dropout1(amp)
        amp = self.convlstm2(amp)
        amp = self.bn2(amp)
        # amp = self.dropout2(amp)
        amp = self.convlstm3(amp)
        amp = self.bn3(amp)
        amp = self.flatten(amp)
        amp = self.dropout3(amp)
        amp = self.dense1(amp)
        amp = self.dropout4(amp)
        amp = self.dense2(amp)
        # amp = self.dropout3(amp)
        amp = self.dense3(amp)

        return amp


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_transformer_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
    n_classes = 5,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

class ResNet_CNN(tf.keras.Model):
    def __init__(self, weight=None):
        super(ResNet_CNN, self).__init__()
        self.basemodel = tf.keras.applications.ResNet152V2(
            include_top = False,
            weights = weight,
            input_shape = (90, 208, 3),
        )
        self.conv1 = tf.keras.layers.Conv2D(3, (1,1), padding='valid', activation = 'relu')
        self.fc1 = tf.keras.layers.Dense(1024, activation = 'relu')
        self.fc2 = tf.keras.layers.Dense(1024, activation = 'relu')
        self.fc3 = tf.keras.layers.Dense(4, activation = 'softmax')

        self.flatten = tf.keras.layers.Flatten()

        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.dropout3 = tf.keras.layers.Dropout(0.5)
    def call(self, x, training=False, mask=None):
        x = self.conv1(x)
        x = self.basemodel(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.dropout3(x)
        x = self.fc3(x)
        return x

class VGG_CNN(tf.keras.Model):
    def __init__(self, weight=None):
        super(VGG_CNN, self).__init__()
        self.basemodel = tf.keras.applications.vgg16.VGG16(
            include_top = False,
            weights = weight,
            input_shape = (90, 208, 3),
        )
        self.conv1 = tf.keras.layers.Conv2D(3, (1,1), padding='valid', activation = 'relu')
        self.fc1 = tf.keras.layers.Dense(1024, activation = 'relu')
        self.fc2 = tf.keras.layers.Dense(1024, activation = 'relu')
        self.fc3 = tf.keras.layers.Dense(4, activation = 'softmax')

        self.flatten = tf.keras.layers.Flatten()

        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.dropout3 = tf.keras.layers.Dropout(0.5)
    def call(self, x, training=False, mask=None):
        x = self.conv1(x)
        x = self.basemodel(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.dropout3(x)
        x = self.fc3(x)
        return x

class basemodel(tf.keras.Model):
    def __init__(self):
        super(basemodel, self).__init__()
        self.nl1 = layers.Normalization(axis = -1)
        self.fc1 = layers.Dense(units = 10, activation = 'relu')
        self.bn1 = layers.BatchNormalization()
        self.fc2 = layers.Dense(units = 5, activation = 'relu')
        self.out = layers.Dense(units = 4, activation = 'softmax')
    
    def call(self, x, training = False, mask = None):
        out = self.nl1(x)
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.fc2(out)
        out = self.out(out)
        return out