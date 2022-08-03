import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import layers, models
from classifier import preprocessing as prep

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
            self.identity = keras.layers.Conv1D(filter_out, (1,1), padding = 'same')
        
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
        self.conv1 = tf.keras.layers.Conv1D(8, (3,3), padding='same', activation='relu') #28x28x8
        self.res1 = ResnetLayer(8, (16, 16), (3,3)) # 28X28X16
        self.pool1 = tf.keras.layers.MaxPool1D(2)

        self.res2 = ResnetLayer(16, (32, 32), (3,3))
        self.pool2 = tf.keras.layers.MaxPool1D(2)

        self.res3 = ResnetLayer(32, (64, 64), (3, 3))
        
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(10, activation = 'softmax')

    def call(self, x, training=False, mask=None):
        amp = x[:, 0, :]
        phs = x[:, 1, :]

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
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(8, (3,3), padding='same', activation='relu') #28x28x8
        self.res1 = ResnetLayer(8, (16, 16), (3,3)) # 28X28X16
        self.pool1 = tf.keras.layers.MaxPool1D(2)

        self.res2 = ResnetLayer(16, (32, 32), (3,3))
        self.pool2 = tf.keras.layers.MaxPool1D(2)

        self.res3 = ResnetLayer(32, (64, 64), (3, 3))
        
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(10, activation = 'softmax')

    def call(self, x, training=False, mask=None):
        amp = x[:, 0, :]
        phs = x[:, 1, :]

        amp = self.conv1(amp)
        amp = self.res1(amp)
        amp = self.pool1(amp)
        amp = self.res2(amp)
        amp = self.pool2(amp)
        amp = self.res3(amp)
        amp = self.flatten(amp)

        out = self.dense1(amp)
        out = self.dense2(out)

        return out
