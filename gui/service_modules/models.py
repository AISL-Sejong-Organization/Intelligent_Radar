import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import layers, models
from classifier import preprocessing as prep

def VGG_branch(X, Y):
    
    amp_input = keras.Input(shape=(X.shape[2],1), name="amplitude")
    phs_input = keras.Input(shape=(X.shape[2],1), name='phase')
    
    amp_features = layers.Conv1D(64, (3), activation = 'relu', input_shape = (X.shape[2], 1), padding = 'same')(amp_input)
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

    phs_features = layers.Conv1D(64, (3), activation = 'relu', input_shape = (X.shape[2], 1), padding = 'same')(phs_input)
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

    model = keras.Model(inputs = [amp_input, phs_input],
                        outputs = [material_output],)

    model.summary()
    keras.utils.plot_model(model, "./branced_model.png", show_shapes=True)
    model.compile(optimizer = 'adam', loss = tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    return model
