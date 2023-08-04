import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

class NeuralNet:
    def __init__(self, fc_layer_size=11, sc_layer_size=7, dropout=0):
        self.fc_layer_size = fc_layer_size
        self.sc_layer_size = sc_layer_size
        self.dropout = dropout

        # Set the GPU as the backend for Keras
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)

    def create_model(self):
        # Initiate the network
        network = Sequential()

        # Input layer that is fully connected with ReLU activation
        network.add(Dense(4, input_dim=4, activation="relu", use_bias=True, bias_initializer="random_normal", kernel_initializer="he_normal"))
        network.add(Dropout(self.dropout))

        # Hidden layers
        network.add(Dense(self.fc_layer_size, activation="relu", use_bias=True, bias_initializer="glorot_normal", kernel_initializer="glorot_normal"))
        network.add(Dropout(self.dropout))
        network.add(Dense(self.sc_layer_size, activation="relu", use_bias=True, bias_initializer="glorot_normal", kernel_initializer="glorot_normal"))

        # Output layer that is fully connected with sigmoid activation
        network.add(Dense(1, activation="sigmoid", kernel_initializer="glorot_normal"))

        # Compile network
        network.compile(optimizer="adam", loss="binary_crossentropy", metrics=["binary_accuracy"])

        # Return completed network
        return network
