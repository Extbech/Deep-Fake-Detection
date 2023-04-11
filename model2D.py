import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

def build_2D_model(input_data):
    model = keras.models.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=(7,7), padding="same", strides=1, activation="relu", input_shape=input_data.shape[1:]))
    model.add(layers.MaxPool2D(pool_size=(3,3)))
    model.add(layers.Conv2D(filters=3, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(3,3)))
    model.add(layers.Conv2D(filters=3, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(3,3)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(2))
    return model

def compile_2D_model(model, lr):
    loss = keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    metrics=["accuracy"]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model