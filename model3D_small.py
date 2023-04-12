import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

def build_small_model(input_data):
    model = keras.models.Sequential()
    model.add(layers.Conv3D(filters=3, kernel_size=3, padding="same", strides=1, activation="relu", input_shape=input_data.shape[1:]))
    model.add(layers.MaxPool3D(pool_size=3, padding="same"))
    model.add(layers.Conv3D(filters=3, kernel_size=3, padding="same", strides=1, activation="relu"))
    model.add(layers.MaxPool3D(pool_size=3, padding="same"))
    model.add(layers.Conv3D(filters=3, kernel_size=3, padding="same", strides=1, activation="relu"))
    model.add(layers.MaxPool3D(pool_size=3, padding="same"))
    model.add(layers.Conv3D(filters=3, kernel_size=3, padding="same", strides=1, activation="relu"))
    model.add(layers.MaxPool3D(pool_size=3, padding="same"))
    model.add(layers.Conv3D(filters=3, kernel_size=3, padding="same", strides=1, activation="relu"))
    model.add(layers.MaxPool3D(pool_size=3, padding="same"))
    model.add(layers.Conv3D(filters=3, kernel_size=3, padding="same", strides=1, activation="relu"))
    model.add(layers.MaxPool3D(pool_size=3, padding="same"))
    model.add(layers.Conv3D(filters=3, kernel_size=3, padding="same", strides=1, activation="relu"))
    model.add(layers.MaxPool3D(pool_size=3, padding="same"))
    model.add(layers.Conv3D(filters=3, kernel_size=3, padding="same", strides=1, activation="relu"))
    model.add(layers.MaxPool3D(pool_size=3, padding="same"))
    model.add(layers.Conv3D(filters=3, kernel_size=3, padding="same", strides=1, activation="relu"))
    model.add(layers.MaxPool3D(pool_size=3, padding="same"))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(1))
    return model
