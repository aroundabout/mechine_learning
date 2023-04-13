import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from d2l import tensorflow as d2l
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os


import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os


class Cnn_model():
    def __init__(self, input_shape, num_class):
        self.input_shape = input_shape
        self.num_class = num_class
        self._model = keras.Sequential(
            [
                keras.Input(shape=self.input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(self.num_class, activation="softmax"),
            ]
        )
        self._model.summary()

    def get_model(self):
        return self._model


class Resnet():
    def __init__(self, input_shape, num_class) -> None:
        self.input_shape = input_shape
        self.num_class = num_class
        n = 3
        depth = n * 9 + 2
        self._model = self.resnet_v2(self.input_shape, depth, self.num_class)
        self._model.summary()

    def get_model(self):
        return self._model

    def resnet_layer(self, inputs,
                     num_filters=16,
                     kernel_size=3,
                     strides=1,
                     activation='relu',
                     batch_normalization=True,
                     conv_first=True):
        conv = Conv2D(num_filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4))

        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
        else:
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
            x = conv(x)
        return x

    def resnet_v2(self, input_shape, depth, num_classes=10):

        if (depth - 2) % 9 != 0:
            raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
        # Start model definition.
        num_filters_in = 16
        num_res_blocks = int((depth - 2) / 9)

        inputs = Input(shape=input_shape)
        # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
        x = self.resnet_layer(inputs=inputs,
                              num_filters=num_filters_in,
                              conv_first=True)

        # Instantiate the stack of residual units
        for stage in range(3):
            for res_block in range(num_res_blocks):
                activation = 'relu'
                batch_normalization = True
                strides = 1
                if stage == 0:
                    num_filters_out = num_filters_in * 4
                    if res_block == 0:  # first layer and first stage
                        activation = None
                        batch_normalization = False
                else:
                    num_filters_out = num_filters_in * 2
                    if res_block == 0:  # first layer but not first stage
                        strides = 2    # downsample

                # bottleneck residual unit
                y = self.resnet_layer(inputs=x,
                                      num_filters=num_filters_in,
                                      kernel_size=1,
                                      strides=strides,
                                      activation=activation,
                                      batch_normalization=batch_normalization,
                                      conv_first=False)
                y = self.resnet_layer(inputs=y,
                                      num_filters=num_filters_in,
                                      conv_first=False)
                y = self.resnet_layer(inputs=y,
                                      num_filters=num_filters_out,
                                      kernel_size=1,
                                      conv_first=False)
                if res_block == 0:
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = self.resnet_layer(inputs=x,
                                          num_filters=num_filters_out,
                                          kernel_size=1,
                                          strides=strides,
                                          activation=None,
                                          batch_normalization=False)
                x = keras.layers.add([x, y])

            num_filters_in = num_filters_out

        # Add classifier on top.
        # v2 has BN-ReLU before Pooling
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        outputs = Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal')(y)

        # Instantiate model.
        model = Model(inputs=inputs, outputs=outputs)
        return model


class Deeper_cnn():
    def __init__(self, input_shape, num_class) -> None:
        self.input_shape = input_shape
        self.num_class = num_class
        self._model = self.build_model()

    def get_model(self):
        return self._model

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=self.input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_class))
        model.add(Activation('softmax'))
        
        return model
