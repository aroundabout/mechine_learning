import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from d2l import tensorflow as d2l



class Mnist_model():
    def __init__(self,input_shape,num_class):
        self.input_shape=input_shape
        self.num_class=num_class
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