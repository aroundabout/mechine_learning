import os
from typing import Optional, Dict, List
from utils.tools import read_config
import argparse
from model.Mnist_model import Mnist_model
from keras.datasets import mnist
import numpy as np
import keras
from keras.callbacks import ModelCheckpoint
from utils.tools import get_cur_path
from utils.const import CHECKPOINTS

class Trainer(object):

    def __init__(self, config: Dict) -> None:
        self.loss = config['loss']
        self.opt = config['optimizer']
        self.metric = config['metrics']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.val_split = config['val_split']
        self.num_class = config['num_class']
        self.input_x = config['input_x']
        self.input_y = config['input_y']
        self.input_c = config['input_c']
        
        self.save_path=os.path.join(get_cur_path(),CHECKPOINTS,'mnist_model_weights.h5')
        print("save and load wights path is {}".format(self.save_path))

    def _parse_config(self, config: Dict) -> None:
        self.device = config['device']

    def _save(self,model:keras.Sequential) -> None:
        model.save_weights(self.save_path)

    def _load(self,model:keras.Sequential)-> keras.Sequential:
        model.load_weights(self.save_path)

    def load_data(self, dataset: str = "mnist"):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        # Make sure images have shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        print("x_train shape:", x_train.shape)
        print(x_train.shape[0], "train samples")
        print(x_test.shape[0], "test samples")
        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train,  self.num_class)
        y_test = keras.utils.to_categorical(y_test,  self.num_class)
        return (x_train, y_train), (x_test, y_test)

    def train(self) -> None:
        (x_train, y_train), (x_test, y_test) = self.load_data()
        input_size = (self.input_x, self.input_y, self.input_c)

        mnist_model = Mnist_model(
            input_shape=input_size, num_class=self.num_class)
        model = mnist_model.get_model()
        model.compile(loss=self.loss, optimizer=self.opt, metrics=self.metric)
        model.fit(x_train, y_train, batch_size=self.batch_size,
                  epochs=self.epochs, validation_split=self.val_split)

        score = model.evaluate(x_test, y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])
        self._save(model)
    
    def test(self)->None:
        (x_train, y_train), (x_test, y_test) = self.load_data()
        input_size = (self.input_x, self.input_y, self.input_c)
        mnist_model = Mnist_model(
            input_shape=input_size, num_class=self.num_class)
        model = mnist_model.get_model()
        model.compile(loss=self.loss, optimizer=self.opt, metrics=self.metric)
        self._load(model)
        score = model.evaluate(x_test, y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

def run(config) -> None:
    trainer = Trainer(config=config)
    trainer.train()

def test(config)->None:
    trainer=Trainer(config=config)
    trainer.test()
    

def get_args():
    parser = argparse.ArgumentParser(description="mnist model")
    parser.add_argument('--mode', type=str, default='test')
    return parser.parse_args()


if __name__ == "__main__":
    config = read_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = config['device']
    args = get_args()
    if args.mode=='test':
        test(config)
    else:
        run(config)
