import os
import pickle
from typing import Optional, Dict, List
from utils.tools import read_config
import argparse
from model.Cnn_model import Cnn_model, Resnet, Deeper_cnn
from keras.datasets import mnist, cifar10
import numpy as np
import keras
from keras.callbacks import ModelCheckpoint
from utils.tools import get_cur_path
from utils.const import CHECKPOINTS
from matplotlib import pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from keras import losses

from utils.loss import symmetrical_cross_entropy, cross_entropy, fusion_cross_entropy

losses.categorical_crossentropy


class Trainer(object):

    def __init__(self, config: Dict, args) -> None:
        self.loss = config['loss']
        self.metric = config['metrics']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.val_split = config['val_split']
        self.num_class = config['num_class']
        self.input_x = config['input_x']
        self.input_y = config['input_y']
        self.input_c = config['input_c']

        self.loss_type = config['loss_type']
        self.dataset = args.dataset

        self.save_path = os.path.join(get_cur_path(), CHECKPOINTS, str(self.dataset)+self.loss+'_model_weights.h5')
        self.hist_save_path = os.path.join(get_cur_path(), CHECKPOINTS, str(self.dataset)+self.loss+'.history')
        self.fig_save_path = os.path.join(get_cur_path(), CHECKPOINTS, str(self.dataset)+self.loss)
        print("save and load wights path is {}".format(self.save_path))

        self.opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
        self.loss_fn = self._build_loss_fn(self.loss)

        self.build(model_name=args.model_name)

    def build(self, model_name):
        input_size = (self.input_x, self.input_y, self.input_c)
        num_class = self.num_class
        if model_name == 'resnet':
            model = Resnet(input_shape=input_size, num_class=self.num_class)
            self.model = model.get_model()
        elif model_name == 'cnn':
            model = Cnn_model(input_shape=input_size, num_class=self.num_class)
            self.model = model.get_model()
        elif model_name == 'deepercnn':
            model = Deeper_cnn(input_shape=input_size, num_class=self.num_class)
            self.model = model.get_model()
        else:
            model = Deeper_cnn(input_shape=input_size, num_class=self.num_class)
            self.model = model.get_model()

    def _build_loss_fn(self, loss_name):
        if loss_name == 'cross_entropy':
            return cross_entropy
        elif loss_name == 'symmetrical_cross_entropy':
            return symmetrical_cross_entropy
        elif loss_name == 'fusion_cross_entropy':
            return fusion_cross_entropy
        else:
            return cross_entropy

    def _lr_schedule(slef, epoch):
        lr = 1e-3
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1
        print('Learning rate: ', lr)
        return lr

    def _parse_config(self, config: Dict) -> None:
        self.device = config['device']

    def _save(self, model: keras.Sequential) -> None:
        model.save_weights(self.save_path)

    def _load(self, model: keras.Sequential) -> keras.Sequential:
        model.load_weights(self.save_path)

    def load_data(self, dataset: str = "mnist"):
        if dataset == 'mnist':
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
        elif dataset == 'CIFAR10':
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        else:
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255

        print("x_train shape:", x_train.shape)
        print(x_train.shape[0], "train samples")
        print(x_test.shape[0], "test samples")
        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, self.num_class)
        y_test = keras.utils.to_categorical(y_test, self.num_class)
        return (x_train, y_train), (x_test, y_test)

    def train(self) -> None:
        (x_train, y_train), (x_test, y_test) = self.load_data(self.dataset)

        model = self.model
        if self.loss_type == 'custom':
            model.compile(loss=self.loss_fn, optimizer=self.opt, metrics=self.metric)
        else:
            model.compile(loss=self.loss, optimizer=self.opt, metrics=self.metric)

        checkpoint = ModelCheckpoint(filepath=self.save_path,
                                     monitor='val_accuracy',
                                     verbose=1,
                                     save_best_only=True)

        lr_scheduler = LearningRateScheduler(self._lr_schedule)

        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=5,
                                       min_lr=0.5e-6)

        callbacks = [checkpoint, lr_reducer, lr_scheduler]
        datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            zca_epsilon=1e-06,
            rotation_range=0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.,
            zoom_range=0.,
            channel_shift_range=0.,
            fill_mode='nearest',
            cval=0.,
            horizontal_flip=True,
            vertical_flip=False,
            rescale=None,
            preprocessing_function=None,
            data_format=None,
            validation_split=0.0)

        datagen.fit(x_train)

        hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=self.batch_size),
                                   steps_per_epoch=len(x_train) // self.batch_size,
                                   validation_data=(x_test, y_test),
                                   epochs=self.epochs, verbose=1, workers=4,
                                   callbacks=callbacks)
        score = model.evaluate(x_test, y_test, verbose=0)

        plt.plot(hist.epoch, hist.history['accuracy'], label='acc')
        plt.plot(hist.epoch, hist.history['val_accuracy'], label='val_acc')

        plt.legend()

        with open(self.hist_save_path, 'wb') as file_pi:
            pickle.dump(hist.history, file_pi)

        # with open('/trainHistoryDict', "rb") as file_pi:
        #     history = pickle.load(file_pi)

        print("Test loss:", score[0])
        print("Test accuracy:", score[1])
        # self._save(model)

    def test(self) -> None:
        (x_train, y_train), (x_test, y_test) = self.load_data()

        model = self.model
        model.compile(loss=self.loss, optimizer=self.opt, metrics=self.metric)
        self._load(model)
        score = model.evaluate(x_test, y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])


def run(config, arg) -> None:
    trainer = Trainer(config=config, args=arg)
    trainer.train()


def test(config, arg) -> None:
    trainer = Trainer(config=config, args=arg)
    trainer.test()


def get_args():
    parser = argparse.ArgumentParser(description="mnist model")
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--model_name', type=str, default='deepercnn')
    parser.add_argument('--loss', type=str, default='cross_entropy')
    parser.add_argument('--device', type=str, default='0')

    return parser.parse_args()


if __name__ == "__main__":
    config = read_config()
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    if args.mode == 'test':
        test(config, arg=args)
    else:
        run(config, arg=args)
