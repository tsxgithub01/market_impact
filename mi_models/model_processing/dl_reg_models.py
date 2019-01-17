# -*- coding: utf-8 -*-
# @time      : 2018/10/19 11:45
# @author    : yuxiaoqi@cmschina.com.cn
# @file      : dl_reg_models.py

from model_processing.models import Model
from utils.utils import get_config
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.optimizers import SGD

config = get_config()


class Dl_Reg_Model(Model):
    def __init__(self, model_name='linear'):
        self.model_name = model_name
        learning_rate = float(config['dl_reg_model']['learning_rate']) or 0.1
        self.defsgd = SGD(lr=learning_rate)

    def build_model(self, **kwargs):
        units = int(config['dl_reg_model']['units']) or 1
        input_dim = int(config['dl_reg_model']['input_dim']) or 1
        opt = config['dl_reg_model']['optimizer'] or 'sgd'
        loss = config['dl_reg_model']['loss'] or 'mse'
        activation = config['dl_reg_model']['activation'] or 'tanh'
        hidden_units = int(config['dl_reg_model']['hidden_units']) or 10

        if self.model_name == 'linear':
            self.model = Sequential()
            self.model.add(Dense(units=units, input_dim=input_dim))
            self.model.compile(optimizer=opt, loss=loss)
        elif self.model_name == 'nonlinear':  # TODO TBD whether to hv different subtype
            self.model = Sequential()
            self.model.add(Dense(units=hidden_units, input_dim=1))
            self.model.add(Activation(activation=activation))
            self.model.add(Dense(units=hidden_units, input_dim=input_dim, activation='relu'))
            self.model.add(Dense(units=units))
            self.model.add(Activation(activation=activation))
            self.model.compile(optimizer=opt, loss=loss)

    def train_model(self, train_X=[], train_Y=[], **kwargs):
        batches = kwargs.get('batches') or 100
        for step in range(batches):
            cost = self.model.train_on_batch(train_Y, train_Y)
            if step % 50 == 0:
                print('After {0} trainings, the cost: {1}'.format(step, cost))

    def eval_model(self, y_true, y_pred, metrics):
        '''

        :param y_true:
        :param y_pred:
        :param metrics: []
        :return:
        '''
        pass

    def output_model(self):
        if self.model_name == 'linear':
            W, b = self.model.layers[0].get_weights()
            print(W, b)
