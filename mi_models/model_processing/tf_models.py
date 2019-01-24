# -*- coding: utf-8 -*-
# @time      : 2019/1/17 19:51
# @author    : yuxiaoqi@cmschina.com.cn
# @file      : tf_models.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data
# from ..data_processing.dataset import DataSet
from mi_remote.mi_models.data_processing.dataset import DataSet
from mi_remote.mi_models.utils.utils import get_parent_dir
# from ..utils.utils import get_parent_dir


class TF_Reg_Model():
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self._dataset = None
        self.training_op = None
        self.accuracy = None

    def build_model(self, n_hidden1, n_hidden2):
        self.train_X = tf.placeholder(tf.float32, shape=(None, 100, 10), name='X')
        self.train_Y = tf.placeholder(tf.float32, shape=(None), name="Y")
        with tf.name_scope('dnn'):
            h1 = fully_connected(self.train_X, n_hidden1, scope='hidden1', activation_fn=tf.nn.leaky_relu)
            h2 = fully_connected(h1, n_hidden2, scope='hidden2', activation_fn=tf.nn.leaky_relu)
            logits = fully_connected(h2, n_hidden1, scope='outputs', activation_fn=tf.nn.leaky_relu)

        with tf.name_scope('loss'):
            # xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.train_Y, logits=logits)
            loss = tf.reduce_mean(tf.square(logits-self.train_Y))
            # loss = tf.reduce_mean(xentropy, name='loss')

        with tf.name_scope('train'):
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            self.training_op = optimizer.minimize(loss)

        with tf.name_scope('eval'):
            # correct = tf.nn.in_top_k(logits, self.train_Y, 1)
            correct = tf.subtract(logits, self.train_Y)
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    def train_model(self, train_X, train_Y, n_epochs=100, batch_size=50):
        init = tf.global_variables_initializer()
        # saver = tf.train.Saver()
        self._dataset = DataSet(train_X, train_Y)
        with tf.Session() as self.sess:
            init.run()
            for epoch in range(n_epochs):
                for iteration in range(self._dataset.num_examples // batch_size):
                    x_batch, y_batch = self._dataset.next_batch(batch_size)
                    self.sess.run(self.training_op, feed_dict={self.train_X: x_batch, self.train_Y: y_batch})
                x_test, y_test = self._dataset.next_batch(batch_size)
                # acc_train = self.accuracy.eval(feed_dict={self.train_X: x_batch, self.train_Y: y_batch})
                acc_test = self.accuracy.eval(feed_dict={self.train_X: x_test, self.train_Y: y_test})
                print(epoch, 'test_accuracy:', acc_test)
            # save_path = saver.save(self.sess, "./my_model_final")
            # print(save_path)

    def save_model(self, model_name):
        model_path = os.path.join(get_parent_dir(), 'data', 'models', model_name)
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, model_path)
        print(save_path)

    def load_model(self, model_name):
        model_path = os.path.join(get_parent_dir(), 'data', 'models', model_name)
        pass


if __name__ == '__main__':
    x = np.random.random(size=(10000, 100, 10))
    y = np.random.random(size=(10000))
    m = TF_Reg_Model()
    m.build_model(100, 50)
    m.train_model(x, y, n_epochs=10)
    m.save_model('testing')
