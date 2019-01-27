# -*- coding: utf-8 -*-
# @time      : 2019/1/17 19:51
# @author    : yuxiaoqi@cmschina.com.cn
# @file      : tf_models.py

import os
import numpy as np
import tensorflow as tf
from logger import Logger
from tensorflow.contrib.layers import fully_connected
from mi_remote.mi_models.data_processing.dataset import DataSet
from mi_remote.mi_models.utils.utils import get_parent_dir

logger = Logger('log.txt', 'INFO', __name__).get_log()


class TF_Reg_Model():
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self._dataset = None
        self.training_op = None
        self.accuracy = None
        self.sess = None

    def build_model(self, n_hidden1, n_hidden2):
        self.train_X = tf.placeholder(tf.float32, shape=(None, 100), name='X')
        self.train_Y = tf.placeholder(tf.float32, shape=(None, 1), name="Y")
        # currently it is mainly for pov for the tmp impact
        self.acc_input = tf.placeholder(tf.float32, shape=(None, 1), name='acc')
        self.b1 = tf.Variable(np.random.random(), name='b1')

        with tf.name_scope('dnn'):
            h1 = fully_connected(self.train_X, n_hidden1, scope='hidden1', activation_fn=tf.nn.leaky_relu)
            h2 = fully_connected(h1, n_hidden2, scope='hidden2', activation_fn=tf.nn.leaky_relu)
            self.i_star = fully_connected(h2, 1, scope='i_star', activation_fn=tf.nn.leaky_relu)
            acc_h1 = fully_connected(self.acc_input, n_hidden1, scope='acc_hidden1', activation_fn=tf.nn.leaky_relu)
            acc_h2 = fully_connected(acc_h1, n_hidden2, scope='acc_hidden2', activation_fn=tf.nn.leaky_relu)
            acc_h3 = fully_connected(acc_h2, 1, scope='acc_hidden3', activation_fn=tf.nn.leaky_relu)
            self.tmp_impact = self.b1 * self.i_star * acc_h3
            self.perm_impact = (1 - self.b1) * acc_h3
            self.output = self.b1 * self.i_star * acc_h3 + (1 - self.b1) * acc_h3

        with tf.name_scope('loss'):
            self.loss = tf.losses.mean_squared_error(labels=self.train_Y, predictions=self.output)

        with tf.name_scope('train'):
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            self.training_op = optimizer.minimize(self.loss)

        with tf.name_scope('summary'):
            tf.summary.scalar("loss", self.loss)
            self.summary_op = tf.summary.merge_all()

    def train_model(self, train_X, train_Y, acc, n_epochs=100, batch_size=50, model_name=None):
        init = tf.global_variables_initializer()
        # saver = tf.train.Saver()
        self._dataset = DataSet(train_X, train_Y, acc)
        with tf.Session() as self.sess:
            init.run()
            for epoch in range(n_epochs):
                for iteration in range(self._dataset.num_examples // batch_size):
                    x_batch, y_batch, acc = self._dataset.next_batch(batch_size)
                    self.sess.run([self.training_op],
                                  feed_dict={self.train_X: x_batch, self.train_Y: y_batch, self.acc_input: acc})
                x_test, y_test, acc_test = self._dataset.next_batch(batch_size)
                loss = self.sess.run(self.loss,
                                     feed_dict={self.train_X: x_test, self.train_Y: y_test, self.acc_input: acc_test})
                logger.info(epoch, 'test_loss:', loss)
            if model_name:
                model_path = os.path.join(get_parent_dir(), 'data', 'models', model_name)
                saver = tf.train.Saver()
                save_path = saver.save(self.sess, model_path)
                logger.info("Saved to path:{0}".format(save_path))

    def save_model(self, model_name):
        model_path = os.path.join(get_parent_dir(), 'data', 'models', model_name)
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, model_path)
        logger.info("Saved to path:{0}".format(save_path))

    def load_model(self, model_name):
        model_path = os.path.join(get_parent_dir(), 'data', 'models', model_name)
        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess=self.sess,
                      save_path=model_path)

    def predict(self, x=None, acc=None):
        prediction = self.sess.run([self.tmp_impact, self.perm_impact, self.i_star],
                                   feed_dict={self.train_X: x, self.acc_input: acc})
        return prediction


if __name__ == '__main__':
    # train and save model
    x = np.random.random(size=(10000, 100))
    y = np.random.random(size=(10000, 1))
    acc = np.random.random(size=(10000, 1))
    m = TF_Reg_Model()
    m.build_model(100, 50)
    # m.train_model(x, y, acc, n_epochs=10, model_name='testing_istar')
    # m.save_model('testing_istar')

    # load and predict
    m.load_model('testing_istar')
    test_x = np.random.random(size=(1, 100))
    test_b = np.random.random(size=(1, 1))
    pred_y = m.predict(test_x, test_b)
    print(pred_y)
