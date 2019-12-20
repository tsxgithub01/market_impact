# -*- coding: utf-8 -*-
# @time      : 2019/1/17 19:51
# @author    : yuxiaoqi
# @file      : tf_models.py

import os
import numpy as np
import tensorflow as tf
from ..utils.logger import Logger
from tensorflow.contrib.layers import fully_connected
from ..data_processing.dataset import DataSet
from ..utils.utils import get_parent_dir
from ..model_processing.models import Model
from tensorflow.python.tools import inspect_checkpoint as chkp

logger = Logger('log.txt', 'INFO', __name__).get_log()


class TFRegModel(Model):
    def __init__(self, learning_rate=0.01):
        super(TFRegModel, self).__init__('tf_dnn')
        self.learning_rate = learning_rate
        self._dataset = None
        self.training_op = None
        self.accuracy = None
        self.sess = None
        self._test_loss = []
        self._model_name = None
        self.summary_op = None

    def build_model(self, n_hidden1=10, n_hidden2=10, x_shape=(), acc_shape=(), n_hidden_layers=2, save_prefix=''):
        assert x_shape[0] == acc_shape[0], (
                "images.shape: %s labels.shape: %s" % (x_shape,
                                                       acc_shape))
        self.train_X = tf.placeholder(tf.float32, shape=(None, x_shape[1]), name='X')
        self.train_Y = tf.placeholder(tf.float32, shape=(None, 1), name="Y")
        # currently it is q/pov and price chg pct
        self.acc_input = tf.placeholder(tf.float32, shape=(None, acc_shape[1]), name='acc')
        _tmp = np.random.random()
        _ = tf.clip_by_value(_tmp, 0.1, 1.0)
        self.b1 = tf.Variable(_, name='b1')

        with tf.name_scope('dnn'):
            h1 = fully_connected(self.train_X, x_shape[1], scope='first_hidden', activation_fn=tf.nn.leaky_relu)
            h_layers = []
            for i in range(n_hidden_layers):
                _hlayer = fully_connected(h1, x_shape[1], scope='hidden{0}'.format(i), activation_fn=tf.nn.leaky_relu)
                h_layers.append(_hlayer)
            self.i_star = fully_connected(h_layers[-1], 1, scope='i_star', activation_fn=tf.nn.leaky_relu)
            # TODO check the value clip
            # self.i_star = tf.assign(self.i_star, tf.clip_by_value(self.i_star, 0, np.inf))
            self.i_star = tf.clip_by_value(self.i_star, 0, np.inf)
            acc_h1 = fully_connected(self.acc_input, acc_shape[1], scope='first_acc_hidden',
                                     activation_fn=tf.nn.leaky_relu)
            a_layers = []
            # acc_h2 = fully_connected(acc_h1, n_hidden2, scope='acc_hidden{0}', activation_fn=tf.nn.leaky_relu)
            for i in range(n_hidden_layers):
                _acc_h3 = fully_connected(acc_h1, acc_shape[1], scope='acc_hidden{0}'.format(i),
                                          activation_fn=tf.nn.leaky_relu)
                a_layers.append(_acc_h3)
            acc_h3 = fully_connected(a_layers[-1], 1, scope='last_acc_hidden', activation_fn=tf.nn.leaky_relu)
            # self.b1 = tf.assign(self.b1, tf.clip_by_value(self.b1, 0.1, 1.0))
            self.b1 = tf.clip_by_value(self.b1, 0.1, 1.0)
            self.tmp_impact = self.b1 * self.i_star * acc_h3
            self.perm_impact = (1 - self.b1) * self.i_star
            self.output = self.tmp_impact + self.perm_impact

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
            train_writer = tf.summary.FileWriter("E:\pycharm\\algo_trading\mi_remote\mi_models\data\models",
                                                 self.sess.graph)
            init.run()
            for epoch in range(n_epochs):
                logger.info('Run the {0} epoch out of {1}, with '.format(epoch, n_epochs))
                for iteration in range(self._dataset.num_examples // batch_size):
                    x_batch, y_batch, acc = self._dataset.next_batch(batch_size)
                    self.sess.run([self.training_op],
                                  feed_dict={self.train_X: x_batch, self.train_Y: y_batch, self.acc_input: acc})
                x_test, y_test, acc_test = self._dataset.next_batch(batch_size)
                test_loss, test_summary = self.sess.run([self.loss, self.summary_op],
                                                        feed_dict={self.train_X: x_test, self.train_Y: y_test,
                                                                   self.acc_input: acc_test})
                self._test_loss.append(test_loss)
                logger.info('epoch: {0}, test_loss:{1}'.format(epoch, test_loss))
                train_writer.add_summary(test_summary, epoch)
            if model_name:
                self._model_name = model_name
                model_path = os.path.join(get_parent_dir(), 'data', 'models', model_name)
                saver = tf.train.Saver()
                save_path = saver.save(self.sess, model_path)
                logger.info("Saved to path:{0}".format(save_path))
        self._test_loss = np.array(self._test_loss)
        logger.info('mean for test loss:{0}, std:{1}, var:{2}'.format(self._test_loss.mean(), self._test_loss.std(),
                                                                      self._test_loss.var()))

    def save_model(self, model_name):
        self._model_name = model_name
        model_path = os.path.join(get_parent_dir(), 'data', 'models', model_name)
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, model_path)
        logger.info("Saved to path:{0}".format(save_path))

    def load_model(self, model_name):
        self._model_name = model_name
        model_path = os.path.join(get_parent_dir(), 'data', 'models', model_name)
        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess=self.sess,
                      save_path=model_path)

    def predict(self, x=None, acc=None):
        prediction = self.sess.run([self.output, self.tmp_impact, self.perm_impact, self.i_star],
                                   feed_dict={self.train_X: x, self.acc_input: acc})
        return prediction

    def output_model(self, path=None, modle_name=None):
        ret = chkp.print_tensors_in_checkpoint_file(file_name=modle_name, tensor_name=None, all_tensors=True,
                                                    all_tensor_names=True)
        print(ret)


if __name__ == '__main__':
    import numpy as np
    m = TFRegModel()
    m.build_model(x_shape=(1, 8), acc_shape=(1, 2))
    x = np.random.random(1600).reshape(200, 8)
    y = np.random.random(200).reshape(200, 1)

    acc = np.random.random(400).reshape(200, 2)
    m.train_model(x, y, acc, 20, 50, 'test')
    # m.output_model('test')
    m.load_model('test')
    for i in range(3):
        # m.load_model('tf_dnn')
        r = m.predict(np.random.random(8).reshape(1, 8), np.random.random(2).reshape(1, 2))
        total, tmp, perm, instant = r
        print(total[0], tmp[0], perm[0], instant[0])
