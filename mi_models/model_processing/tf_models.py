# -*- coding: utf-8 -*-
# @time      : 2019/1/17 19:51
# @author    : yuxiaoqi@cmschina.com.cn
# @file      : tf_models.py

import tensorflow as tf
from tensorflow.contrib.layers import  fully_connected

class TF_Reg_Model():
    def __init__(self):
        pass

    def build_model(self, X, n_hidden1, n_hidden2,n_hidden3):
        with tf.name_scope('dnn'):
            h1 = fully_connected(X, n_hidden1, scope='hidden1',activation_fn=tf.nn.leaky_relu)
            h2 = fully_connected(h1, n_hidden2, scope='hidden2', activation_fn=tf.nn.leaky_relu)
            logits = fully_connected(h2, n_hidden1, scope='outputs', activation_fn=tf.nn.leaky_relu)