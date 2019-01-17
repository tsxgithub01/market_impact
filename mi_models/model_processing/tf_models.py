# -*- coding: utf-8 -*-
# @time      : 2019/1/17 19:51
# @author    : yuxiaoqi@cmschina.com.cn
# @file      : tf_models.py

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data


class TF_Reg_Model():
    def __init__(self):
        self.learning_rate = 0.01

    def build_model(self, n_hidden1, n_hidden2, n_hidden3):
        X = tf.placeholder(tf.float32, shape=(None, n_hidden1), name='X')
        Y = tf.placeholder(tf.int64, shape=(None), name="Y")
        with tf.name_scope('dnn'):
            h1 = fully_connected(X, n_hidden1, scope='hidden1', activation_fn=tf.nn.leaky_relu)
            h2 = fully_connected(h1, n_hidden2, scope='hidden2', activation_fn=tf.nn.leaky_relu)
            logits = fully_connected(h2, n_hidden1, scope='outputs', activation_fn=tf.nn.leaky_relu)

        with tf.name_scope('loss'):
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
            loss = tf.reduce_mean(xentropy, name='loss')

        with tf.name_scope('train'):
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            training_op = optimizer.minimize(loss)

        with tf.name_scope('eval'):
            correct = tf.nn.in_top_k(logits, Y, 1)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        mnist = input_data.read_data_sets("/tmp/data/")

        n_epochs = 400
        batch_size = 50

        with tf.Session() as sess:
            init.run()
            for epoch in range(n_epochs):
                for iteration in range(mnist.train.num_examples // batch_size):
                    x_batch, y_batch = mnist.train.next_batch(batch_size)
                    sess.run(training_op, feed_dict={X: x_batch, Y: y_batch})
                acc_train = accuracy.eval(feed_dict={X: x_batch, Y: y_batch})
                acc_test = accuracy.evel(feed_dict={X: mnist.test.images, Y: mnist.test.labels
                                                    })
                print(epoch, 'train accuracy:', acc_train, 'test_accuracy:', acc_test)
            save_path = saver.save(sess, "my_model_final.ckpt")
            print(save_path)
