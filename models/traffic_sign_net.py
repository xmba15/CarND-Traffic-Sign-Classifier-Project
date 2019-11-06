#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class TrafficSignNet(object):
    def __init__(
        self,
        x,
        y,
        keep_prob,
        keep_prob_conv,
        n_out=43,
        mu=0,
        sigma=0.1,
        learning_rate=0.001,
    ):
        self.mu = mu
        self.sigma = sigma
        initializer = tf.initializers.he_normal()

        self.conv1_W = tf.get_variable(
            name="conv1_w", shape=(3, 3, 1, 32), initializer=initializer
        )

        self.conv1 = tf.nn.conv2d(x, self.conv1_W, strides=[
                                  1, 1, 1, 1], padding="SAME")

        self.conv1 = tf.nn.relu(self.conv1)

        self.conv2_W = tf.get_variable(
            name="conv2_w", shape=(3, 3, 32, 32), initializer=initializer
        )

        self.conv2 = tf.nn.conv2d(
            self.conv1, self.conv2_W, strides=[1, 1, 1, 1], padding="SAME"
        )

        self.conv2 = tf.nn.relu(self.conv2)

        self.conv2 = tf.nn.max_pool(
            self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID"
        )
        self.conv2 = tf.nn.dropout(self.conv2, keep_prob_conv)

        self.conv3_W = tf.get_variable(
            name="conv3_w", shape=(3, 3, 32, 64), initializer=initializer
        )

        self.conv3 = tf.nn.conv2d(
            self.conv2, self.conv3_W, strides=[1, 1, 1, 1], padding="SAME"
        )

        self.conv3 = tf.nn.relu(self.conv3)

        self.conv4_W = tf.get_variable(
            name="conv4_w", shape=(3, 3, 64, 64), initializer=initializer
        )

        self.conv4 = tf.nn.conv2d(
            self.conv3, self.conv4_W, strides=[1, 1, 1, 1], padding="SAME"
        )
        self.conv4 = tf.nn.relu(self.conv4)
        self.conv4 = tf.nn.max_pool(
            self.conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID"
        )
        self.conv4 = tf.nn.dropout(self.conv4, keep_prob_conv)  # dropout

        self.conv5_W = tf.get_variable(
            name="conv5_w", shape=(3, 3, 64, 128), initializer=initializer
        )

        self.conv5 = tf.nn.conv2d(
            self.conv4, self.conv5_W, strides=[1, 1, 1, 1], padding="SAME"
        )
        self.conv5 = tf.nn.relu(self.conv5)

        self.conv6_W = tf.get_variable(
            name="conv6_w", shape=(3, 3, 128, 128), initializer=initializer
        )

        self.conv6 = tf.nn.conv2d(
            self.conv5, self.conv6_W, strides=[1, 1, 1, 1], padding="SAME"
        )
        self.conv6 = tf.nn.relu(self.conv6)

        self.conv6 = tf.nn.max_pool(
            self.conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID"
        )
        self.conv6 = tf.nn.dropout(self.conv6, keep_prob_conv)

        self.fc0 = tf.reshape(self.conv6, [tf.shape(self.conv6)[0], -1])

        self.fc1_W = tf.get_variable(
            name="fc1_w", shape=(2048, 128), initializer=initializer
        )

        self.fc1 = tf.matmul(self.fc0, self.fc1_W)
        self.fc1 = tf.nn.relu(self.fc1)
        self.fc1 = tf.nn.dropout(self.fc1, keep_prob)

        self.fc2_W = tf.get_variable(
            name="fc2_w", shape=(128, 128), initializer=initializer
        )
        self.fc2 = tf.matmul(self.fc1, self.fc2_W)

        self.fc2 = tf.nn.relu(self.fc2)
        self.fc2 = tf.nn.dropout(self.fc2, keep_prob)

        self.fc3_W = tf.get_variable(
            name="w", shape=(128, n_out), initializer=initializer
        )
        self.logits = tf.matmul(self.fc2, self.fc3_W)

        self.one_hot_y = tf.one_hot(y, n_out)
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits, labels=self.one_hot_y
        )
        self.loss_operation = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.training_operation = self.optimizer.minimize(self.loss_operation)

        self.correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.one_hot_y, 1)
        )
        self.accuracy_operation = tf.reduce_mean(
            tf.cast(self.correct_prediction, tf.float32)
        )

        self.saver = tf.train.Saver()

    def y_predict(self, x, X_data, keep_prob, keep_prob_conv, BATCH_SIZE=64):
        num_examples = len(X_data)
        y_pred = np.zeros(num_examples, dtype=np.int32)
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x = X_data[offset: offset + BATCH_SIZE]
            y_pred[offset: offset + BATCH_SIZE] = sess.run(
                tf.argmax(self.logits, 1),
                feed_dict={x: batch_x, keep_prob: 1, keep_prob_conv: 1},
            )
        return y_pred

    def evaluate(self, x, y, X_data, y_data, keep_prob, keep_prob_conv, BATCH_SIZE=64):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = (
                X_data[offset: offset + BATCH_SIZE],
                y_data[offset: offset + BATCH_SIZE],
            )
            accuracy = sess.run(
                self.accuracy_operation,
                feed_dict={x: batch_x, y: batch_y,
                           keep_prob: 1.0, keep_prob_conv: 1.0},
            )
            total_accuracy += accuracy * len(batch_x)
        return total_accuracy / num_examples
