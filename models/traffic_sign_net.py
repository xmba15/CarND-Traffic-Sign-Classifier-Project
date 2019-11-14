#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

if tf.__version__[0] == "2":
    import tensorflow.compat.v1 as tf

    tf.disable_eager_execution()
    tf.disable_v2_behavior()


class TrafficSignNet(object):
    def __init__(
        self,
        in_channel=3,
        n_out=43,
        learning_rate=0.001,
        beta=0.001,
        initializer=tf.initializers.he_normal,
    ):
        self.in_channel = in_channel
        self.n_out = n_out
        self.learning_rate = learning_rate
        self.initializer = initializer
        self.beta = beta

        self.x = tf.placeholder(tf.float32, (None, 32, 32, self.in_channel))
        self.y = tf.placeholder(tf.int32, (None))
        self.keep_prob = tf.placeholder(tf.float32)
        self.keep_prob_conv = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate)

        self.build_model()

    def build_model(self):
        self.conv1_W = tf.get_variable(
            name="conv1_w", shape=(3, 3, self.in_channel, 32), initializer=self.initializer()
        )
        self.conv1_b = tf.Variable(tf.zeros(32))

        self.conv1 = tf.add(tf.nn.conv2d(self.x, self.conv1_W, strides=[
            1, 1, 1, 1], padding="SAME"), self.conv1_b)
        self.conv1 = tf.nn.relu(self.conv1)

        self.conv2_W = tf.get_variable(
            name="conv2_w", shape=(3, 3, 32, 32), initializer=self.initializer()
        )
        self.conv2_b = tf.Variable(tf.zeros(32))
        self.conv2 = tf.add(tf.nn.conv2d(
            self.conv1, self.conv2_W, strides=[1, 1, 1, 1], padding="SAME"
        ), self.conv2_b)

        self.conv2 = tf.nn.relu(self.conv2)
        self.conv2 = tf.nn.max_pool(
            self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID"
        )
        self.conv2 = tf.nn.dropout(self.conv2, self.keep_prob_conv)

        self.conv3_W = tf.get_variable(
            name="conv3_w", shape=(3, 3, 32, 64), initializer=self.initializer()
        )
        self.conv3_b = tf.Variable(tf.zeros(64))
        self.conv3 = tf.add(tf.nn.conv2d(
            self.conv2, self.conv3_W, strides=[1, 1, 1, 1], padding="SAME"
        ), self.conv3_b)
        self.conv3 = tf.nn.relu(self.conv3)

        self.conv4_W = tf.get_variable(
            name="conv4_w", shape=(3, 3, 64, 64), initializer=self.initializer()
        )
        self.conv4_b = tf.Variable(tf.zeros(64))
        self.conv4 = tf.add(tf.nn.conv2d(
            self.conv3, self.conv4_W, strides=[1, 1, 1, 1], padding="SAME"
        ), self.conv4_b)
        self.conv4 = tf.nn.relu(self.conv4)
        self.conv4 = tf.nn.max_pool(
            self.conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID"
        )
        self.conv4 = tf.nn.dropout(self.conv4, self.keep_prob_conv)

        self.conv5_W = tf.get_variable(
            name="conv5_w", shape=(3, 3, 64, 128), initializer=self.initializer()
        )
        self.conv5_b = tf.Variable(tf.zeros(128))
        self.conv5 = tf.add(tf.nn.conv2d(
            self.conv4, self.conv5_W, strides=[1, 1, 1, 1], padding="SAME"
        ), self.conv5_b)
        self.conv5 = tf.nn.relu(self.conv5)

        self.conv6_W = tf.get_variable(
            name="conv6_w", shape=(3, 3, 128, 128), initializer=self.initializer()
        )
        self.conv6_b = tf.Variable(tf.zeros(128))
        self.conv6 = tf.add(tf.nn.conv2d(
            self.conv5, self.conv6_W, strides=[1, 1, 1, 1], padding="SAME"
        ), self.conv6_b)
        self.conv6 = tf.nn.relu(self.conv6)
        self.conv6 = tf.nn.max_pool(
            self.conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID"
        )
        self.conv6 = tf.nn.dropout(self.conv6, self.keep_prob_conv)

        self.fc0 = tf.reshape(self.conv6, [tf.shape(self.conv6)[0], -1])
        self.fc1_W = tf.get_variable(
            name="fc1_w", shape=(2048, 128), initializer=self.initializer()
        )

        self.fc1 = tf.matmul(self.fc0, self.fc1_W)
        self.fc1 = tf.nn.relu(self.fc1)
        self.fc1 = tf.nn.dropout(self.fc1, self.keep_prob)

        self.fc2_W = tf.get_variable(
            name="fc2_w", shape=(128, 128), initializer=self.initializer()
        )
        self.fc2 = tf.matmul(self.fc1, self.fc2_W)

        self.fc2 = tf.nn.relu(self.fc2)
        self.fc2 = tf.nn.dropout(self.fc2, self.keep_prob)

        self.fc3_W = tf.get_variable(
            name="fc3_w", shape=(128, self.n_out), initializer=self.initializer()
        )
        self.logits = tf.matmul(self.fc2, self.fc3_W)

        self.one_hot_y = tf.one_hot(self.y, self.n_out)
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits, labels=self.one_hot_y
        )
        self.loss_operation = tf.reduce_mean(self.cross_entropy)
        self.regularizers = tf.nn.l2_loss(self.conv1_W) + tf.nn.l2_loss(self.conv2_W) + \
            tf.nn.l2_loss(self.conv3_W) + tf.nn.l2_loss(self.conv4_W) + \
            tf.nn.l2_loss(self.conv5_W) + tf.nn.l2_loss(self.conv6_W)
        self.loss_operation = tf.reduce_mean(
            self.loss_operation + self.beta * self.regularizers)

        self.training_operation = self.optimizer.minimize(self.loss_operation)
        self.correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.one_hot_y, 1)
        )
        self.accuracy_operation = tf.reduce_mean(
            tf.cast(self.correct_prediction, tf.float32)
        )
        self.saver = tf.train.Saver()

    def predict(self, X_data, BATCH_SIZE=64):
        num_examples = len(X_data)
        y_pred = np.zeros(num_examples, dtype=np.int32)
        sess = tf.get_default_session()

        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x = X_data[offset: offset + BATCH_SIZE]
            y_pred[offset: offset + BATCH_SIZE] = sess.run(
                tf.argmax(self.logits, 1),
                feed_dict={
                    self.x: batch_x,
                    self.keep_prob: 1,
                    self.keep_prob_conv: 1,
                    self.is_training: False,
                },
            )

        return y_pred

    def evaluate(self, X_data, y_data, BATCH_SIZE=64):
        num_examples = len(X_data)
        total_accuracy = 0
        total_loss = 0
        sess = tf.get_default_session()

        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = (
                X_data[offset: offset + BATCH_SIZE],
                y_data[offset: offset + BATCH_SIZE],
            )
            accuracy, loss = sess.run(
                [self.accuracy_operation, self.loss_operation],
                feed_dict={
                    self.x: batch_x,
                    self.y: batch_y,
                    self.keep_prob: 1,
                    self.keep_prob_conv: 1,
                    self.is_training: False,
                },
            )
            total_accuracy += accuracy * len(batch_x)
            total_loss += loss * len(batch_x)

        return total_accuracy / num_examples, total_loss / num_examples
