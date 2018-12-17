"""
This is the Factorization Machine 

auother: Yixin Su
"""

import tensorflow as tf
import numpy as np


class FM():
    def __init__(self, flags, feature_M, user_M, feature_n):
        self.flags = flags
        self.feature_M = feature_M
        self.user_M = user_M 
        self.feature_n = feature_n

        self.users_ph = tf.placeholder(tf.int32, shape=(None), name='users_ph')
        self.attr_ph = tf.placeholder(tf.int32, shape=(None, self.feature_n), name='movies_ph')
        self.ratings_ph = tf.placeholder(tf.float32, shape=(None), name='ratings_ph')

        self.build_model()
    
    def build_model(self):

        #U = tf.Variable(initial_value=tf.truncated_normal([self.user_M, self.flags.user_dim]), name='users_vec')
        #A = tf.Variable(initial_value=tf.truncated_normal([self.feature_M, self.flags.feature_dim]), name='features_vec')
        U = tf.Variable(initial_value=tf.random_normal([self.user_M, self.flags.user_dim], mean=0, stddev=0.01), name='users_vec')
        A = tf.Variable(initial_value=tf.random_normal([self.feature_M, self.flags.feature_dim], mean=0, stddev=0.01), name='features_vec')
        u_input = tf.nn.embedding_lookup(U, self.users_ph)
        a_input = tf.nn.embedding_lookup(A, self.attr_ph)


        self.permutate(a_input, u_input)
        self.predict = self.FM_net(a_input, u_input)

        #self.predict.assign(tf.maximum(tf.zeros_like(self.predict), self.predict))
        #self.predict.assign(tf.minimum(tf.ones_like(self.predict), self.predict))


        self.loss = tf.losses.mean_squared_error(self.ratings_ph, self.predict)
        #self.loss = tf.losses.log_loss(self.ratings_ph, self.predict)

        self.rmse = self.cal_rmse(self.predict, self.ratings_ph)

        lr = tf.constant(self.flags.lr, name='learning_rate')
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(lr, global_step, 10000, 0.98, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self.update = optimizer.minimize(self.loss, global_step=global_step)


    def permutate(self, features, users):
        feature_n = self.feature_n + 1 
        feature_dim = self.flags.feature_dim

        user_stack = tf.expand_dims(users, 1)
        features = tf.concat(features, user_stack)
        sta1 = tf.reshape(tf.stack([features for i in range(feature_n)], axis=1), [-1, feature_n*feature_n, feature_dim])
        sta2 = tf.reshape(tf.stack([features for i in range(feature_n)], axis=2), [-1, feature_n*feature_n, feature_dim])


        con = tf.concat([sta1, sta2], 2)
        con1 = tf.concat([con, user_stack], 2)

        return con1

    def FM_net(self, attr, user):
        return 0.

    def combine(self, layer_output, method):
        if method == 'sum':
            """
            simply sum together
            """
            return tf.reduce_sum(layer_output, axis=1) 
    
    def cal_rmse(self, predict, true_value):
        return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(predict, true_value)))) 
