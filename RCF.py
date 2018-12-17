"""
This is the Relational Collaborative Filtering model

auother: Yixin Su
"""

import tensorflow as tf
import numpy as np


class RCF():
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
        U = tf.Variable(initial_value=tf.random_normal([self.user_M, self.flags.user_dim], mean=0, stddev=0.05), name='users_vec')
        A = tf.Variable(initial_value=tf.random_normal([self.feature_M, self.flags.feature_dim], mean=0, stddev=0.05), name='features_vec')
        u_input = tf.nn.embedding_lookup(U, self.users_ph)
        a_input = tf.nn.embedding_lookup(A, self.attr_ph)


        self.predict = self.relation_net(a_input, u_input)

        self.predict.assign(tf.maximum(tf.zeros_like(self.predict), self.predict))
        self.predict.assign(tf.minimum(tf.ones_like(self.predict), self.predict))


        self.loss = tf.losses.mean_squared_error(self.ratings_ph, self.predict)
        #self.loss = tf.losses.log_loss(self.ratings_ph, self.predict)

        self.rmse = self.cal_rmse(self.predict, self.ratings_ph)

        lr = tf.constant(self.flags.lr, name='learning_rate')
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(lr, global_step, 10000, 0.98, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self.update = optimizer.minimize(self.loss, global_step=global_step)


    def permutate(self, features, users):
        feature_n = self.feature_n 
        feature_dim = self.flags.feature_dim

        user_stack = tf.stack([users for i in range(feature_n*feature_n)], 1)

        sta1 = tf.reshape(tf.stack([features for i in range(feature_n)], axis=1), [-1, feature_n*feature_n, feature_dim])
        sta2 = tf.reshape(tf.stack([features for i in range(feature_n)], axis=2), [-1, feature_n*feature_n, feature_dim])


        con = tf.concat([sta1, sta2], 2)
        con1 = tf.concat([con, user_stack], 2)

        return con1

    def relation_net(self, features, users):

        x_input = self.permutate(features, users)

        user_dim = self.flags.user_dim
        feature_dim = self.flags.feature_dim
        input_dim = feature_dim * 2 + user_dim
        x_input_reshape = tf.reshape(x_input, [-1, input_dim])

        #layer = tf.layers.dense(x_input_reshape, 10, activation=tf.nn.relu)
        #output_reshape = tf.layers.dense(layer, 1)
        output_reshape = tf.layers.dense(x_input_reshape, 1)

        self.output = tf.reshape(output_reshape, tf.shape(x_input)[:-1]) 

        result = self.combine(self.output, self.flags.combine_method)

        return result 

     

    def combine(self, layer_output, method):
        if method == 'sum':
            """
            simply sum together
            """
            return tf.reduce_sum(layer_output, axis=1) 
    
    def cal_rmse(self, predict, true_value):
        return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(predict, true_value)))) 
