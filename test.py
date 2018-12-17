import tensorflow as tf
import numpy as np


a = [[[1, 1, 1],[2,2, 2], [3,3,3]],[[3,3,3],[4,4,4], [5,5,5]],[[5,5,5], [6,6,6], [7,7,7]],[[7, 7,7], [8, 8,8], [9,9,9]]]
at = tf.constant(a,dtype=tf.float32)

feature_n = at.get_shape()[1]
vector_len = at.get_shape()[2]
print(feature_n, vector_len)


b = [[10, 10, 10], [11, 11, 11], [12, 12, 12], [13, 13, 13]]
bt = tf.constant(b, dtype=tf.float32)
bts = tf.stack([bt for i in range(feature_n*feature_n)], 1)

union = tf.concat([at, tf.expand_dims(bt, 1)], axis=1)


sta1 = tf.reshape(tf.stack([at for i in range(feature_n)], axis=1), [-1, feature_n*feature_n, vector_len])
sta2 = tf.reshape(tf.stack([at for i in range(feature_n)], axis=2), [-1, feature_n*feature_n, vector_len])

#print(sta1)

con = tf.concat([sta1, sta2], 2)
con1 = tf.concat([con, bts], 2)

#print(con1)

inputx = tf.reshape(con1, [-1, 9])

#W = tf.Variable(initial_value=tf.truncated_normal([8, 5]), name='weight')
W = tf.Variable(initial_value=tf.ones([9, 1]), name='weight')

result = tf.matmul(inputx, W)
result = tf.reshape(result, [-1, feature_n*feature_n, 1])

result1 = tf.reduce_sum(result, axis=1)
#print(result)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#print(sess.run(result))
#print(sess.run(result1))
print(sess.run(union))
