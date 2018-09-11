# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 21:00:59 2018

@author: liuHongBing
"""

import tensorflow as tf
import numpy as np

learning_rate = 1e-4
n_input = 2
n_label = 1
n_hidden = 15

##  input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_label])

## peremeter
weight = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.1)),
        'h2': tf.Variable(tf.random_normal([n_hidden,n_label], stddev=0.1))
        }

bias = {
        'h1': tf.Variable(tf.zeros([n_hidden])),
        'h2': tf.Variable(tf.zeros([n_label]))
        }


layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weight['h1']), bias['h1']))
y_pred = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weight['h2']), bias['h2']))

##cost

loss = tf.reduce_mean(tf.square(y_pred - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)



## create data
X = [[0,0],[0,1],[1,0],[1,1]]
Y = [[0], [1], [1], [0]]

X = np.array(X).astype('float32')
Y = np.array(Y).astype('int16')



## create model

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(20000):
        sess.run(optimizer, feed_dict={x:X,y:Y})
        if i%1000==0:
            print(sess.run(y_pred, feed_dict={x:X}))
    print(sess.run(layer_1, feed_dict={x:X}))
    


















