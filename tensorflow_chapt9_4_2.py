# -*- coding: utf-8 -*-

"""
Created on Thu Sep 20 22:45:35 2018

@author: liuHongBing
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("E:/liuhongbing/python/data/MNIST_data/",one_hot=True)


learning_rate = 0.001
n_input = 28
n_step = 28
n_hidden = 128
n_classes = 10
batch_size = 100
display_size = 1
train_epoch = 25

tf.reset_default_graph()

x = tf.placeholder(tf.float32, [None, n_step, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

x1 = tf.unstack(x, n_step, 2)


## create lstm cell
lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
## add dropout
lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, input_keep_prob=0.5)

lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
## add dropout
lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, input_keep_prob=0.5)

outputs,_,_ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell, x1, dtype=tf.float32)
## cost
pred = tf.contrib.layers.fully_connected(outputs[-1], n_classes, activation_fn=None)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels = y))

correct_predict = tf.equal(tf.arg_max(tf.nn.softmax(pred), 1), tf.arg_max(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

optiminzer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(train_epoch):
        
        avg_cost = 0
        total_batch = int(mnist.train.images.shape[0]/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs_shape = np.reshape(batch_xs,(-1,n_step, n_input))
            _, c = sess.run([optiminzer, cost], feed_dict={x:batch_xs_shape,y:batch_ys})
            avg_cost += c/total_batch
        if epoch % display_size ==0:
            print("epoch:", '%04d' % (epoch+1), 'cost=', format(avg_cost))
            
    test_image = np.reshape(mnist.test.images,  (-1, n_step, n_input))
    print('sess accuracy:', sess.run(accuracy, 
                                     feed_dict={x:test_image, y:mnist.test.labels}))   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    