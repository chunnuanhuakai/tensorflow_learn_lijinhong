

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 21:00:59 2018

@author: liuHongBing
"""

import tensorflow as tf
import numpy as np
import pylab

from tensorflow.examples.tutorials.mnist import input_data

learning_rate = 0.001
n_input = 784
n_hidden_1 = 256
n_hidden_2 = 256
n_label = 10


##  input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_label])


## peremeter
weight = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=0.1)),
        'h2': tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2], stddev=0.1)),
        'out': tf.Variable(tf.random_normal([n_hidden_2,n_label], stddev=0.1))
        }

bias = {
        'h1': tf.Variable(tf.zeros([n_hidden_1])),
        'h2': tf.Variable(tf.zeros([n_hidden_2])),
        'out': tf.Variable(tf.zeros([n_label]))
        }


layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weight['h1']), bias['h1']))

layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weight['h2']), bias['h2']))

y_pred = tf.add(tf.matmul(layer_2, weight['out']), bias['out'])

##cost

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


## load data
mnist = input_data.read_data_sets('F:\DATA\mnist', one_hot=True)

## create model
train_epoch = 20
batch_size = 100
display_step = 1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(train_epoch):
        total_epoch = int(len(mnist.train.images)/batch_size)
        for epoch in range(total_epoch):
            X,Y = mnist.train.next_batch(batch_size)
            _,cost2 = sess.run([optimizer, cost], feed_dict={x:X, y:Y})
        print('epoch:',i+1,'\t','cost:',cost2)
    print('finshed')
    
    ## add test  axis =1 行最大  axis = 0 列最大
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.arg_max(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    out_value = tf.argmax(y_pred, 1)
    acc, outvalue = sess.run([accuracy, out_value], feed_dict={x:mnist.test.images, y:mnist.test.labels})
    print(acc, outvalue)
    
    
    ## fetch a image
    image2 = mnist.train.images[1,:]
    im = image2.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()
    
    ## fetch a image
    batch_xs, batch_ys = mnist.train.next_batch(1)
    im1 = batch_xs[0].reshape(-1, 28)
    pylab.imshow(im1)
    pylab.show()
    
    outvalue = sess.run(out_value, feed_dict={x: image2.reshape(-1,784)})
    print('images predict label:', outvalue)
    


















