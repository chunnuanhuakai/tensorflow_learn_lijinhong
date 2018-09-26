# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 21:29:36 2018

@author: liuHongBing
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('F:/DATA/mnist/',one_hot=True)

learning_rate = 0.01
n_hidden_1 = 256
n_hidden_2 = 128
n_input = 784


## 占位符
X = tf.placeholder(tf.float32, [None, n_input])
Y = X

# weight
weights = {
        'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
        'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
        }

biases = {
        'encoder_b1': tf.Variable(tf.zeros([n_hidden_1])),
        'encoder_b2': tf.Variable(tf.zeros([n_hidden_2])),
        'decoder_b1': tf.Variable(tf.zeros([n_hidden_1])),
        'decoder_b2': tf.Variable(tf.zeros([n_input])),
        }



# 编码
def encoder(x):
    
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, weights['encoder_h1']), biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2
    
# 解码
def decoder(x):
    
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    
    return layer_2
    
## 输出的节点
encoder_out = encoder(X)
pred = decoder(encoder_out)

## cost
cost = tf.reduce_mean(tf.pow(Y-pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


## 训练参数

training_epochs = 20
batch_size = 256
display_step =5

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_batch = int(mnist.train.num_examples/batch_size)
    
    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            _,c = sess.run([optimizer, cost], feed_dict={X:batch_xs})
            
        if epoch % display_step ==0:
            print("epoch:", '%04d' % (epoch+1), 'cost=', '{:.9f}'.format(c))
    print('finshed!')
    
    '''
    自编码器直接进行分类
    '''
    corect_predict = tf.equal(tf.arg_max(pred , 1), tf.arg_max(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(corect_predict, tf.float32))
    print('accuracy:', 1-sess.run(accuracy, feed_dict={X:mnist.test.images,
                                                       Y:mnist.test.images}))
        

    
    
    
    
    
    
    
    
    
    
    
    