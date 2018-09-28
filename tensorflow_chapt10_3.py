# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 21:29:36 2018

@author: liuHongBing
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('E:\liuhongbing\python\data\MNIST_data',one_hot=True)

learning_rate = 0.01
n_input = 784
n_hidden_1 = 256

train_X = mnist.train.images
train_Y = mnist.train.labels
test_X = mnist.test.images
test_Y = mnist.test.labels


## 占位符
X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_input])
drop_keep_prob = tf.placeholder(tf.float32)

# weight
weights = {
        'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_1])),
                                                   
        'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),


        }
# biases
biases = {
        'encoder_b1': tf.Variable(tf.zeros([n_hidden_1])),
        'encoder_b2': tf.Variable(tf.zeros([n_hidden_1])),
        'decoder_b1': tf.Variable(tf.zeros([n_input])),
        }


# 去噪自编码器
def denoise_auto_encoder(_X, _weights,_biases,_keep_prob):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['encoder_h1']), biases['encoder_b1']))
    layer_1out = tf.nn.dropout(layer_1, _keep_prob)
    
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1out, _weights['encoder_h2']), biases['encoder_b2']))
    layer_2out = tf.nn.dropout(layer_2, _keep_prob)
    
    return tf.nn.sigmoid(tf.matmul(layer_2out, _weights['decoder_h1'])+_biases['decoder_b1'])
## 输出的节点
reconstruction = denoise_auto_encoder(X, weights,biases, drop_keep_prob)

## cost
cost = tf.reduce_mean(tf.pow(Y-reconstruction, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


## 训练参数

training_epochs = 25
batch_size = 256
display_step =5

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_batch = int(mnist.train.num_examples/batch_size)
    
    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            noisy_xs = batch_xs + 0.3*np.random.randn(batch_size, 784)
            
            _,c = sess.run([optimizer, cost], feed_dict={X:noisy_xs, Y:batch_xs, drop_keep_prob:.5})
            
        if epoch % display_step ==0:
            print("epoch:", '%04d' % (epoch+1), 'cost=', '{:.9f}'.format(c))
    print('finshed!')

    
    show_num = 10
    test_noisy = mnist.test.images[:show_num] + 0.3*np.random.randn(show_num, 784)
    encoder_decode = sess.run(reconstruction, feed_dict={X:test_noisy, drop_keep_prob:1.})
    
    f, a = plt.subplots(3, 10, figsize=(10, 3))
    for i in range(show_num):
        a[0][i].imshow(np.reshape(test_noisy[i], (28,28)))
        a[1][i].imshow(np.reshape(mnist.test.images[i],(28,28)))
        a[2][i].imshow(np.reshape(encoder_decode[i],(28,28)),cmap=plt.get_cmap('gray'))
        
    plt.draw()
    

    

    
    
    
    
    
    
    
    
    
    
    
    