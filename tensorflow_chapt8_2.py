# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 16:58:41 2018

@author: liuHongBing
"""


import tensorflow as tf
import pylab
import sys
import numpy as np
from tensorflow.contrib.layers.python.layers import batch_norm
sys.path.append('F:/python/git/tensorflow_learn_lijinhong/chapter8/models/tutorials/image/cifar10')
import cifar10_input

#batch_size = 128
#data_dir = 'F:/tmp/cifar10_data/cifar-10-batches-bin'
#
## fetch the test data , eval_data = True
## fetch the train data, eval_data = False
#images_test, labels_test = cifar10_input.inputs(eval_data = True, data_dir = data_dir,batch_size=batch_size)
#
#
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    # 定义协调器
#    coord = tf.train.Coordinator()
#    tf.train.start_queue_runners(sess, coord)
#    
#    image_batch, label_batch = sess.run([images_test, labels_test])
#    print('__\n', image_batch[0])
#    print('__\n', label_batch[0])
#    pylab.imshow(image_batch[0])
#    pylab.show()
#    
#    coord.request_stop()



batch_size = 128
data_dir = 'F:/tmp/cifar10_data/cifar-10-batches-bin'
print('begin')

images_train, labels_train = cifar10_input.inputs(eval_data = False, data_dir=data_dir,batch_size=batch_size)
images_test, labels_test = cifar10_input.inputs(eval_data = True, data_dir=data_dir,batch_size=batch_size)
print('begin data')

# fiter 
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
    
def conv2d(x,W):
    return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding = 'SAME')

def max_pool_2X2(x):
    return tf.nn.avg_pool(x, ksize = [1,2,2,1],strides=[1,2,2,1], padding = 'SAME')

def avg_pool_6X6(x):
    return tf.nn.avg_pool(x, ksize=[1,6,6,1],strides=[1,6,6,1], padding= 'SAME')

## batch--norm ： 不同层的权重更新不同，导致正向传播时，不同层的样本分布改变，影响训练效果
def batch_norm_layer(value, train = None, name = 'batch_norm'):
    if train is not None:
        # train : 不断更新每一层的 样本 均值和方差
        return batch_norm(value, decay = 0.9, updates_collections=None, is_training = True)
    else:
        # test : 使用训练时候的均值和方差
        return batch_norm(value, decay = 0.9, updates_collections=None, is_training=False)
    
##  input
X = tf.placeholder(tf.float32, [None, 24, 24, 3])
Y = tf.placeholder(tf.float32, [None, 10])
## 定义flag: 表明是训练过程，还是测试
train_flag = tf.placeholder(tf.float32)
##  filter ----> pooling ： 24*24-----> 12*12 【64】
W_conv1 = weight_variable(shape = [5,5,3,64])
b_conv1 = bias_variable(shape = [64])
x_image = tf.reshape(X, [-1, 24, 24, 3])
#h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
## 添加每一层  归一化
h_conv1 = tf.nn.relu(batch_norm_layer(conv2d(x_image, W_conv1) + b_conv1, train=train_flag))
h_pool1 = max_pool_2X2(h_conv1)  ##  12 * 12 

## filter ----> pooling :  12*12 -----> 6*6  【64】
W_conv2 = weight_variable(shape = [5,5,64,64])
b_conv2 = bias_variable(shape = [64])
#h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
## 添加每一层  归一化
h_conv2 = tf.nn.relu(batch_norm_layer(conv2d(h_pool1, W_conv2) + b_conv2, train=train_flag))
h_pool2 = max_pool_2X2(h_conv2) # 6 * 6
channel = 64 

### 第二层 卷积层： 5*5  改为   5*1  1*5两层--------BEGIN-----
#W_conv21 = weight_variable(shape = [5, 1, 64, 64])
#b_conv21 = bias_variable(shape = [64])
#h_conv21 = tf.nn.relu(conv2d(h_pool1, W_conv21) + b_conv21)
#
#W_conv2 = weight_variable(shape = [1,5,64,64])
#b_conv2 = bias_variable(shape = [64])
#h_conv2 = tf.nn.relu(conv2d(h_conv21, W_conv2) + b_conv2)
#h_pool2 = max_pool_2X2(h_conv2) # 6 * 6
#channel = 64 
### 第二层 卷积层： 5*5  改为   5*1  1*5两层--------END-------


### 第二层： 多通道卷积技巧： 5*5  7*7  3*3  1*1------------begin-------
#W_conv21_5X5 = weight_variable(shape = [5, 5, 64, 64])
#b_conv21_5X5 = bias_variable(shape = [64])
#h_conv21_5X5 = tf.nn.relu(conv2d(h_pool1, W_conv21_5X5), b_conv21_5X5)
#
#W_conv21_7X7 = weight_variable(shape = [7, 7, 64, 64])
#b_conv21_7X7 = bias_variable(shape = [64])
#h_conv21_7X7 = tf.nn.relu(conv2d(h_pool1, W_conv21_7X7), b_conv21_7X7)
#
#W_conv21_3X3 = weight_variable(shape = [3, 3, 64, 64])
#b_conv21_3X3 = bias_variable(shape = [64])
#h_conv21_3X3 = tf.nn.relu(conv2d(h_pool1, W_conv21_3X3), b_conv21_3X3)
#
#W_conv21_1X1 = weight_variable(shape = [1, 1, 64, 64])
#b_conv21_1X1 = bias_variable(shape = [64])
#h_conv21_1X1 = tf.nn.relu(conv2d(h_pool1, W_conv21_1X1), b_conv21_1X1)
#
#h_conv2 = tf.concat([h_conv21_5X5, h_conv21_7X7, h_conv21_3X3,h_conv21_1X1], 0)
#h_pool2 = tf.max_pool_2X2(h_conv2)
#channel = 256  # ----------> 64 * 4 = 256
### 第二层： 多通道卷积技巧： 5*5  7*7  3*3  1*1------------end-------


## filter ----> pooling : 6*6 -----> 1*1    【10】
W_conv3 = weight_variable(shape = [5,5,channel,10])
b_conv3 = bias_variable(shape = [10])
#h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
## 添加每一层  归一化
h_conv3 = tf.nn.relu(batch_norm_layer(conv2d(h_pool2, W_conv3) +b_conv3, train=train_flag))
h_pool3 = avg_pool_6X6(h_conv3) 

h_pool3_flat = tf.reshape(h_pool3, [-1, 10])
Y_conv = tf.nn.softmax(h_pool3_flat)

### 最后一层改为全连接------------------BEGIN -----------------------
#out_pool2 = tf.reshape(h_pool2, [-1, 6*6*64])
#out_weight = tf.Variable(tf.truncated_normal(shape=[6*6*64,10], stddev=0.1))
#bias = bias_variable(shape = [10])
#h_pool3_flat = tf.add(tf.matmul(out_pool2, out_weight), bias)
#Y_conv = tf.nn.softmax(h_pool3_flat)
### 最后一层全链接层---------------------END-------------------

## cost
cross_entropy = -tf.reduce_sum(Y*tf.log(Y_conv))

##  trainable = False  不参与训练，训练时不参与更新
global_step = tf.Variable(0, trainable = False)
decaylearning_rate = tf.train.exponential_decay(learning_rate = 0.04,
                     global_step=global_step, decay_steps=1000,decay_rate= 0.9)
## global_step=global_step minimize 记录更新步数，并实现自增操作
optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, global_step=global_step)

## accuracy
correct_prediction = tf.equal(tf.argmax(Y_conv, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess, coord)
    for i in range(15000):
        image_batch, label_batch = sess.run([images_train, labels_train])
        label_b = np.eye(10, dtype = float)[label_batch]
        
        sess.run([optimizer], feed_dict = {X:image_batch, Y:label_b})
         
        if i%200 == 0:
            train_accuracy = sess.run(accuracy, feed_dict = {X:image_batch,Y:label_b,train_flag : 1})
            print("step %d, train_accuray %g" % (i, train_accuracy))
    weigh1 = sess.run([W_conv1], feed_dict = {X:image_batch, Y:label_b})
