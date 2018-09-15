# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 21:25:39 2018

@author: liuHongBing
"""

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def get_one_hot(labels, num_class):
    m = np.zeros([labels.shape[0], num_class], dtype=np.int32)
    for i in range(labels.shape[0]):
        m[i][labels[i]] = 1
    return m

def generate(sample_size, mean,cov,diff, one_hot = True):
    num_classes = 3
    samples_per_class = int(sample_size/2)
    
    xo = np.random.multivariate_normal(mean, cov, samples_per_class)
    yo = np.zeros([samples_per_class],dtype=np.int32)

    #把list变成 索引-元素树，同时迭代索引和元素本身
    #生成均值为mean+d,协方差为cov sample_per_class x len(mean)个样本 类别为ci+1
    for ci, d in enumerate(diff):
        x1 = np.random.multivariate_normal(mean + d, cov, samples_per_class)
        y1 = (ci+1)*np.ones(samples_per_class,dtype=np.int32)
        
        xo = np.concatenate((xo, x1))
        yo = np.concatenate((yo, y1))
    if one_hot == False:
        yo = get_one_hot(yo, num_classes)
        
    return xo, yo


## create sample
    
np.random.seed(10)

num_feature = 2
num_classes = 3
mean = np.random.randn(num_feature)
cov = np.eye(num_feature)
X, Y = generate(2000, mean, cov, [[3.0,0.0],[3.0,3.0]],False)
aa = [np.argmax(l) for l in Y]

colors = ['r' if l==0 else 'b' if l==1 else 'y' for l in aa[:]]
plt.scatter(X[:,0],X[:,1], c = colors)
plt.xlabel("scaled age (in yrs)")
plt.ylabel("tummor size (in cm)")

plt.show()





## input

label_dim = num_classes

input_feature = tf.placeholder(tf.float32, [None, num_feature])
input_label = tf.placeholder(tf.float32, [None, label_dim])

## parameter

weight = tf.Variable(tf.random_normal([num_feature, label_dim]), name = 'weight')
bias = tf.Variable(tf.zeros([label_dim]), name='bias')

## output
Z = tf.matmul(input_feature, weight) + bias

##cost 
preb = tf.nn.softmax_cross_entropy_with_logits(logits = Z, labels=input_label)
loss = tf.reduce_mean(preb)
optimizer = tf.train.AdamOptimizer(0.04).minimize(loss)
## count wrong sample nums

Z_softmax = tf.nn.softmax(Z)
Z_softmax_index = tf.argmax(Z_softmax, axis = 1) # find the max index by row
err = tf.count_nonzero(Z_softmax_index - tf.argmax(input_label, axis = 1))


## create session

max_epoch = 50
minibatchSize = 25

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(max_epoch):
        summerr = 0
        for i in range(np.int32(len(Y)/minibatchSize)):
            x1 = X[i*minibatchSize:(i+1)*minibatchSize,:]
            y1 = Y[i*minibatchSize:(i+1)*minibatchSize,:]
            
            _,loss2, err2 = sess.run([optimizer, loss, err], feed_dict={input_feature:x1, input_label:y1})

            
            summerr = summerr +  err2/minibatchSize
        print('epoch:',epoch, '\t','loss:',loss2,'\t', "sumerr:",summerr)   
    train_x, train_y = X,Y
    label_value = [np.argmax(l) for l in train_y]
    
    colors = ['r' if l==0 else 'b' if l==1 else 'y' for l in label_value]
    plt.scatter(train_x[:, 0], train_x[:, 1], c=colors)
    
    x = np.linspace(-1, 8, 200)
    y =( -x* sess.run(weight)[0][0] - sess.run(bias)[0])/sess.run(weight)[1][0]
    plt.plot(x, y, label='first line')
    
    
    y =( -x* sess.run(weight)[0][1] - sess.run(bias)[1])/sess.run(weight)[1][1]
    plt.plot(x, y, label='second line')
    

    y =( -x* sess.run(weight)[0][2] - sess.run(bias)[2])/sess.run(weight)[1][2]
    plt.plot(x, y, label='third line')
    
    plt.legend()
    plt.show()
    print(sess.run(weight), sess.run(bias))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    