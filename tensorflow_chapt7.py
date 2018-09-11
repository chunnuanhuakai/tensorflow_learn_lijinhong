# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 15:59:24 2018

@author: zbj
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


def generate(sample_size, mean, cov, diff, regression):
    num_classes = 2
    sample_pre_class = int(sample_size/2)
    
    x0 = np.random.multivariate_normal(mean, cov, sample_pre_class)
    y0 = np.zeros(sample_pre_class)
    
    for ci, d in enumerate(diff):
        x1 = np.random.multivariate_normal(mean+d, cov, sample_pre_class)
        y1 = (ci+1)*np.ones(sample_pre_class)
        
        x0 = np.concatenate((x0, x1))
        y0 = np.concatenate((y0, y1))
        
    if regression==False:
        class_ind = [y==class_number for class_number in range(num_classes)]
        y = np.asarray(np.hstack(class_ind), dtype=np.float32)
        
    
    return x0, y0
    
    
np.random.seed(10)
mean = np.array([0,0])
cov = np.eye(2)
x, y = generate(1000, mean, cov, [3],True)
#colors = ['r' if l ==0 else 'b' for l in y[:]]
#plt.scatter(x[:,0], x[:,1],c=colors)
#plt.xlabel('scaled age (in cm)')
#plt.ylabel('tumor size (in cm)')
#plt.show()


##  create logistic model

# input data

input_dim = 2
label_dim = 1

input_features = tf.placeholder(tf.float32, [None, input_dim])
input_labels = tf.placeholder(tf.float32, [None, label_dim])


# weight  parameter

W = tf.Variable(tf.random_normal([input_dim, label_dim]), name='weight')
b = tf.Variable(tf.zeros([label_dim]), name='bias')

# cost 
Z = tf.matmul(input_features, W) + b
preb = tf.nn.sigmoid(Z)


##  attention : cross_entropy
#cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = Z, labels = input_labels)

cross_entropy = -(input_labels * tf.log(preb)) + (1- tf.log(preb))*(1-input_labels)
loss = tf.square(preb-input_labels)

cost_cross_entropy = tf.reduce_mean(cross_entropy)
cost_squre = tf.reduce_mean(loss)

optimizer = tf.train.AdamOptimizer(learning_rate = 0.04).minimize(cost_cross_entropy)


##  create session:
max_epoch = 50
minibatchSize = 25
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(max_epoch):
        summer = 0
        for i in range(np.int32(len(y)/minibatchSize)):
            x1 = x[i*minibatchSize:(i+1)*minibatchSize,:]
            y1 = np.reshape(y[i*minibatchSize:(i+1)*minibatchSize],[-1,1])
            
            _, cce, cs =sess.run([optimizer, cost_cross_entropy,cost_squre], feed_dict={input_features:x1, input_labels:y1})
        print('epoch:', epoch, '\t',"cost_cross_entroy:",cce,'\t', 'cost_squre',cs)
        
        
        
        ## plot
    colors = ['r' if l ==0 else 'b' for l in y[:]]
    plt.scatter(x[:,0],x[:,1], c= colors)
    
    x_test=np.linspace(-1, 8, 200)
    print(sess.run(W)[0], sess.run(W)[1], sess.run(b))
    
    y_test =(-x_test * sess.run(W)[0]-sess.run(b))/(sess.run(W)[1])
    
    plt.plot(x_test, y_test, label = 'fitted line')
    plt.show()


        
        
        











