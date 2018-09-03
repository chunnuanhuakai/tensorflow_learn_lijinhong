# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 21:25:54 2018

@author: zbj
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# init data
train_x = np.linspace(-1,1,100)
train_y = 2 * train_x + np.random.randn(train_x.shape[0]) * 0.3
                                        
plt.plot(train_x, train_y, 'ro', label='origin data')
plt.legend()
plt.show()

# 创建模型
# 占位符
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
# 模型参数
W = tf.Variable(tf.random_normal([1]), name="weight")
bias = tf.Variable(tf.random_normal([1]), name="bias")
# 前向结构
Z = tf.multiply(X,W) + bias

# 反向优化
cost = tf.reduce_mean(tf.square(Y-Z))   
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


## 迭代训练
# 初始化所有遍历
init = tf.global_variables_initializer()
# 定义参数
training_epochs = 20
display_step =2



def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
        
# return [val if idx<w else sum(a[(idx-w):idx])/w for idx,val in enumerate(a)]
    tmp =[]
    for idx, val in enumerate(a):
        
        if idx < w:
            tmp.append(val)
        else:
            tmp.append(sum(a[(idx-w):idx])/w)
    return tmp
                       
# 启动Session
with tf.Session() as sess:
    sess.run(init)
    plotdata = {"batchsize":[], "loss":[]}
    for epoch in range(training_epochs):
        for (x,y) in zip(train_x, train_y):
            sess.run(optimizer, feed_dict={X:x, Y:y})
            
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={X:train_x, Y:train_y})
            print("epoch:",epoch+1, "cost=", loss, "W=", sess.run(W))
            if not (loss=='NA'):
                plotdata['batchsize'].append(epoch)
                plotdata['loss'].append(loss)
                
                
    print("Finshed!")
    print("cost=", sess.run(cost, feed_dict={X:train_x,Y:train_y}),"W=", sess.run(W), "b=",sess.run(bias))
    print("x=0.2, Z = ", sess.run(Z, feed_dict={X:0.2}))
    
    ## 可视化
    plt.plot(train_x, train_y, 'ro',label='origin data')
    plt.plot(train_x, sess.run(W)*train_x + sess.run(bias), label='Fittedline')
    plt.legend()
    plt.show()
    
    plotdata['avgloss'] = moving_average(plotdata['loss'])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata['batchsize'], plotdata['avgloss'],'b--')
    plt.xlabel('minibatch number')
    plt.ylabel('loss')
    plt.title('minibatch run vs. training loss')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            
            
