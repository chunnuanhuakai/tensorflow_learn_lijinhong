# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 10:22:33 2018

@author: liuHongBing
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mping
import tensorflow as tf
import os

mying = mping.imread(os.path.abspath('.')+'/chapter8/data/lyl2.jpg')
plt.imshow(mying)
plt.axis('off')
plt.show()
print(mying.shape)


full = np.reshape(mying, [1, 4032, 3024, 3])
#inputfull = tf.Variable(tf.constant(tf.float32, shape = [1, 4032, 3024, 3]))

inputfull = tf.placeholder(tf.float32, shape = [1, 4032, 3024, 3])

filter = tf.Variable(tf.constant([[-1.0, -1.0, -1.0],[0,0,0],[1.0,1.0,1.0],
                                 [-2.0, -2.0, -2.0],[0,0,0],[1.0,1.0,1.0]],shape=[3,3,3,1]))
## conv 处理
op = tf.nn.conv2d(inputfull, filter, strides=[1,1,1,1], padding='SAME')

## 归一化 处理
o = tf.cast(  (op-tf.reduce_min(op))/(tf.reduce_max(op) - tf.reduce_min(op))*255, tf.uint8)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    t, f = sess.run([o, filter], feed_dict={inputfull:full})
    
    t = np.reshape(t, [4032,3024])
    
    plt.imshow(t)
   # plt.axis('off')
    plt.show()