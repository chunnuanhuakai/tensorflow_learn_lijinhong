# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 20:52:52 2018

@author: liuHongBing
"""

import tensorflow as tf
import numpy as np
tf.reset_default_graph()

X = np.random.randn(2, 4, 5)
X[1,1:] = 0
seq_lengths = [4, 1]

## LSTM  创建   cell
## tf.contrib计算图中的  
## 网络层、正则化、摘要操作、是构建计算图的高级操作，
## 但是tf.contrib包含不稳定和实验代码，有可能以后API会改变
cell = tf.contrib.rnn.BasicLSTMCell(num_units=3, state_is_tuple=True)

## GRU    创建   cell
gru = tf.contrib.rnn.GRUCell(3)


## 创建  动态RNN
outputs, last_states = tf.nn.dynamic_rnn(cell, X, seq_lengths, dtype=tf.float64)

gruoutput, grulast_states = tf.nn.dynamic_rnn(gru, X, seq_lengths, dtype=tf.float64)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

result, sta, gruresult, grusta = sess.run([outputs, last_states,gruoutput,grulast_states])
print("全序列：", result[0])
print("短序列：", result[1])
print("LSTMstate：", len(sta),'\n',sta[1])
print("GRU短序列：",gruresult[1])
print("GRUstate：", len(grusta), grusta[1])
































