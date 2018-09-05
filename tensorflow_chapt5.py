# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 18:35:00 2018

@author: zbj
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pylab

mnist = input_data.read_data_sets("E:/liuhongbing/python/data/MNIST_data/",one_hot=True)

print('输入数据：',mnist.train.images)
print('输入数据打印shape:', mnist.train.images.shape)
print('输入数据label shape:', mnist.train.labels.shape)
print('输入数据label:',mnist.train.labels)
#im = mnist.train.images[i+1]
#im = im.reshape(-1,28)
#pylab.imshow(im)
#pylab.show()

tf.reset_default_graph()

# define input 
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])


#------------- create model  ----------------------

# 1: define parameter  
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 2: define output node
pred = tf.nn.softmax(tf.matmul(x,W)+b)
nodevalue = tf.matmul(x,W)+b

# 3: define feedback  
#cost =tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), axis=1))
#cost = (-tf.reduce_sum(y*tf.log(pred), axis=1))
#cost = tf.nn.softmax_cross_entropy_with_logits(logits = nodevalue, labels = y)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = nodevalue, labels = y))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# --------------create model end ---------------------------
# --------------test model --------------------------------
correct_predict = tf.equal(tf.argmax(pred, 1),tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
#--------------end test model -----------------------------

saver = tf.train.Saver(max_to_keep=2)
savedir = './log_chapt5/mnist_model.ckpt'

#---------------model train--------------------------
training_epichs = 25
batch_size = 100
display_size = 1
#
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    for epoch in range(training_epichs):
#        avg_cost = 0
#        total_bactch = int(mnist.train.images.shape[0]/batch_size)
#        for i in range(total_bactch):
#            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#            _, c = sess.run([optimizer, cost], feed_dict={x:batch_xs,y:batch_ys})
#            avg_cost += c/total_bactch
#        if (epoch+1) % display_size == 0:
#            print("epoch:", '%04d' % (epoch+1), 'cost=', format(avg_cost))
#              
#            
##        print('accuracy', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))   
#        print('sess accuracy:', sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels}))
#        saver.save(sess, save_path = savedir) 
#        
#    print('finshed!')


    
##   restore train model
print('starting 2nd session-----')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #------------用到检查节点、方便后续迭代继续计算-----------------
#    model_file=tf.train.latest_checkpoint(savedir)
#    if model_file!=None:
#        saver.restore(sess,model_file)

   #------------读取保存的模型文件----------------------
    saver.restore(sess, savedir)
   
        
    # calc  准确度
    print('accuracy：', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))   
    # test model----> read 2 image 
    batch_xs, batch_ys = mnist.train.next_batch(2)
    output = tf.argmax(pred, 1)
    outputval, predv = sess.run([output, pred], feed_dict={x:batch_xs})
    print(outputval, predv, batch_ys)
    
    im = batch_xs[0]
    im = im.reshape([-1,28])
    pylab.imshow(im)
    pylab.show()
    
    im = batch_xs[1]
    im = im.reshape([-1,28])
    pylab.imshow(im)
    pylab.show()

























