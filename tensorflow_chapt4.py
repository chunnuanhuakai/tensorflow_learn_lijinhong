# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 18:39:34 2018

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

# 重置图 只被当前线程使用，保证线程安全
tf.reset_default_graph()


# 创建模型
# 占位符
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
# 模型参数
W = tf.Variable(tf.random_normal([1]), name="weight")
bias = tf.Variable(tf.random_normal([1]), name="bias")
# 前向结构
Z = tf.multiply(X,W) + bias
tf.summary.histogram('Z',Z)

# 反向优化
cost = tf.reduce_mean(tf.square(Y-Z))   
tf.summary.scalar('loss_function', cost)

learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


## 迭代训练
# 初始化所有遍历
init = tf.global_variables_initializer()
# 生成 saver器 以及保存路径
saver = tf.train.Saver(max_to_keep=2)
savedir = 'log/' 
# 定义参数
training_epochs = 20
display_step =2

    
                       
#启动Session
with tf.Session() as sess:
    sess.run(init)
    
    # 合并merger  创建 summary
    mergeed_summary_op = tf.summary.merge_all()
    summary_write = tf.summary.FileWriter('log/linear_with_summaries', sess.graph)
    
    # 向模型写数据
    plotdata = {"batchsize":[], "loss":[]}
    for epoch in range(training_epochs):
        for (x,y) in zip(train_x, train_y):
            sess.run(optimizer, feed_dict={X:x, Y:y})
             # 生成 summary
            summary_str = sess.run(mergeed_summary_op, feed_dict={X:x, Y:y})
            summary_write.add_summary(summary_str, epoch)
            
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={X:train_x, Y:train_y})
            print("epoch:",epoch+1, "cost=", loss, "W=", sess.run(W))
            if not (loss=='NA'):
                plotdata['batchsize'].append(epoch)
                plotdata['loss'].append(loss)
            # 模型保存  保存检查点
            saver.save(sess, save_path= savedir + 'linermodel.cpkt',global_step=epoch)
#           

                
        
    print("Finshed!")
    print("cost=", sess.run(cost, feed_dict={X:train_x,Y:train_y}),"W=", sess.run(W), "b=",sess.run(bias))
    print("x=0.2, Z = ", sess.run(Z, feed_dict={X:0.2}))
    
#    # 模型保存  最终模型
#    saver.save(sess, save_path= savedir + 'linermodel.cpkt')
    
    
    ## 可视化
    plt.plot(train_x, train_y, 'ro',label='origin data')
    plt.plot(train_x, sess.run(W)*train_x + sess.run(bias), label='Fittedline')
    plt.legend()
    plt.show()
   
    
    
# 重启一个 sess, 载入检查点
with tf.Session() as sess2:
#    load_epoch =18
#    sess2.run(tf.global_variables_initializer())
#    saver.restore(sess2, savedir+'linermodel.cpkt-'+ str(load_epoch))
#    print("x=0.2, Z = ", sess2.run(Z, feed_dict={X:0.2}))
  
    kpt = tf.train.latest_checkpoint(savedir)
    if kpt != None:
        saver.restore(sess2, kpt)
    print("x=0.2, Z = ", sess2.run(Z, feed_dict={X:0.2}))
    
    
    
    
    
    
    
    
    
    
    
    