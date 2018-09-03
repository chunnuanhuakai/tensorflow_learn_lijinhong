# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 20:57:55 2018

@author: zbj
"""

import tensorflow as tf

hello = tf.constant('hello ,Tensorflow!')
print(hello)
sess = tf.Session()
print(hello)
print(sess.run(hello))
sess.close()


a= tf.constant(3)
b = tf.constant(4)
with tf.Session() as sess:
    print("相加：%i" % sess.run(a+b))
    print("相乘：%i" % sess.run(a*b))

    
    
a = tf.placeholder(dtype=tf.int16)
b = tf.placeholder(dtype=tf.int16)


    
    
add = tf.add(a,b)
mul = tf.multiply(a,b)

with tf.Session() as sess:
    with tf.device("gpu:1"):
        print("相加： %i" % sess.run(add, feed_dict={a:3, b:4}))
        print("相乘： %i" % sess.run(mul, feed_dict={a:3, b:4}))
        
        
        
from sklearn.svm import SVC
import numpy as np

X=np.array([[1,1],[1,2],[1,3],[1,4],[2,1],[2,2],[3,1],[4,1],[5,1],

       [5,2],[6,1],[6,2],[6,3],[6,4],[3,3],[3,4],[3,5],[4,3],[4,4],[4,5]])

Y=np.array([1]*14+[-1]*6)


T = np.array([[0.5,0.5],[1.5,1.5],[3.5,3.5],[4,5.5]])


svc = SVC(kernel='poly',degree=2,gamma=1,coef0=0)
svc.fit(X,Y)
pre = svc.predict_proba(T)
print(pre)
print(svc.n_support_)
print(svc.support_)
print(svc.support_vectors_)


##  multi-class classification
X = [[0],[1],[2],[3]]
Y = [0, 1, 2, 3]

clf = SVC(C=1.0, decision_function_shape='ovo',probability = True)
clf.fit(X, Y)
dec = clf.decision_function([1])
dec
dec.shape

result1 = clf.predict_proba([1])
result2 = clf.predict_proba([2])


prob_per_class_dictionary = dict(zip(clf.classes_, result1))
prob_per_class_dictionary2 = dict(zip(clf.classes_, result2))


results_ordered_by_probability1 = map(lambda x: x[0], sorted(zip(clf.classes_, result1), key=lambda x: x[1], reverse=True))
results_ordered_by_probability2 = map(lambda x: x[0], sorted(zip(clf.classes_, result2), key=lambda x: x[1], reverse=True))



sess = tf.Session()
print(sess.run(tf.random_normal([1])))
print(sess.run(tf.random_normal([5])))
print(sess.run(tf.random_normal(5)))
print(sess.run(tf.random_normal([5,3])))
sess.close()










