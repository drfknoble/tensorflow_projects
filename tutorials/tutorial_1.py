'''Tutorial 1.'''
'''In this program, a simple mathematical expression is expressed as a graph.'''

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

A = tf.placeholder(dtype=tf.float32, shape=None, name='A')
B = tf.placeholder(dtype=tf.float32, shape=None, name='B')

C = A + B

init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

with tf.Session() as s:

    s.run(init)

    c = s.run(C, feed_dict={A: 1.0, B: 2.0})

    print(c)
