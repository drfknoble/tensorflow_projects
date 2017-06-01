'''Tutorial 2. In this program, a simple mathematical expression is expressed as a graph. 
Placeholders are used to provide input.'''

# pylint: disable=C0413
# pylint: disable=C0103
# pylint: disable=E0401

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppresses warnings

A = tf.placeholder(dtype=tf.float32, shape=None, name='A')
B = tf.placeholder(dtype=tf.float32, shape=None, name='B')

C = A + B

with tf.Session() as s:

    print(s.run(C, feed_dict={A: 1.0, B: 2.0}))


