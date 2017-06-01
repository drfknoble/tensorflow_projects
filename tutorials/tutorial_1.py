'''Tutorial 1. In this program, a simple mathematical expression is expressed as a graph.'''

# pylint: disable=C0413
# pylint: disable=C0103
# pylint: disable=E0401

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppresses warnings

A = tf.constant(value=1.0, dtype=tf.float32)
B = tf.constant(value=2.0, dtype=tf.float32)

C = A + B

with tf.Session() as s:

    print(s.run(C))

