'''Tutorial 1. In this program, a simple mathematical expression is expressed as a graph.'''

# pylint: disable=C0413
# pylint: disable=C0103

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppresses warnings

A = tf.constant(value=1.0, dtype=tf.float32)
B = tf.constant(value=2.0, dtype=tf.float32)

C = A + B

init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

with tf.Session() as s:

    s.run(init)

    c = s.run(C)

    print(c)
