'''Activity_1.'''

# pylint: disable=C0413
# pylint: disable=C0103
# pylint: disable=E0401

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppresses warnings

x = tf.placeholder(dtype=tf.float32, shape=None, name='x')

y = tf.multiply(3.0, tf.square(x)) + tf.multiply(10.0, x) + 5.0

with tf.Session() as s:

    print(s.run(y, feed_dict={x: 2.0}))

