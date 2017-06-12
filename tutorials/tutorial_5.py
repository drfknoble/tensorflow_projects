'''Tutorial_5'''

# pylint: disable=C0413
# pylint: disable=C0103
# pylint: disable=E0401

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppresses warnings

if not os.path.exists('./model'):
    os.makedirs('./model')

A = tf.placeholder(tf.float32, None, 'A')
B = tf.placeholder(tf.float32, None, 'B')

W = tf.Variable(0)

C = A + B

init = [tf.global_variables_initializer()]

with tf.Session() as s:

    s.run(init)

    saver = tf.train.Saver()

    feed_dict = {A: 2.0, B: 4.0}
    print(s.run(C, feed_dict=feed_dict))

    saver.save(s, './model/main.ckpt', 0)

with tf.Session() as l:

    loader = tf.train.Saver()

    ckpt = tf.train.latest_checkpoint('./model/')

    loader.restore(l, ckpt)

    feed_dict = {A: 2.0, B: 4.0}
    print(l.run(C, feed_dict=feed_dict))
    