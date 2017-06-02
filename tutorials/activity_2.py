'''Activity_2'''

# pylint: disable=C0413
# pylint: disable=C0103
# pylint: disable=E0401
# pylint: disable=E1101

import os
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppresses warnings


def setup():
    '''Setup data.'''
    x_setup = []
    y_setup = []
    for i_setup in range(0, 10, 1):
        x_setup.append(i_setup)
        y_setup.append(2*np.power(i_setup, 2))

    return [x_setup, y_setup]


x_data, y_data = setup()

print(x_data)
print(y_data)

x = tf.placeholder(dtype=tf.float32, shape=[1], name='A')
y = tf.placeholder(dtype=tf.float32, shape=[1], name='B')

a0 = tf.get_variable(name='a0', shape=[1], dtype=tf.float32,
                     initializer=tf.random_uniform_initializer(minval=0.1, maxval=10))
a1 = tf.get_variable(name='a1', shape=[1], dtype=tf.float32,
                     initializer=tf.random_uniform_initializer(minval=0.1, maxval=10))
a2 = tf.get_variable(name='a2', shape=[1], dtype=tf.float32,
                     initializer=tf.random_uniform_initializer(minval=0.1, maxval=10))

y_ = tf.multiply(a0, tf.square(x)) + tf.multiply(a1, x) + a2

loss = tf.losses.mean_squared_error(y, y_)
training = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

with tf.Session() as s:

    s.run(init)

    for i in range(0, 10000, 1):
        l, _ = s.run([loss, training],
                     feed_dict={x: [x_data[i % 10]], y: [y_data[i % 10]]})

        if i % 1000 == 0:
            print(l)

    print(s.run([a0, a1, a2]))
