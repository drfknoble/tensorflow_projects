'''Tutorial 4. In this program, a simple Multi-Layered Perceptron (MLP) is implemeted and trained. Placeholders are used to provide input.'''

# pylint: disable=C0413
# pylint: disable=C0103
# pylint: disable=E0401

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppresses warnings

x = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='x')
y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y')

def mlp_layer(in_x, w_shape, b_shape):
    '''mlp_layer'''
    W = tf.get_variable(name='W', shape=w_shape, dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=0.1, maxval=1))
    b = tf.get_variable(name='b', shape=b_shape, dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=0.1, maxval=1))

    # out_y = W*in_x + b
    out_y = tf.add(tf.matmul(in_x, W), b)

    return out_y

with tf.variable_scope('layer_1') as vs:
    h = mlp_layer(x, [1, 3], [3])
    vs.reuse_variables()

with tf.variable_scope('layer_2') as vs:
    y_ = mlp_layer(h, [3, 1], [1])


loss = tf.losses.mean_squared_error(y, y_)
training = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

with tf.Session() as s:

    s.run(init)

    for i in range(0, 100, 1):
        # l, _ = s.run([loss, training], feed_dict={x : [1.0], y: [2.0]})
        a = [1.0]
        b = [2.0]

        a = s.run(tf.reshape(a, [-1, 1]))
        b = s.run(tf.reshape(b, [-1, 1]))

        l, _ = s.run([loss, training], feed_dict={x : a, y: b})

        if i%10 == 0:
            print(l)

    print(s.run(["layer_1/W:0", "layer_1/b:0"]))

    print(s.run(y_, feed_dict={x: [[1.0]]}))

