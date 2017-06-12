'''Activity_3'''

# pylint: disable=C0413
# pylint: disable=C0103
# pylint: disable=E0401

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppresses warnings

if not os.path.exists('./logs'):
    os.makedirs('./logs')

x = tf.placeholder(dtype=tf.float32, shape=None, name='x')
y = tf.placeholder(dtype=tf.float32, shape=None, name='y')

M = tf.Variable(initial_value=1.0, dtype=tf.float32)
c = tf.Variable(initial_value=0.1, dtype=tf.float32)

y_ = M *x + c

loss = tf.losses.mean_squared_error(y, y_)
training = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

tf.summary.scalar('l', loss)

init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

with tf.Session() as s:

    s.run(init)

    merged = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter('./logs', s.graph)

    feed_dict = {x : [1.0, 2.0, 3.0], y: [2.0, 4.0, 6.0]}
    for i in range(0, 10, 1):
        m, _ = s.run([merged, training], feed_dict)

        summary_writer.add_summary(m, i)
