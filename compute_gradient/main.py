'''main.py. Here, I wanted to explore how to compute gradients for a function.'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# input
a = tf.placeholder(dtype=tf.float32, name='a')

# function. Here, it's y = x^3.
b = tf.pow(a, 3.0)

# computation of the gradient. It is 3*x^2.
c = tf.gradients(b, a, name='gradient')

init = [tf.local_variables_initializer()]

with tf.Session() as sess:

    sess.run(init)

    # execute graph. It should compute the gradient and substitute a = 5.0.
    g = sess.run(c, feed_dict={a: 5.0})

    print(g)

# In this section, I will implement an ODE.


def equation(state, t):
    '''ODE equation'''

    x1, x2 = tf.unstack(state)
    dx1 = x2
    dx2 = -1 * x1 + 1

    return tf.stack([dx1, dx2])


t = np.linspace(0, 1, 2)
init_state = tf.constant(value=[0.0, 0.0], dtype=tf.float32)

tensor_state, tensor_info = tf.contrib.integrate.odeint(
    equation, init_state, t, full_output=True)

with tf.Session() as ode:

    state, info = ode.run([tensor_state, tensor_info])

    x, v = state.T
    plt.plot(t, x)
    plt.show()

    print(x, t)
    