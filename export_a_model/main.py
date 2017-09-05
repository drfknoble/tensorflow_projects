'''Export'''

# pylint: disable=E1101
# pylint: disable=E0611

import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder

in1 = tf.placeholder(dtype=tf.float32, shape=None, name='in1')

with tf.name_scope('network'):

    W = tf.get_variable(name='W', shape=[1],
                        dtype=tf.float32, initializer=tf.constant_initializer(1.0))

out1 = W * in1

init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

# saver = saved_model_builder.SavedModelBuilder('./logs')

# with tf.Session() as save:

#     save.run(init)

#     print(save.run(out1, feed_dict={in1: 1.0}))

#     saver.add_meta_graph_and_variables(save, ['training'])

#     saver.add_meta_graph( ['serving'])

#     saver.save()

with tf.Session() as load:

    load.run(init)

    tf.saved_model.loader.load(load, ['serving'], './logs')

    print(load.run(out1, feed_dict={in1: 2.0}))
