'''Export'''

# pylint: disable=E1101
# pylint: disable=E0611

import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder

in1 = tf.placeholder(dtype=tf.float32, shape=None, name='in1')

out1 = 1.0 * in1

# saver = saved_model_builder.SavedModelBuilder('./logs')

# with tf.Session() as save:

#     print(save.run(out1, feed_dict={in1: 1.0}))

#     saver.add_meta_graph_and_variables(save, ['training'])

#     saver.add_meta_graph( ['serving'])

#     saver.save()

with tf.Session() as load:

    tf.saved_model.loader.load(load, ['serving'], './logs')

    print(load.run(out1, feed_dict={in1: 2.0}))
