'''main.py'''


import tensorflow as tf

# Input

a = tf.placeholder(dtype = tf.uint8, shape = [160, 120, 3], name = "a")

# Network

with tf.variable_scope("W"):
    W = tf.get_variable(name = "W", shape = [1], dtype = tf.uint8, initializer = tf.constant_initializer(value = [1], dtype = tf.uint8))

b = tf.multiply(a, W, name = "b")

init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

export_dir = "E:\\dev_libraries\\tensorflow\\tensorflow\\tf_cpp_3\\trained_model"

builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    
with tf.Session() as sess:

    sess.run(init)
    
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.TRAINING])

    builder.save()

