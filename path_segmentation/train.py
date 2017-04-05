'''path_segmentation'''

#%%

# pylint: disable=E0401
# pylint: disable=C0103
# pylint: disable=W0621

# Import. Here, we import tensorflow, which gives us access to the library.
import tensorflow as tf
import utilities as utils

# Here, we define important directories.
input_dir = './data/input/'
output_dir = './data/output/'

# Here, we define important file names.
record_file = output_dir + 'training_record.tfrecords'

# Here, we define the number of times we read a record file, and what size
# each batch is.
num_epochs = 1000
batch_size = 1

# Here, we create handles for reading and writing TFRecord files.
record = utils.input_pipeline([record_file], num_epochs, batch_size)

# Here, we define out input
with tf.name_scope('input'):

    x = tf.placeholder(tf.float32, shape=[None, None, None, None], name='x')
    y = tf.placeholder(tf.int32, shape=[None, None, None, None], name='y')

# Here, we define our graph.
with tf.name_scope('network'):

    token = x

    with tf.variable_scope('conv1_1'):

        kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(0, 0.2),
                                 shape=[3, 3, 3, 3], name='kernel')
        bias = tf.get_variable(initializer=tf.constant_initializer(0.1), shape=[3], name='bias')
        conv = tf.nn.conv2d(token, kernel, strides=[1, 1, 1, 1], padding='SAME')
        token = tf.nn.bias_add(conv, bias)

# Here, we define our output
with tf.name_scope('Output'):

    y_ = token

cost = tf.losses.mean_squared_error(y, y_)
optimizer = tf.train.AdamOptimizer(1e-2).minimize(cost)
tf.summary.scalar('Cost', cost)

# Initialisation commands
init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

# In this session, we read the TFRecord file and use its examples with our
# graph.
with tf.Session() as l:

    l.run(init)

    summary_writer = tf.summary.FileWriter('./logs', l.graph)

    merged = tf.summary.merge_all()

    saver = tf.train.Saver()

    model_dir = 'E:/Users/fknoble/My_FCN/' # for very large models.

    try:
        loader = tf.train.import_meta_graph(model_dir + 'model/main.meta')
        ckpt = tf.train.latest_checkpoint(model_dir + 'model/')
        loader.restore(l, ckpt)
    except IOError as e:
        saver.save(l, model_dir + '/model/main.ckpt', 0)
        saver.export_meta_graph(model_dir + 'model/main.meta')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        i = 0

        print('Step ' + str(i))

        while not coord.should_stop():

            feature, label = l.run(record)

            feed_dict = {x: feature, y: label}
            summary, c, _ = l.run([merged, cost, optimizer], feed_dict)

            summary_writer.add_summary(summary, i)

            if i % (num_epochs/10) == 0:
                print('Step ' + str(i))
                print('Cost: ' + str(c))

            if i % 50 == 0:
                saver.save(l, model_dir + 'model/main.ckpt', i)

            i += 1

    except tf.errors.OutOfRangeError:
        print('EoF')
    finally:
        coord.request_stop()

    coord.join(threads)

    test = l.run(y_, {x: feature})[0]

    f_file = l.run(tf.image.encode_png(test))
    W = open(output_dir + './test.png', 'wb+')
    W.write(f_file)
    W.close()
