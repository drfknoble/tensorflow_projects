'''path_segmentation'''

#%%

# pylint: disable=E0401
# pylint: disable=C0103

# Import. Here, we import tensorflow, which gives us access to the library.
import tensorflow as tf
import utilities as utils

# Here, we define important directories.
input_dir = './data/input/'
output_dir = './data/output/'

# Here, we define important file names.
record_file = output_dir + 'validation_record.tfrecords'

# Here, we define the number of times we read a record file, and what size
# each batch is.
num_epochs = 1
batch_size = 1

# Here, we create handles for reading and writing TFRecord files.
record = utils.input_pipeline([record_file], num_epochs, batch_size)

# Here, we define our graph.
with tf.name_scope('input'):

    x = tf.placeholder(tf.float32, shape=[None, None, None, None], name='x')
    y = tf.placeholder(tf.float32, shape=[None, None, None, None], name='y')

with tf.name_scope('network'):

    with tf.variable_scope('conv1_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 1, 3, 3], 0, 0.2), name='kernel')
        y_ = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], 'SAME', name='conv')

# Initialisation commands
init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

# Here, we restore our latest checkpoint and test our graph.
with tf.Session() as f:

    f.run(init)

    loader = tf.train.import_meta_graph('./model/main.meta')
    ckpt = tf.train.latest_checkpoint('./model/')
    loader.restore(f, ckpt)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        i = 0

        while not coord.should_stop():

            feature, label = f.run(record)

            test = f.run(y_, {x: feature})[0]

            f_file = f.run(tf.image.encode_png(test))
            W = open(output_dir + './validation_' + str(i) + '.png', 'wb+')
            W.write(f_file)
            W.close()

            i += 1

    except tf.errors.OutOfRangeError:
        print('EoF')
    finally:
        coord.request_stop()

    coord.join(threads)
