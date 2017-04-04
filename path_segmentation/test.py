'''path_segmentation'''

#%%

# pylint: disable=E0401
# pylint: disable=C0103

# Import. Here, we import tensorflow, which gives us access to the library.
import os
import tensorflow as tf

# Here, we define helper functions for writing data to an example to a
# TFRecord file.

# Here, we define a function that reads a TFRecord file; parsing a single
# example.
def read_record(filename_queue):
    '''Read record'''

    # Here, the record contains examples derived from PNG images.

    reader = tf.TFRecordReader()
    _, serialised_example = reader.read(filename_queue)

    example = tf.parse_single_example(
        serialised_example,
        features={
            'feature_height': tf.FixedLenFeature([], tf.int64),
            'feature_width': tf.FixedLenFeature([], tf.int64),
            'feature_depth': tf.FixedLenFeature([], tf.int64),
            'label_height': tf.FixedLenFeature([], tf.int64),
            'label_width': tf.FixedLenFeature([], tf.int64),
            'label_depth': tf.FixedLenFeature([], tf.int64),
            'feature_raw': tf.FixedLenFeature([], tf.string),
            'label_raw': tf.FixedLenFeature([], tf.string),
        }
    )

    return example

# Here, we extract data from a single example.
def extract_example_data(example):
    '''Extract example's data'''

    # feature = sequence_example['feature_list']

    feature_height = tf.cast(example['feature_height'], tf.int32)
    feature_width = tf.cast(example['feature_width'], tf.int32)
    feature_depth = tf.cast(example['feature_depth'], tf.int32)
    label_height = tf.cast(example['label_height'], tf.int32)
    label_width = tf.cast(example['label_width'], tf.int32)
    label_depth = tf.cast(example['label_depth'], tf.int32)
    feature_raw = tf.decode_raw(example['feature_raw'], tf.uint8)
    label_raw = tf.decode_raw(example['label_raw'], tf.uint8)

    # feature = tf.reshape(feature_raw, tf.stack([feature_height, feature_width, feature_depth]))
    # label = tf.reshape(label_raw, tf.stack([label_height, label_width, label_depth]))

    feature = tf.reshape(feature_raw, tf.stack([feature_height, feature_width, feature_depth]))
    label = tf.reshape(label_raw, tf.stack([label_height, label_width, label_depth]))

    return feature, label

# Here, we define a function that reads a TFRecord file.
def input_pipeline(filenames, num_epochs=1, batch_size=1):
    """Read a TFRecord"""

    filename_queue = tf.train.string_input_producer(
        filenames,
        num_epochs=num_epochs,
        shuffle=False
    )

    example = read_record(filename_queue)

    feature, label = extract_example_data(example)

    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    feature_batch, label_batch = tf.train.batch(
        [feature, label],
        batch_size=batch_size,
        capacity=capacity,
        dynamic_pad=True)

    return feature_batch, label_batch

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
record = input_pipeline([record_file], num_epochs, batch_size)

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
            W = open(output_dir + './test/test_' + str(i) + '.png', 'wb+')
            W.write(f_file)
            W.close()

            i += 1

    except tf.errors.OutOfRangeError:
        print('EoF')
    finally:
        coord.request_stop()

    coord.join(threads)
