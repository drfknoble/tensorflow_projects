'''CNN'''

#%%

# pylint: disable=E0401
# pylint: disable=C0103

# Import. Here, we import tensorflow, which gives us access to the library.
import os
import tensorflow as tf

# Here, we define helper functions for writing data to an example to a
# TFRecord file.


def float_feature(value):
    '''Create float_list-based feature'''
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int_feature(value):
    '''Create int64_list-based feature'''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    '''Create bytes_list-based feature'''
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Here, we define a function that makes an example, which is written to a
# TFRecord file.
def make_example(feature, label):
    '''Make example from feature and label'''

    # Here, the feature and label are decoded PNG images.

    feature_raw = feature.tostring()
    label_raw = label.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'feature_height': int_feature(feature.shape[0]),
        'feature_width': int_feature(feature.shape[1]),
        'feature_depth': int_feature(feature.shape[2]),
        'label_height': int_feature(label.shape[0]),
        'label_width': int_feature(label.shape[1]),
        'label_depth': int_feature(label.shape[2]),
        'feature_raw': bytes_feature(feature_raw),
        'label_raw': bytes_feature(label_raw),
    }))

    return example

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

def populate_file(file, file_dir):
    ''' populate file with file names ending in '.png' '''

    F = open(file, 'w')

    lst = os.listdir(file_dir)
    lst = lst.sort()

    for file in sorted(os.listdir(file_dir)):
        if file.endswith('.png'):
            F.write(file_dir + file + '\n')

    F.close()

    return

def parse_file(file):
    ''' parse file for list of filenames'''

    F = open(file, 'r')

    file_list = F.read().splitlines()

    F.close()

    return file_list

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

# Here, we define a function that writes a TFRecord file.


def output_pipeline(filenames, num_epochs=1):
    """Write a TFRecord"""

    filename_queue = tf.train.string_input_producer(
        filenames,
        num_epochs=num_epochs,
        shuffle=False
    )

    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)

    image = tf.image.decode_png(value)

    return image

# Here, we define important directories.
input_dir = './data/input/'
output_dir = './data/output/'
features_dir = input_dir + 'features/'
labels_dir = input_dir + 'labels/'

# Here, we define important file names.
features_txt_file = input_dir + 'features.txt'
labels_txt_file = input_dir + 'labels.txt'
record_file = output_dir + 'img_record.tfrecords'

# Here, we populate the feature_file and label_file with name of feature and label images.
populate_file(features_txt_file, features_dir)
populate_file(labels_txt_file, labels_dir)

features_list = parse_file(features_txt_file)
labels_list = parse_file(labels_txt_file)

# Here, we define the number of times we read a record file, and what size
# each batch is.
num_epochs = 1000
batch_size = 1

# Here, we create handles for reading and writing TFRecord files.
features = output_pipeline(features_list, 1)
labels = output_pipeline(labels_list, 1)
record = input_pipeline([record_file], num_epochs, batch_size)

# Here, we define network parameters

# Here, we define our graph.
with tf.name_scope('input'):

    x = tf.placeholder(tf.float32, shape=[None, None, None, None], name='x')
    y = tf.placeholder(tf.float32, shape=[None, None, None, None], name='y')

with tf.name_scope('network'):

    with tf.variable_scope('conv1_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 1, 3, 3], 0, 0.2), name='kernel')
        # y_ = tf.nn.depthwise_conv2d(x, kernel, [1, 1, 1, 1], 'SAME', name='depthwise_conv')
        y_ = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], 'SAME', name='depthwise_conv')


cost = tf.losses.mean_squared_error(y, y_)
optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)
tf.summary.scalar('Cost', cost)

# Initialisation commands
init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

# In this session, we read our raw data and create a TFRecord file.
with tf.Session() as s:

    s.run(init)

    writer = tf.python_io.TFRecordWriter(record_file)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        while not coord.should_stop():

            single_feature = s.run(features)
            single_label = s.run(labels)

            example = make_example(single_feature, single_label)

            print(example)

            writer.write(example.SerializeToString())

    except tf.errors.OutOfRangeError:
        print('EoF')
    finally:
        coord.request_stop()

    coord.join(threads)

    writer.close()

# In this session, we read the TFRecord file and use its examples with our
# graph.
with tf.Session() as l:

    l.run(init)

    summary_writer = tf.summary.FileWriter('./logs', l.graph)

    merged = tf.summary.merge_all()

    saver = tf.train.Saver()

    try:
        loader = tf.train.import_meta_graph('./model/main.meta')
        ckpt = tf.train.latest_checkpoint('./model/')
        loader.restore(l, ckpt)
    except Exception as e:
        saver.save(l, './model/main.ckpt', 0)
        saver.export_meta_graph('./model/main.meta')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        i = 0

        while not coord.should_stop():

            feature, label = l.run(record)

            feed_dict = {x: feature, y: label}
            summary, c, _ = l.run([merged, cost, optimizer], feed_dict)

            summary_writer.add_summary(summary, i)

            if i % (num_epochs / 10) == 0:
                print(c)

            if i % 50 == 0:
                saver.save(l, './model/main.ckpt', i)

            i += 1

    except tf.errors.OutOfRangeError:
        print('EoF')
    finally:
        coord.request_stop()

    coord.join(threads)

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

            feed_dict = {x: feature}

            test = f.run(y_, feed_dict)[0]

            f_file = f.run(tf.image.encode_png(test))
            W = open(output_dir + 'test_.png', 'wb+')
            W.write(f_file)
            W.close()

            i += 1

            print(f.run(kernel))

            break

    except tf.errors.OutOfRangeError:
        print('EoF')
    finally:
        coord.request_stop()

    coord.join(threads)
