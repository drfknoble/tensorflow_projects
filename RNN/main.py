'''RNN'''

#%%

# pylint: disable=E0401
# pylint: disable=C0103

# Import. Here, we import tensorflow, which gives us access to the library.
import os
import tensorflow as tf

if not os.path.exists('./data/output'):
    os.makedirs('./data/output')

if not os.path.exists('./model'):
    os.makedirs('./model')

if not os.path.exists('./logs'):
    os.makedirs('./logs')

# Here, we define helper functions for writing data to an example to a
# TFRecord file.


def float_feature(value):
    '''Create float_list-based feature'''
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int_feature(value):
    '''Create float_list-based feature'''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    '''Create bytes_list-based feature'''
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Here, we define a function that makes an example, which is written to a
# TFRecord file.


def make_example(feature, label=None):
    '''Make example from feature'''

    example = tf.train.Example(features=tf.train.Features(feature={
        'feature': float_feature(feature),
        'label': float_feature(label),
    }))

    return example


def make_sequence_example(sequence, sequence_label):
    '''Make a sequence example from sequence'''

    input_features = [
        tf.train.Feature(float_list=tf.train.FloatList(value=token))
        for token in sequence]
    label_feature = [
        tf.train.Feature(float_list=tf.train.FloatList(value=token_label))
        for token_label in sequence_label]
    feature_list = {
        'feature_list': tf.train.FeatureList(feature=input_features),
        'feature_list_labels': tf.train.FeatureList(feature=label_feature)
    }
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    sequence_example = tf.train.SequenceExample(feature_lists=feature_lists)

    return sequence_example

# Here, we define a function that reads a TFRecord file; parsing a single
# example.


def read_record(filename_queue):
    '''Read record'''

    reader = tf.TFRecordReader()
    _, serialised_example = reader.read(filename_queue)

    example = tf.parse_single_example(
        serialised_example,
        features={
            'feature': tf.FixedLenFeature([], tf.float32),
            'label': tf.FixedLenFeature([], tf.float32),
        }
    )

    return example


def read_sequence_record(filename_queue):
    """Read record"""

    reader = tf.TFRecordReader()
    _, record_string = reader.read(filename_queue)

    _, sequence_example = tf.parse_single_sequence_example(
        record_string,
        None,
        sequence_features={
            'feature_list': tf.FixedLenSequenceFeature([], tf.float32),
            'feature_list_labels': tf.FixedLenSequenceFeature([], tf.float32)
        })

    return sequence_example


def extract_example_data(example):
    '''Extract example's data'''

    feature = tf.cast(example['feature'], tf.float32)
    label = tf.cast(example['label'], tf.float32)

    return feature, label


def extract_sequence_example_data(sequence_example):
    '''Extract sequence example's data'''

    feature = sequence_example['feature_list']
    label = sequence_example['feature_list_labels']

    return feature, label

# Here, we define a function that reads a TFRecord file.


def input_pipeline(filenames, num_epochs=1, batch_size=1):
    """Read a TFRecord"""

    filename_queue = tf.train.string_input_producer(
        filenames,
        num_epochs=num_epochs,
        shuffle=False
    )

    sequence_example = read_sequence_record(filename_queue)

    feature, label = extract_sequence_example_data(sequence_example)

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

    reader = tf.TextLineReader()
    _, value = reader.read(filename_queue)

    col_1, col_2, col_3, col_4 = tf.decode_csv(value, record_defaults=[[0.0], [0.0], [0.0], [0.0]])

    feature = [[col_1], [col_2], [col_3]]
    label = [[col_4]]

    return feature, label

# Here, we define important directories.
input_dir = './data/input/'
output_dir = './data/output/'

# Here, we define important file names.
csv_file = input_dir + 'csv_sequence_data.csv'
record_file = output_dir + 'csv_sequence_record.tfrecords'

# Here, we define the number of times we read a record file, and what size
# each batch is.
num_epochs = 40
batch_size = 1

# Here, we create handles for reading and writing TFRecord files.
csv_sequence_data = output_pipeline([csv_file], 1)
record = input_pipeline([record_file], num_epochs, batch_size)

# Here, we define network parameters
num_hidden = 10

def MLP_layer(x, W, b):
    '''Default MLP layer'''

    W = tf.get_variable(name='W', shape=W,
                        dtype=tf.float32, initializer=tf.constant_initializer(1.0))
    b = tf.get_variable(name='b', shape=b,
                        dtype=tf.float32, initializer=tf.constant_initializer(0.1))

    y_ = tf.add(tf.matmul(x, W), b)

    return y_

def RNN_layer(cell, x, W, b):
    '''Default RNN layer'''

    W = tf.get_variable(name='W', shape=W,
                        dtype=tf.float32, initializer=tf.constant_initializer(1.0))
    b = tf.get_variable(name='b', shape=b,
                        dtype=tf.float32, initializer=tf.constant_initializer(0.1))

    output, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    val = tf.transpose(output, [1, 0, 2])
    last = tf.gather(val, int(val.get_shape()[0]) - 1)

    y_ = tf.add(tf.matmul(last, W), b)

    return y_

# Here, we define our graph.
with tf.name_scope('input'):

    x = tf.placeholder(tf.float32, shape=[None, 1, 3], name='x')
    y = tf.placeholder(tf.float32, shape=[None, 1], name='y')

with tf.name_scope('network'):

    cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)

    y_ = RNN_layer(cell, x, [num_hidden, 1], [1])

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

            feature, label = s.run(csv_sequence_data)

            sequence_example = make_sequence_example(feature, label)

            print(sequence_example)

            writer.write(sequence_example.SerializeToString())

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
    saver.save(l, './model/main.ckpt', 0)
    saver.export_meta_graph('./model/main.meta')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        i = 0

        while not coord.should_stop():

            X, Y = l.run(record)

            feed_dict = {x: [X], y: Y}
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

    input = [[0.0], [0.0], [0.0]]
    X = f.run(tf.reshape(input, [1, 1, 3]))
    ans = f.run(y_, {x: X})

    print(ans)
