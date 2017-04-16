
# pylint: disable=E0401
# pylint: disable=C0103

import os
import numpy
import tensorflow as tf

def project_setup():
    '''Sets up the project's directories'''

    if not os.path.exists('./data/output'):
        os.makedirs('./data/output')

    if not os.path.exists('./model'):
        os.makedirs('./model')

    if not os.path.exists('./logs'):
        os.makedirs('./logs')

    return

# Here, we define helper functions for writing data to an example to a TFRecord file.
def float_feature(value):
    '''Create float_list-based feature'''
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def int_feature(value):
    '''Create float_list-based feature'''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
    '''Create bytes_list-based feature'''
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Here, we define a function that makes an example, which is written to a TFRecord file.
def make_example(feature, label):
    '''Make example from feature'''

    example = tf.train.Example(features=tf.train.Features(feature={
        'feature': float_feature(feature),
        'label': float_feature(label),
    }))

    return example

# Here, we define a function that reads a TFRecord file; parsing a single example.
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

def extract_example_data(example):
    '''Extract record'''

    feature = tf.cast(example['feature'], tf.float32)
    label = tf.cast(example['label'], tf.float32)

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

    x, y = tf.decode_csv(value, record_defaults=[[0.0], [0.0]])

    return [x, y]


