'''TensorFlow Utility Functions'''

#%%

# pylint: disable=E0401
# pylint: disable=C0103
# pylint: disable=W0621

import os
import tensorflow as tf


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


def float_feature(value):
    '''Create float_list-based feature'''
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int_feature(value):
    '''Create int64_list-based feature'''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    '''Create bytes_list-based feature'''
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def make_example(feature, label):
    '''Make example from feature and label'''

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


def read_record(filename_queue):
    '''Read record'''

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


def extract_example_data(example):
    '''Extract example's data'''

    feature_height = tf.cast(example['feature_height'], tf.int32)
    feature_width = tf.cast(example['feature_width'], tf.int32)
    feature_depth = tf.cast(example['feature_depth'], tf.int32)
    label_height = tf.cast(example['label_height'], tf.int32)
    label_width = tf.cast(example['label_width'], tf.int32)
    label_depth = tf.cast(example['label_depth'], tf.int32)
    feature_raw = tf.decode_raw(example['feature_raw'], tf.uint8)
    label_raw = tf.decode_raw(example['label_raw'], tf.uint8)

    feature = tf.reshape(feature_raw, tf.stack(
        [feature_height, feature_width, feature_depth]))
    label = tf.reshape(label_raw, tf.stack(
        [label_height, label_width, label_depth]))

    return feature, label


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
