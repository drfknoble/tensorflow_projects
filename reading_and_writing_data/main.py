'''
Reading and Writing Data
'''
#%%
# pylint: disable=C0103
# pylint: disable=E0401

import tensorflow as tf


def float_feature(value):
    '''Create float_list-based feature'''
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int_feature(value):
    '''Create float_list-based feature'''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    '''Create bytes_list-based feature'''
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def make_example(feature, label=None):
    '''Make example from decoded jpeg'''

    example = tf.train.Example(features=tf.train.Features(feature={
        'A': int_feature(feature[0]),
        'B': int_feature(feature[1]),
    }))

    return example

def read_record(filename_queue):
    '''Reads a TFRecord'''

    reader = tf.TFRecordReader()
    _, serialised_example = reader.read(filename_queue)

    example = tf.parse_single_example(
        serialised_example,
        features={
            'A': tf.FixedLenFeature([], tf.int64),
            'B': tf.FixedLenFeature([], tf.int64),
        }
    )

    return example

def input_pipeline(filenames, num_epochs=1):
    """Input pipeline"""

    filename_queue = tf.train.string_input_producer(
        filenames,
        num_epochs=num_epochs,
        shuffle=False
    )

    record = read_record(filename_queue)

    return record

data_dir = './reading_and_writing_data/data/output/'
record_file = data_dir + 'record.tfrecords'

record_in = input_pipeline([record_file], 1)

init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

with tf.Session() as s:

    feature = [1, 2]

    example = make_example(feature)

    print(example)

    writer = tf.python_io.TFRecordWriter(record_file)

    writer.write(example.SerializeToString())

with tf.Session() as l:

    l.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:

        while not coord.should_stop():

            r = l.run(record_in)

            A = l.run(tf.cast(r['A'], tf.int64))
            B = l.run(tf.cast(r['B'], tf.int64))

            print(A, B)

    except tf.errors.OutOfRangeError:
        print('EoF')
    finally:
        coord.request_stop()

    coord.join(threads)
