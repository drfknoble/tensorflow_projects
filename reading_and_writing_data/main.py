'''
Reading and Writing Data
'''
#%%
# pylint: disable=C0103
# pylint: disable=E0401

# Import 'tensorflow' to get access to the TensorFlow library.
import os
import tensorflow as tf

if not os.path.exists('./data/output'):
    os.makedirs('./data/output')

if not os.path.exists('./model'):
    os.makedirs('./model')

if not os.path.exists('./logs'):
    os.makedirs('./logs')
    
# Adds a float_list feature to an example.
def float_feature(value):
    '''Create float_list-based feature'''
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

# Adds a int64_list feature to an example.
def int_feature(value):
    '''Create float_list-based feature'''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Adds a bytes_list feature to an example.
def bytes_feature(value):
    '''Create bytes_list-based feature'''
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Makes an example using either a float, int, or bytes list as an element.
def make_example(feature, label=None):
    '''Make example from feature'''

    example = tf.train.Example(features=tf.train.Features(feature={
        'A': int_feature(feature[0]),
        'B': int_feature(feature[1]),
    }))

    return example

# Reads a TFRecord file and parses it for the elements we are after.
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

# Pipeline for reading in a file.
def input_pipeline(filenames, num_epochs=1):
    """Input pipeline"""

    filename_queue = tf.train.string_input_producer(
        filenames,
        num_epochs=num_epochs,
        shuffle=False
    )

    record = read_record(filename_queue)

    return record

# Directory and name of the record file.
data_dir = './data/output/'#'./reading_and_writing_data/data/output/'
record_file = data_dir + 'record.tfrecords'

# Creates a graph to read in a file.
record_in = input_pipeline([record_file], 1)

init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

with tf.Session() as s:

    # Initialises global and local variables.
    s.run(init)

    # Sample data.
    feature = [1, 2]

    # Create an example.
    example = make_example(feature)

    # Print record to screen.
    print(example)

    # Write example to record file.
    writer = tf.python_io.TFRecordWriter(record_file)
    writer.write(example.SerializeToString())
    writer.close()

with tf.Session() as l:

    # Initialise global and local variables.
    l.run(init)

    # Create a thread, which will read in the record file and extract examples.
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
