'''
Reading and Writing Data
'''
#%%
# pylint: disable=C0103
# pylint: disable=E0401

# Import 'tensorflow' to get access to the TensorFlow library.
import tensorflow as tf

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
        'x': float_feature(feature[0]),
        'y': float_feature(feature[1]),
    }))

    return example

# Pipeline for reading in a file.
def input_pipeline(filenames, num_epochs=1):
    """Input pipeline"""

    filename_queue = tf.train.string_input_producer(
        filenames,
        num_epochs=num_epochs,
        shuffle=False
    )

    reader = tf.TextLineReader()
    _, value = reader.read(filename_queue)

    x, y = tf.decode_csv(value, record_defaults=[[0.0], [0.0]])

    return [x, y]

# Directory and name of the csv and record file.
input_dir = './reading_and_writing_data/data/input/'
output_dir = './reading_and_writing_data/data/output/'

csv_file = input_dir + 'csv_data.csv'
record_file = output_dir + 'csv_record.tfrecords'

# Creates a graph to read in a file.
csv_data = input_pipeline([csv_file], 1)

init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

with tf.Session() as s:

    # Initialise global and local variables.
    s.run(init)

    writer = tf.python_io.TFRecordWriter(record_file)

    # Create a thread, which will read in the record file and extract examples.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        while not coord.should_stop():

            # Sample data.
            feature = s.run(csv_data)

            # Create an example.
            example = make_example(feature)

            # Print record to screen.
            print(example)

            # Write example to record file.
            writer.write(example.SerializeToString())

    except tf.errors.OutOfRangeError:
        print('EoF')
    finally:
        coord.request_stop()

    coord.join(threads)

    writer.close()
