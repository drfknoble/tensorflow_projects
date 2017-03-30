'''
Reading and Writing Data
'''
#%%
# pylint: disable=C0103
# pylint: disable=E0401

# Import 'tensorflow' to get access to the TensorFlow library.
import tensorflow as tf

def read_record(filename_queue):

    reader = tf.TFRecordReader()
    _, serialised_example = reader.read(filename_queue)

    example = tf.parse_single_example(
        serialised_example,
        features={
            'x': tf.FixedLenFeature([], tf.float32),
            'y': tf.FixedLenFeature([], tf.float32),
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

# Directory and name of the csv and record file.
output_dir = './reading_and_writing_data/data/output/'

record_file = output_dir + 'csv_record.tfrecords'

# Creates a graph to read in a file.
record = input_pipeline([record_file], 1)

init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

with tf.Session() as s:

    # Initialise global and local variables.
    s.run(init)

    # Create a thread, which will read in the record file and extract examples.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        while not coord.should_stop():

            r = s.run(record)

            x = s.run(tf.cast(r['x'], tf.int64))
            y = s.run(tf.cast(r['y'], tf.int64))

            print(x, y)

    except tf.errors.OutOfRangeError:
        print('EoF')
    finally:
        coord.request_stop()

    coord.join(threads)
