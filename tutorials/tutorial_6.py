'''Tutorial_5'''

# pylint: disable=C0413
# pylint: disable=C0103
# pylint: disable=E0401

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppresses warnings


def read_CSV(filename_queue):
    '''read_CSV'''

    reader = tf.TextLineReader()
    _, record_string = reader.read(filename_queue)

    record_defaults = [[0], [0], [0]]
    col1, col2, col3 = tf.decode_csv(record_string, record_defaults)

    return tf.stack([col1, col2, col3])


def input_pipeline(filenames, batch_size, num_epochs=None):
    '''input_pipeline'''

    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=True)
    example = read_CSV(filename_queue)

    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch = tf.train.batch(
        [example], batch_size=batch_size, capacity=capacity)

    return example_batch


if not os.path.exists('./data'):
    os.makedirs('./data')

data_file = ['./data/data.csv']

features = input_pipeline(data_file, 1, 1)

init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

with tf.Session() as s:

    s.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        i = 0
        while not coord.should_stop():

            f = s.run([features])

            print(i, f)

            i += 1

    except tf.errors.OutOfRangeError:
        print('EoF')
    finally:
        coord.request_stop()

    coord.join(threads)
