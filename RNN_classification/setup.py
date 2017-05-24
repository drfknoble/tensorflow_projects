'''RNN_classification'''

#pylint:disable=C0103

import tensorflow as tf
import utilities as utils

# get list of files in data folder; open list; parse and
# create records based on file lists' contents.

# parse ./data/input/features for training feature files.
utils.generate_file_list(
    './data/input/features/training',
    '.csv',
    './data/input/training_features.txt')

# parse ./data/input/labels for training label files.
utils.generate_file_list(
    './data/input/labels/training',
    '.csv',
    './data/input/training_labels.txt')

# generate training record
temp = utils.parse_file_list(['./data/input/training_features.txt'], 1)

init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

with tf.Session() as sess:

    sess.run(init)

        # Create a thread, which will read in the record file and extract examples.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        while not coord.should_stop():

            f = sess.run(temp)
            print(f)

    except tf.errors.OutOfRangeError:
        print('EoF')
    finally:
        coord.request_stop()

    coord.join(threads)

    sess.close()
