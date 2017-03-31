'''
record_to_png.py
'''
#%%
#pylint: disable=C0103
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

print('record_to_png.py\n')

def read_record(filename_queue):

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

def input_pipeline(filenames, num_epochs=1):
    """Input pipeline"""

    filename_queue = tf.train.string_input_producer(
        filenames,
        num_epochs=num_epochs,
        shuffle=False
    )

    record = read_record(filename_queue)

    return record

data_dir = './data/output/'#'./reading_and_writing_data/data/output/'
name_file = data_dir + 'png_record.tfrecords'

record = input_pipeline([name_file], 1)

init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

with tf.Session() as s:

    s.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        i = 0
        while not coord.should_stop():

            r = s.run(record)

            feature_height = s.run(tf.cast(r['feature_height'], tf.int64))
            feature_width = s.run(tf.cast(r['feature_width'], tf.int64))
            feature_depth = s.run(tf.cast(r['feature_depth'], tf.int64))
            label_height = s.run(tf.cast(r['label_height'], tf.int64))
            label_width = s.run(tf.cast(r['label_width'], tf.int64))
            label_depth = s.run(tf.cast(r['label_depth'], tf.int64))
            feature_raw = s.run(tf.decode_raw(r['feature_raw'], tf.uint8))
            label_raw = s.run(tf.decode_raw(r['label_raw'], tf.uint8))

            feature_raw = np.reshape(feature_raw, [feature_height, feature_width, feature_depth])

            f = s.run(tf.image.encode_png(feature_raw))
            W = open(data_dir+'feature_'+str(i)+'.png', 'wb+')
            W.write(f)
            W.close()

            label_raw = np.reshape(label_raw, [label_height, label_width, label_depth])

            l = s.run(tf.image.encode_png(label_raw))
            W = open(data_dir+'label_'+str(i)+'.png', 'wb+')
            W.write(l)
            W.close()

            i += 1

    except tf.errors.OutOfRangeError:
        print('EoF')
    finally:
        coord.request_stop()

    coord.join(threads)
