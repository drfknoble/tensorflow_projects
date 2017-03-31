'''
img_to_record.py
'''
#%%
#pylint: disable=C0103
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

print('img_to_record.py\n')

# functions

def float_feature(value):
    '''Create float_list-based feature'''
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int_feature(value):
    '''Create float_list-based feature'''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    '''Create bytes_list-based feature'''
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def make_example(feature, label):
    '''Make example from decoded jpeg'''

    image_raw = feature.tostring()
    label_raw = label.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'feature_height': int_feature(feature.shape[0]),
        'feature_width': int_feature(feature.shape[1]),
        'feature_depth': int_feature(feature.shape[2]),
        'label_height': int_feature(label.shape[0]),
        'label_width': int_feature(label.shape[1]),
        'label_depth': int_feature(label.shape[2]),
        'feature_raw': bytes_feature(image_raw),
        'label_raw': bytes_feature(label_raw),
    }))

    return example


def input_pipeline(filenames, num_epochs=1):
    """Input pipeline"""

    filename_queue = tf.train.string_input_producer(
        filenames,
        num_epochs=num_epochs,
        shuffle=False
    )

    reader = tf.WholeFileReader()
    _, image = reader.read(filename_queue)

    d_image = tf.image.decode_png(image)
    # d_image = tf.image.decode_jpeg(image)

    return d_image


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

feature_dir = './data/input/' #'./reading_and_writing_data/data/input/'
feature_file = feature_dir + 'features.txt'

label_dir = './data/input/' #'./reading_and_writing_data/data/input/'
label_file = label_dir + 'label.txt'

populate_file(feature_file, feature_dir)
populate_file(label_file, label_dir)

feature_list = parse_file(feature_file)
label_list = parse_file(label_file)

feature = input_pipeline(feature_list)
label = input_pipeline(label_list)

init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

with tf.Session() as s:

    s.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    record = tf.python_io.TFRecordWriter('./data/output/png_record.tfrecords') #'./reading_and_writing_data/data/output/record.tfrecords')

    try:
        i = 0
        while not coord.should_stop():

            f = s.run(feature)
            l = s.run(label)

            print(np.shape(f))

            example = make_example(f, l)

            print(example)

            record.write(example.SerializeToString())

            i += 1

    except tf.errors.OutOfRangeError:
        print('EoF\n')
    finally:
        coord.request_stop()

    coord.join(threads)

    record.close()
