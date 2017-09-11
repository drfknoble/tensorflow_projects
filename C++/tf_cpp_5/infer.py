'''robot object detection inference'''

# pylint: disable=E0401
# pylint: disable=C0103
# pylint: disable=W0621
# pylint: disable=E1129
# pylint: disable=E1101

import os
import numpy as np

import tensorflow as tf
import utilities as utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

validation_record = utils.input_pipeline(
    ['./data/input/validation_record.tfrecords'], 1, 1)

logs_dir = './logs/'

init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

with tf.Session() as sess:

    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    summary_writer = tf.summary.FileWriter(logs_dir, sess.graph)

    ckpt = tf.train.latest_checkpoint(logs_dir)
    if ckpt is not None:
        meta_graph = ckpt + '.meta'
        saver = tf.train.import_meta_graph(meta_graph)
        saver.restore(sess, ckpt)
        print('Model restored...')
    else:
        print('Could not restore model')
        exit(0)

    try:
        i = 0

        image = tf.get_collection('image')[0]
        annotation_pred = tf.get_collection('prediction')[0]

        while not coord.should_stop():

            validation_image, validation_annotation = sess.run(
                validation_record)

            p = sess.run(annotation_pred, feed_dict={image:  validation_image})

            if not os.path.isdir(logs_dir + 'predicted'):
                os.mkdir(logs_dir + 'predicted')

            f_file = sess.run(tf.image.encode_png(validation_image[0]))
            W = open(
                logs_dir + 'predicted/' + str(i) + '_validation_image.png', 'wb+')
            W.write(f_file)
            W.close()

            f_file = sess.run(tf.image.encode_png(
                validation_annotation[0]))
            W = open(
                logs_dir + 'predicted/' + str(i) + '_validation_annotation.png', 'wb+')
            W.write(f_file)
            W.close()

            f_file = sess.run(tf.image.encode_png(255 * p[0]))
            W = open(
                logs_dir + 'predicted/' + str(i) + '_predicted.png', 'wb+')
            W.write(f_file)
            W.close()

            i += 1

    except tf.errors.OutOfRangeError:
        print('EoF')
    finally:
        coord.request_stop()

    coord.join(threads)
