'''robot object detection export graph'''

# pylint: disable=E0401
# pylint: disable=C0103
# pylint: disable=W0621
# pylint: disable=E1129
# pylint: disable=E1101

import os

import tensorflow as tf
import utilities as utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

validation_record = utils.input_pipeline(
    ['./data/input/validation_record.tfrecords'], 1, 1)

logs_dir = "./logs/"

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

    export_dir = "./trained_model"

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
   
    try:
        i = 0

        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.TRAINING])

        builder.save()

        print('Model exported...')

    except tf.errors.OutOfRangeError:
        print('EoF')
    finally:
        coord.request_stop()

    coord.join(threads)
