'''path_segmentation'''

#%%

# pylint: disable=E0401
# pylint: disable=C0103
# pylint: disable=W0621

import tensorflow as tf
import utilities as utils

default_dir = 'E:\\users\\fknoble\\My_FCN\\'

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", default_dir +
                       "logs\\", "path to logs directory")
tf.flags.DEFINE_string("data_dir", default_dir + "data\\", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-6",
                      "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", default_dir +
                       "model\\", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(1e4 + 1)
NUM_OF_CLASSESS = 151
IMAGE_SIZE = 224

keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
image = tf.placeholder(
    tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
annotation = tf.placeholder(
    tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")

training_record = utils.input_pipeline(
    [FLAGS.data_dir + 'training_record.tfrecords'], 1)

x_ = image
y_ = annotation

init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

with tf.Session() as sess:

    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        i = 0

        while not coord.should_stop():

            feature, label = sess.run(training_record)

            feature = sess.run(tf.image.resize_images(feature, [IMAGE_SIZE, IMAGE_SIZE]))
            label = sess.run(tf.image.resize_images(label, [IMAGE_SIZE, IMAGE_SIZE]))

            feed_dict = {image: feature, annotation: label}

            image_out, annotation_out = sess.run([x_, y_], feed_dict=feed_dict)

            f_file = sess.run(tf.image.encode_png(image_out[0]))
            W = open(FLAGS.logs_dir + './image_test_' +
                     str(i) + '.png', 'wb+')
            W.write(f_file)
            W.close()

            f_file = sess.run(tf.image.encode_png(annotation_out[0]))
            W = open(FLAGS.logs_dir + './annotation_test_' +
                     str(i) + '.png', 'wb+')
            W.write(f_file)
            W.close()

            i += 1

            break

    except tf.errors.OutOfRangeError:
        print('EoF')
    finally:
        coord.request_stop()

    coord.join(threads)
