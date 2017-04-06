'''path_segmentation'''

#%%

# pylint: disable=E0401
# pylint: disable=C0103
# pylint: disable=W0621

import os
import numpy as np
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

MAX_ITERATION = int(20 + 1)
NUM_OF_CLASSESS = 151
IMAGE_SIZE = 224

def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels,
            # out_channels]
            kernels = utils.get_variable(np.transpose(
                kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net


def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    processed_image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]

        pool5 = utils.max_pool_2x2(conv_final_layer)

        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if FLAGS.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable(
            [4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(
            conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable(
            [4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(
            fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack(
            [shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable(
            [16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(
            fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv_t3


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


# Main

training_record = utils.input_pipeline(
    [FLAGS.data_dir + 'training_record.tfrecords'], MAX_ITERATION)
validation_record = utils.input_pipeline(
    [FLAGS.data_dir + 'validation_record.tfrecords'], 1)

with tf.name_scope('input'):
    keep_probability = tf.placeholder(
        tf.float32, shape=None, name="keep_probabilty")
    image = tf.placeholder(
        tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    annotation = tf.placeholder(
        tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(
        annotation, tf.uint8), max_outputs=2)

# with tf.name_scope('network'):

with tf.name_scope('output'):
    pred_annotation, logits = inference(image, keep_probability)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)

loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits,
    labels=tf.squeeze(annotation, squeeze_dims=[3]),
    name="entropy")))
tf.summary.scalar("entropy", loss)

trainable_var = tf.trainable_variables()
if FLAGS.debug:
    for var in trainable_var:
        utils.add_to_regularization_and_summary(var)
train_op = train(loss, trainable_var)

init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

with tf.Session() as sess:

    sess.run(init)

    summary_op = tf.summary.merge_all()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    saver = tf.train.Saver()

    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

    ckpt = tf.train.latest_checkpoint(FLAGS.logs_dir)
    if ckpt is not None:
        saver.restore(sess, ckpt)
        print('Model restored...')

    if FLAGS.mode == 'train':
        try:
            itr = 0

            while not coord.should_stop():

                training_image, training_annotation = sess.run(training_record)

                training_image = sess.run(tf.image.resize_images(
                    training_image, [IMAGE_SIZE, IMAGE_SIZE]))
                training_annotation = sess.run(tf.image.resize_images(
                    training_annotation, [IMAGE_SIZE, IMAGE_SIZE]))

                sess.run(train_op, feed_dict={image:  training_image,
                                              annotation:  training_annotation,
                                              keep_probability: 0.85})

                if itr % 10 == 0:
                    train_loss, summary_str = sess.run(
                        [loss, summary_op], feed_dict={image:  training_image,
                                                       annotation:  training_annotation,
                                                       keep_probability: 0.85})
                    print("Step: %d, Train_loss:%g" % (itr, train_loss))
                    summary_writer.add_summary(summary_str, itr)

                if itr % 400 == 0:
                    validation_image, validation_annotation = sess.run(
                        validation_record)

                    validation_image = sess.run(tf.image.resize_images(
                        validation_image, [IMAGE_SIZE, IMAGE_SIZE]))
                    validation_annotation = sess.run(tf.image.resize_images(
                        validation_annotation, [IMAGE_SIZE, IMAGE_SIZE]))

                    valid_loss = sess.run(loss, feed_dict={image: validation_image,
                                                           annotation: validation_annotation,
                                                           keep_probability: 1.0})

                    print("Validation_loss: %s" % (valid_loss))
                    saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

                itr += 1

        except tf.errors.OutOfRangeError:
            print('EoF')
        finally:
            coord.request_stop()

        coord.join(threads)

    elif FLAGS.mode == 'visualize':
        try:
            i = 0

            while not coord.should_stop():

                validation_image, validation_annotation = sess.run(
                    validation_record)

                validation_image = sess.run(tf.image.resize_images(
                    validation_image, [IMAGE_SIZE, IMAGE_SIZE]))
                validation_annotation = sess.run(tf.image.resize_images(
                    validation_annotation, [IMAGE_SIZE, IMAGE_SIZE]))

                pred = sess.run(pred_annotation, feed_dict={image:  validation_image,
                                                            annotation:  validation_annotation,
                                                            keep_probability: 1.0})

                f_file = sess.run(tf.image.encode_png(validation_image[0]))
                W = open(FLAGS.logs_dir + './validation_image' + str(i) + '.png', 'wb+')
                W.write(f_file)
                W.close()

                f_file = sess.run(tf.image.encode_png(validation_annotation[0]))
                W = open(FLAGS.logs_dir + './validation_annotation' + str(i) + '.png', 'wb+')
                W.write(f_file)
                W.close()

                f_file = sess.run(tf.image.encode_png(pred[0]))
                W = open(FLAGS.logs_dir + './pred' + str(i) + '.png', 'wb+')
                W.write(f_file)
                W.close()

                i += 1

                break

        except tf.errors.OutOfRangeError:
            print('EoF')
        finally:
            coord.request_stop()

        coord.join(threads)
