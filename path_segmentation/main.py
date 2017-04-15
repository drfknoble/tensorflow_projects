'''path_segmentation'''

#%%

# pylint: disable=E0401
# pylint: disable=C0103
# pylint: disable=W0621

import os
import numpy as np
import tensorflow as tf
import utilities as utils

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "data/output/", "path to output directory")
tf.flags.DEFINE_string("data_dir", "data/input/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-6",
                      "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "model/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(10)
NUM_OF_CLASSESS = 151
IMAGE_SIZE = 224

def VGG19_network(image, weights, keep_probability=0.85):
    '''VGG19 Network implemented in TensorFlow.'''

    with tf.variable_scope("layer1_1"):
        kernels, bias = weights[0][0][0][0][0] # Get initial kernel and bias values.
        conv1_1 = utils.conv_layer(image, [3, 3, 3, 64], [64],
                                   stride=1, kernel_init=kernels, bias_init=bias) # OP_0.
        relu1_1 = tf.nn.relu(conv1_1) # OP_1.
    with tf.variable_scope("layer1_2"):
        kernels, bias = weights[2][0][0][0][0]
        conv1_2 = utils.conv_layer(relu1_1, [3, 3, 64, 64], [64],
                                   stride=1, kernel_init=kernels, bias_init=bias) # OP_2.
        relu1_2 = tf.nn.relu(conv1_2) # OP_3.

    with tf.variable_scope("pool_1"):
        pool_1 = utils.avg_pool_2x2(relu1_2) # OP_4.

    with tf.variable_scope("layer2_1"):
        kernels, bias = weights[5][0][0][0][0]
        conv2_1 = utils.conv_layer(pool_1, [3, 3, 64, 128], [128],
                                   stride=1, kernel_init=kernels, bias_init=bias) # OP_5.
        relu2_1 = tf.nn.relu(conv2_1) # OP_6.
    with tf.variable_scope("layer2_2"):
        kernels, bias = weights[7][0][0][0][0]
        conv2_2 = utils.conv_layer(relu2_1, [3, 3, 128, 128], [128],
                                   stride=1, kernel_init=kernels, bias_init=bias) # OP_7.
        relu2_2 = tf.nn.relu(conv2_2) # OP_8.

    with tf.variable_scope("pool_2"):
        pool_2 = utils.avg_pool_2x2(relu2_2)

    with tf.variable_scope("layer3_1"):
        kernels, bias = weights[10][0][0][0][0]
        conv3_1 = utils.conv_layer(pool_2, [3, 3, 128, 256], [256],
                                   stride=1, kernel_init=kernels, bias_init=bias)
        relu3_1 = tf.nn.relu(conv3_1)
    with tf.variable_scope("layer3_2"):
        kernels, bias = weights[12][0][0][0][0]
        conv3_2 = utils.conv_layer(relu3_1, [3, 3, 256, 256], [256],
                                   stride=1, kernel_init=kernels, bias_init=bias)
        relu3_2 = tf.nn.relu(conv3_2)
    with tf.variable_scope("layer3_3"):
        kernels, bias = weights[14][0][0][0][0]
        conv3_3 = utils.conv_layer(relu3_2, [3, 3, 256, 256], [256],
                                   stride=1, kernel_init=kernels, bias_init=bias)
        relu3_3 = tf.nn.relu(conv3_3)
    with tf.variable_scope("layer3_4"):
        kernels, bias = weights[16][0][0][0][0]
        conv3_4 = utils.conv_layer(relu3_3, [3, 3, 256, 256], [256],
                                   stride=1, kernel_init=kernels, bias_init=bias)
        relu3_4 = tf.nn.relu(conv3_4)

    with tf.variable_scope("pool_3"):
        pool_3 = utils.avg_pool_2x2(relu3_4)

    with tf.variable_scope("layer4_1"):
        kernels, bias = weights[19][0][0][0][0]
        conv4_1 = utils.conv_layer(pool_3, [3, 3, 256, 512], [512],
                                   stride=1, kernel_init=kernels, bias_init=bias)
        relu4_1 = tf.nn.relu(conv4_1)
    with tf.variable_scope("layer4_2"):
        kernels, bias = weights[21][0][0][0][0]
        conv4_2 = utils.conv_layer(relu4_1, [3, 3, 512, 512], [512],
                                   stride=1, kernel_init=kernels, bias_init=bias)
        relu4_2 = tf.nn.relu(conv4_2)
    with tf.variable_scope("layer4_3"):
        kernels, bias = weights[23][0][0][0][0]
        conv4_3 = utils.conv_layer(relu4_2, [3, 3, 512, 512], [512],
                                   stride=1, kernel_init=kernels, bias_init=bias)
        relu4_3 = tf.nn.relu(conv4_3)
    with tf.variable_scope("layer4_4"):
        kernels, bias = weights[25][0][0][0][0]
        conv4_4 = utils.conv_layer(relu4_3, [3, 3, 512, 512], [512],
                                   stride=1, kernel_init=kernels, bias_init=bias)
        relu4_4 = tf.nn.relu(conv4_4)

    with tf.variable_scope("pool_4"):
        pool_4 = utils.avg_pool_2x2(relu4_4)

    with tf.variable_scope("layer5_1"):
        kernels, bias = weights[28][0][0][0][0]
        conv5_1 = utils.conv_layer(pool_4, [3, 3, 512, 512], [512],
                                   stride=1, kernel_init=kernels, bias_init=bias)
        relu5_1 = tf.nn.relu(conv5_1)
    with tf.variable_scope("layer5_2"):
        kernels, bias = weights[30][0][0][0][0]
        conv5_2 = utils.conv_layer(relu5_1, [3, 3, 512, 512], [512],
                                   stride=1, kernel_init=kernels, bias_init=bias)
        relu5_2 = tf.nn.relu(conv5_2)
    with tf.variable_scope("layer5_3"):
        kernels, bias = weights[32][0][0][0][0]
        conv5_3 = utils.conv_layer(relu5_2, [3, 3, 512, 512], [512],
                                   stride=1, kernel_init=kernels, bias_init=bias)
        relu5_3 = tf.nn.relu(conv5_3)
    with tf.variable_scope("layer5_4"):
        kernels, bias = weights[34][0][0][0][0]
        conv5_4 = utils.conv_layer(relu5_3, [3, 3, 512, 512], [512],
                                   stride=1, kernel_init=kernels, bias_init=bias)
        # relu5_4 = tf.nn.relu(conv5_4)

    with tf.variable_scope('pool_5'):
        pool_5 = utils.max_pool_2x2(conv5_4)

    with tf.variable_scope('fc_1'):
        fc_1 = utils.conv_layer(pool_5, [7, 7, 512, 4096], [4096])
        fc_1 = tf.nn.relu(fc_1)
        if FLAGS.debug:
            utils.add_activation_summary(fc_1)
        fc_1_dropout = tf.nn.dropout(fc_1, keep_prob=keep_probability)

    with tf.variable_scope('fc_2'):
        fc_2 = utils.conv_layer(fc_1_dropout, [1, 1, 4096, 4096], [4096])
        fc_2 = tf.nn.relu(fc_2)
        if FLAGS.debug:
            utils.add_activation_summary(fc_2)
        fc_2_dropout = tf.nn.dropout(fc_2, keep_prob=keep_probability)

    with tf.variable_scope('fc_3'):
        fc_3 = utils.conv_layer(fc_2_dropout, [1, 1, 4096, NUM_OF_CLASSESS], [NUM_OF_CLASSESS])

    deconv_shape1 = [1, 14, 14, 512]
    W_t1 = utils.weight_variable(
        [4, 4, deconv_shape1[3], NUM_OF_CLASSESS], name="W_t1")
    b_t1 = utils.bias_variable([deconv_shape1[3]], name="b_t1")
    conv_t1 = utils.conv2d_transpose_strided(
        fc_3, W_t1, b_t1, output_shape=tf.shape(pool_4))
    fuse_1 = tf.add(conv_t1, pool_4, name="fuse_1")

    deconv_shape2 = [1, 28, 28, 256]
    W_t2 = utils.weight_variable(
        [4, 4, deconv_shape2[3], deconv_shape1[3]], name="W_t2")
    b_t2 = utils.bias_variable([deconv_shape2[3]], name="b_t2")
    conv_t2 = utils.conv2d_transpose_strided(
        fuse_1, W_t2, b_t2, output_shape=tf.shape(pool_3))
    fuse_2 = tf.add(conv_t2, pool_3, name="fuse_2")

    shape = tf.shape(image)
    deconv_shape3 = tf.stack(
        [shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
    W_t3 = utils.weight_variable(
        [16, 16, NUM_OF_CLASSESS, deconv_shape2[3]], name="W_t3")
    b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
    conv_t3 = utils.conv2d_transpose_strided(
        fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

    return conv_t3


def inference(image, keep_prob):
    '''Reads VGG data and passes input to network.'''

    print("Getting VGG data...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])
    print('Done')

    image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):

        output = VGG19_network(image, weights, keep_probability=keep_prob)

        annotation_pred = tf.argmax(output, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), output


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
    tf.summary.image("pred_annotation", tf.cast(
        pred_annotation, tf.uint8), max_outputs=2)

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
                    print("Step: %d, Train_loss: %g" % (itr, train_loss))
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

                    print("Validation_loss: %g" % (valid_loss))
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

                if not os.path.isdir(FLAGS.logs_dir + './predicted'):
                    os.mkdir(FLAGS.logs_dir + './predicted')

                f_file = sess.run(tf.image.encode_png(validation_image[0]))
                W = open(
                    FLAGS.logs_dir + './predicted/' + str(i) + '_validation_image.png', 'wb+')
                W.write(f_file)
                W.close()

                f_file = sess.run(tf.image.encode_png(
                    validation_annotation[0]))
                W = open(
                    FLAGS.logs_dir + './predicted/' + str(i) + '_validation_annotation.png', 'wb+')
                W.write(f_file)
                W.close()

                f_file = sess.run(tf.image.encode_png(pred[0]))
                W = open(FLAGS.logs_dir + './predicted/' +
                         str(i) + '_predicted.png', 'wb+')
                W.write(f_file)
                W.close()

                i += 1

        except tf.errors.OutOfRangeError:
            print('EoF')
        finally:
            coord.request_stop()

        coord.join(threads)
