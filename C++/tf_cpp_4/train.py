'''robot object detection'''

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

num_classes = 2

batch_size = 2
num_epochs = 15

training_record = utils.input_pipeline(
    ['./data/input/training_record.tfrecords'], num_epochs, batch_size)
validation_record = utils.input_pipeline(
    ['./data/input/validation_record.tfrecords'], 1, 1)

model_path = './model/vgg16.npy'
data_dict = None

if os.path.isfile(model_path):
    data_dict = np.load(model_path, encoding='latin1').item()
    print('Model loaded.')
else:
    print('Could not load model.')
    exit(-1)

logs_dir = './logs/'
if not os.path.isdir(logs_dir):
    os.makedirs(logs_dir)

with tf.name_scope('input'):

    image = tf.placeholder(
        tf.float32, shape=[None, None, None, 3], name="image")
    annotation = tf.placeholder(
        tf.int32, shape=[None, None, None, 1], name="label")

    tf.add_to_collection('image', image)
    tf.add_to_collection('label', annotation)

with tf.name_scope('network'):
    with tf.variable_scope('conv1_1'):

        kernel_init = data_dict['conv1_1'][0]
        bias_init = data_dict['conv1_1'][1]

        W = tf.get_variable('W', kernel_init.shape, tf.float32,
                            tf.constant_initializer(kernel_init))
        b = tf.get_variable('b', bias_init.shape, tf.float32,
                            tf.constant_initializer(bias_init))

        conv = tf.nn.conv2d(image, W, [1, 1, 1, 1], 'SAME')
        bias_conv = tf.nn.bias_add(conv, b)
        relu = tf.nn.relu(bias_conv)

        conv1_1 = relu

    with tf.variable_scope('conv1_2'):

        kernel_init = data_dict['conv1_2'][0]
        bias_init = data_dict['conv1_2'][1]

        W = tf.get_variable('W', kernel_init.shape, tf.float32,
                            tf.constant_initializer(kernel_init))
        b = tf.get_variable('b', bias_init.shape, tf.float32,
                            tf.constant_initializer(bias_init))

        conv = tf.nn.conv2d(conv1_1, W, [1, 1, 1, 1], 'SAME')
        bias_conv = tf.nn.bias_add(conv, b)
        relu = tf.nn.relu(bias_conv)

        conv1_2 = relu

    with tf.variable_scope('pool_1'):

        pool_1 = tf.nn.avg_pool(conv1_2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    with tf.variable_scope('conv2_1'):

        kernel_init = data_dict['conv2_1'][0]
        bias_init = data_dict['conv2_1'][1]

        W = tf.get_variable('W', kernel_init.shape, tf.float32,
                            tf.constant_initializer(kernel_init))
        b = tf.get_variable('b', bias_init.shape, tf.float32,
                            tf.constant_initializer(bias_init))

        conv = tf.nn.conv2d(pool_1, W, [1, 1, 1, 1], 'SAME')
        bias_conv = tf.nn.bias_add(conv, b)
        relu = tf.nn.relu(bias_conv)

        conv2_1 = relu

    with tf.variable_scope('conv2_2'):

        kernel_init = data_dict['conv2_2'][0]
        bias_init = data_dict['conv2_2'][1]

        W = tf.get_variable('W', kernel_init.shape, tf.float32,
                            tf.constant_initializer(kernel_init))
        b = tf.get_variable('b', bias_init.shape, tf.float32,
                            tf.constant_initializer(bias_init))

        conv = tf.nn.conv2d(conv2_1, W, [1, 1, 1, 1], 'SAME')
        bias_conv = tf.nn.bias_add(conv, b)
        relu = tf.nn.relu(bias_conv)

        conv2_2 = relu

    with tf.variable_scope('pool_2'):

        pool_2 = tf.nn.avg_pool(conv2_2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    with tf.variable_scope('conv3_1'):

        kernel_init = data_dict['conv3_1'][0]
        bias_init = data_dict['conv3_1'][1]

        W = tf.get_variable('W', kernel_init.shape, tf.float32,
                            tf.constant_initializer(kernel_init))
        b = tf.get_variable('b', bias_init.shape, tf.float32,
                            tf.constant_initializer(bias_init))

        conv = tf.nn.conv2d(pool_2, W, [1, 1, 1, 1], 'SAME')
        bias_conv = tf.nn.bias_add(conv, b)
        relu = tf.nn.relu(bias_conv)

        conv3_1 = relu

    with tf.variable_scope('conv3_2'):

        kernel_init = data_dict['conv3_2'][0]
        bias_init = data_dict['conv3_2'][1]

        W = tf.get_variable('W', kernel_init.shape, tf.float32,
                            tf.constant_initializer(kernel_init))
        b = tf.get_variable('b', bias_init.shape, tf.float32,
                            tf.constant_initializer(bias_init))

        conv = tf.nn.conv2d(conv3_1, W, [1, 1, 1, 1], 'SAME')
        bias_conv = tf.nn.bias_add(conv, b)
        relu = tf.nn.relu(bias_conv)

        conv3_2 = relu

    with tf.variable_scope('conv3_3'):

        kernel_init = data_dict['conv3_3'][0]
        bias_init = data_dict['conv3_3'][1]

        W = tf.get_variable('W', kernel_init.shape, tf.float32,
                            tf.constant_initializer(kernel_init))
        b = tf.get_variable('b', bias_init.shape, tf.float32,
                            tf.constant_initializer(bias_init))

        conv = tf.nn.conv2d(conv3_2, W, [1, 1, 1, 1], 'SAME')
        bias_conv = tf.nn.bias_add(conv, b)
        relu = tf.nn.relu(bias_conv)

        conv3_3 = relu

    with tf.variable_scope('pool_3'):

        pool_3 = tf.nn.avg_pool(conv3_3, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    with tf.variable_scope('conv4_1'):

        kernel_init = data_dict['conv4_1'][0]
        bias_init = data_dict['conv4_1'][1]

        W = tf.get_variable('W', kernel_init.shape, tf.float32,
                            tf.constant_initializer(kernel_init))
        b = tf.get_variable('b', bias_init.shape, tf.float32,
                            tf.constant_initializer(bias_init))

        conv = tf.nn.conv2d(pool_3, W, [1, 1, 1, 1], 'SAME')
        bias_conv = tf.nn.bias_add(conv, b)
        relu = tf.nn.relu(bias_conv)

        conv4_1 = relu

    with tf.variable_scope('conv4_2'):

        kernel_init = data_dict['conv4_2'][0]
        bias_init = data_dict['conv4_2'][1]

        W = tf.get_variable('W', kernel_init.shape, tf.float32,
                            tf.constant_initializer(kernel_init))
        b = tf.get_variable('b', bias_init.shape, tf.float32,
                            tf.constant_initializer(bias_init))

        conv = tf.nn.conv2d(conv4_1, W, [1, 1, 1, 1], 'SAME')
        bias_conv = tf.nn.bias_add(conv, b)
        relu = tf.nn.relu(bias_conv)

        conv4_2 = relu

    with tf.variable_scope('conv4_3'):

        kernel_init = data_dict['conv4_3'][0]
        bias_init = data_dict['conv4_3'][1]

        W = tf.get_variable('W', kernel_init.shape, tf.float32,
                            tf.constant_initializer(kernel_init))
        b = tf.get_variable('b', bias_init.shape, tf.float32,
                            tf.constant_initializer(bias_init))

        conv = tf.nn.conv2d(conv4_2, W, [1, 1, 1, 1], 'SAME')
        bias_conv = tf.nn.bias_add(conv, b)
        relu = tf.nn.relu(bias_conv)

        conv4_3 = relu

    with tf.variable_scope('pool_4'):

        pool_4 = tf.nn.avg_pool(conv4_3, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    with tf.variable_scope('conv5_1'):

        kernel_init = data_dict['conv5_1'][0]
        bias_init = data_dict['conv5_1'][1]

        W = tf.get_variable('W', kernel_init.shape, tf.float32,
                            tf.constant_initializer(kernel_init))
        b = tf.get_variable('b', bias_init.shape, tf.float32,
                            tf.constant_initializer(bias_init))

        conv = tf.nn.conv2d(pool_4, W, [1, 1, 1, 1], 'SAME')
        bias_conv = tf.nn.bias_add(conv, b)
        relu = tf.nn.relu(bias_conv)

        conv4_1 = relu

    with tf.variable_scope('conv5_2'):

        kernel_init = data_dict['conv5_2'][0]
        bias_init = data_dict['conv5_2'][1]

        W = tf.get_variable('W', kernel_init.shape, tf.float32,
                            tf.constant_initializer(kernel_init))
        b = tf.get_variable('b', bias_init.shape, tf.float32,
                            tf.constant_initializer(bias_init))

        conv = tf.nn.conv2d(conv4_1, W, [1, 1, 1, 1], 'SAME')
        bias_conv = tf.nn.bias_add(conv, b)
        relu = tf.nn.relu(bias_conv)

        conv4_2 = relu

    with tf.variable_scope('conv5_3'):

        kernel_init = data_dict['conv5_3'][0]
        bias_init = data_dict['conv5_3'][1]

        W = tf.get_variable('W', kernel_init.shape, tf.float32,
                            tf.constant_initializer(kernel_init))
        b = tf.get_variable('b', bias_init.shape, tf.float32,
                            tf.constant_initializer(bias_init))

        conv = tf.nn.conv2d(conv4_2, W, [1, 1, 1, 1], 'SAME')
        bias_conv = tf.nn.bias_add(conv, b)
        relu = tf.nn.relu(bias_conv)

        conv5_3 = relu

    with tf.variable_scope('pool_5'):

        pool_5 = tf.nn.avg_pool(conv5_3, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    with tf.variable_scope('fc_6'):

        W = tf.get_variable(
            'W', [4, 5, 512, 1024], tf.float32, tf.truncated_normal_initializer(0.0, 0.001))
        b = tf.get_variable('b', [1024], tf.float32,
                            tf.constant_initializer(0.01))

        conv = tf.nn.conv2d(pool_5, W, [1, 1, 1, 1], 'SAME')
        bias_conv = tf.nn.bias_add(conv, b)
        relu = tf.nn.relu(bias_conv)

        fc_6 = relu

        fc_6 = tf.nn.dropout(fc_6, 0.5)

    with tf.variable_scope('fc_7'):

        W = tf.get_variable(
            'W', [1, 1, 1024, 1024], tf.float32, tf.truncated_normal_initializer(0.0, 0.001))
        b = tf.get_variable('b', [1024], tf.float32,
                            tf.constant_initializer(0.01))

        conv = tf.nn.conv2d(fc_6, W, [1, 1, 1, 1], 'SAME')
        bias_conv = tf.nn.bias_add(conv, b)
        relu = tf.nn.relu(bias_conv)

        fc_7 = relu

        fc_7 = tf.nn.dropout(fc_7, 0.5)

    with tf.variable_scope('fc_8'):

        W = tf.get_variable(
            'W', [1, 1, 1024, num_classes], tf.float32, tf.truncated_normal_initializer(0.0, 0.001))
        b = tf.get_variable('b', [num_classes], tf.float32,
                            tf.constant_initializer(0.01))

        conv = tf.nn.conv2d(fc_7, W, [1, 1, 1, 1], 'SAME')
        bias_conv = tf.nn.bias_add(conv, b)

        fc_8 = bias_conv


with tf.name_scope('output'):

    with tf.variable_scope('fuse_1'):

        W = tf.get_variable('W', [4, 4, 512, num_classes],
                            tf.float32, tf.truncated_normal_initializer(0, 0.001))
        b = tf.get_variable('b', [512],
                            tf.float32, tf.constant_initializer(0.01))

        deconv = tf.nn.conv2d_transpose(
            fc_8, W, tf.shape(pool_4), [1, 2, 2, 1], 'SAME')
        bias_deconv = tf.nn.bias_add(deconv, b)

        fuse_1 = tf.add(bias_deconv, pool_4)

    with tf.variable_scope('fuse_2'):

        W = tf.get_variable('W', [4, 4, 256, 512],
                            tf.float32, tf.truncated_normal_initializer(0, 0.001))
        b = tf.get_variable('b', [256],
                            tf.float32, tf.constant_initializer(0.01))

        deconv = tf.nn.conv2d_transpose(
            fuse_1, W, tf.shape(pool_3), [1, 2, 2, 1], 'SAME')
        bias_deconv = tf.nn.bias_add(deconv, b)

        fuse_2 = tf.add(bias_deconv, pool_3)

    with tf.variable_scope('fuse_3'):

        W = tf.get_variable('W', [16, 16, num_classes, 256],
                            tf.float32, tf.truncated_normal_initializer(0, 0.001))
        b = tf.get_variable('b', [num_classes],
                            tf.float32, tf.constant_initializer(0.01))

        shape = tf.shape(image)
        new_shape = tf.stack([shape[0], shape[1], shape[2], num_classes])

        deconv = tf.nn.conv2d_transpose(
            fuse_2, W, new_shape, [1, 8, 8, 1], 'SAME')

        logits = deconv

        annotation_pred = tf.expand_dims((tf.argmax(logits, axis=3)), dim=3)

        tf.add_to_collection('prediction', annotation_pred)

with tf.name_scope('loss'):

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=tf.squeeze(annotation, squeeze_dims=[3])))

with tf.name_scope('train'):

    train_op = tf.train.AdamOptimizer(1e-6).minimize(loss)

with tf.name_scope('export'):

    prediction = tf.cast(annotation_pred, tf.uint8, name='predicted')

init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

with tf.Session() as sess:

    sess.run(init)

    summary_op = tf.summary.merge_all()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    saver = tf.train.Saver()

    summary_writer = tf.summary.FileWriter(logs_dir, sess.graph)

    ckpt = tf.train.latest_checkpoint(logs_dir)
    if ckpt is not None:
        saver.restore(sess, ckpt)
        print('Model restored...')

    try:
        itr = 0

        while not coord.should_stop():

            training_image, training_annotation = sess.run(training_record)

            sess.run(train_op, feed_dict={image:  training_image,
                                              annotation:  training_annotation})

            if itr % 10 == 0:
                train_loss, summary_str = sess.run(
                    [loss, summary_op], feed_dict={image:  training_image,
                                                       annotation:  training_annotation})
                print("Step: %d, Train_loss: %g" % (itr, train_loss))
                summary_writer.add_summary(summary_str, itr)

            if itr % 250 == 0:
                validation_image, validation_annotation = sess.run(
                    validation_record)

                valid_loss = sess.run(loss, feed_dict={image: validation_image,
                                                           annotation: validation_annotation})

                print("Validation_loss: %g" % (valid_loss))
                saver.save(sess, logs_dir + 'model.ckpt', itr)

            itr += 1

    except tf.errors.OutOfRangeError:
        print('EoF')
    finally:
        coord.request_stop()

    coord.join(threads)
