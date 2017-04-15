'''TensorFlow Utility Functions'''

#%%

# pylint: disable=E0401
# pylint: disable=C0103
# pylint: disable=W0621

import os
import sys
import zipfile
import tarfile
import numpy as np
import scipy
import scipy.io
import scipy.misc as misc
import tensorflow as tf
from six.moves import urllib


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


def float_feature(value):
    '''Create float_list-based feature'''
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int_feature(value):
    '''Create int64_list-based feature'''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    '''Create bytes_list-based feature'''
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def make_example(feature, label):
    '''Make example from feature and label'''

    feature_raw = feature.tostring()
    label_raw = label.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'feature_height': int_feature(feature.shape[0]),
        'feature_width': int_feature(feature.shape[1]),
        'feature_depth': int_feature(feature.shape[2]),
        'label_height': int_feature(label.shape[0]),
        'label_width': int_feature(label.shape[1]),
        'label_depth': int_feature(label.shape[2]),
        'feature_raw': bytes_feature(feature_raw),
        'label_raw': bytes_feature(label_raw),
    }))

    return example


def output_pipeline(filenames, num_epochs=1):
    """Write a TFRecord"""

    filename_queue = tf.train.string_input_producer(
        filenames,
        num_epochs=num_epochs,
        shuffle=False
    )

    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)

    image = tf.image.decode_png(value)

    return image


def read_record(filename_queue):
    '''Read record'''

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


def extract_example_data(example):
    '''Extract example's data'''

    feature_height = tf.cast(example['feature_height'], tf.int32)
    feature_width = tf.cast(example['feature_width'], tf.int32)
    feature_depth = tf.cast(example['feature_depth'], tf.int32)
    label_height = tf.cast(example['label_height'], tf.int32)
    label_width = tf.cast(example['label_width'], tf.int32)
    label_depth = tf.cast(example['label_depth'], tf.int32)
    feature_raw = tf.decode_raw(example['feature_raw'], tf.uint8)
    label_raw = tf.decode_raw(example['label_raw'], tf.uint8)

    feature = tf.reshape(feature_raw, tf.stack(
        [feature_height, feature_width, feature_depth]))
    label = tf.reshape(label_raw, tf.stack(
        [label_height, label_width, label_depth]))

    return feature, label


def input_pipeline(filenames, num_epochs=1, batch_size=1):
    """Read a TFRecord"""

    filename_queue = tf.train.string_input_producer(
        filenames,
        num_epochs=num_epochs,
        shuffle=False
    )

    example = read_record(filename_queue)

    feature, label = extract_example_data(example)

    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    feature_batch, label_batch = tf.train.batch(
        [feature, label],
        batch_size=batch_size,
        capacity=capacity,
        dynamic_pad=True)

    return feature_batch, label_batch


def conv_layer(input, kernel_shape, bias_shape, stride=1, kernel_init=None, bias_init=None):
    '''Create a convolutional layer.'''

    if kernel_init is None:
        weights = tf.get_variable(
            name='_W', shape=kernel_shape, initializer=tf.random_normal_initializer(mean=1.0))
    else:
        weights = tf.get_variable(
            name='_W', shape=kernel_shape, initializer=tf.constant_initializer(kernel_init))

    if bias_init is None:
        biases = tf.get_variable(
            name='_b', shape=bias_shape, initializer=tf.constant_initializer(0.1))
    else:
        biases = tf.get_variable(
            name='_b', shape=bias_shape, initializer=tf.constant_initializer(bias_init))

    conv = tf.nn.conv2d(input, weights, strides=[1, stride, stride, 1], padding='SAME')

    return tf.nn.bias_add(conv, biases)


def transpose_conv_layer(input, kernel_shape, bias_shape, output_shape, stride=2):
    '''Create a transpose convolutional layer.'''

    weights = tf.get_variable(
        name='_W', shape=kernel_shape, initializer=tf.random_normal_initializer(mean=1.0))
    biases = tf.get_variable(
        name='_b', shape=bias_shape, initializer=tf.constant_initializer(0.1))

    conv = tf.nn.conv2d_transpose(input, weights, output_shape, strides=[
        1, stride, stride, 1], padding="SAME")

    return tf.nn.bias_add(conv, biases)

# End of my utilities.


def get_model_data(dir_path, model_url):
    maybe_download_and_extract(dir_path, model_url)
    filename = model_url.split("/")[-1]
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        raise IOError("VGG Model not found!")
    data = scipy.io.loadmat(filepath)
    return data


def maybe_download_and_extract(dir_path, url_name, is_tarfile=False, is_zipfile=False):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    filename = url_name.split('/')[-1]
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(
            url_name, filepath, reporthook=_progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
        if is_tarfile:
            tarfile.open(filepath, 'r:gz').extractall(dir_path)
        elif is_zipfile:
            with zipfile.ZipFile(filepath) as zf:
                zip_dir = zf.namelist()[0]
                zf.extractall(dir_path)


def save_image(image, save_dir, name, mean=None):
    """
    Save image by unprocessing if mean given else just save
    :param mean:
    :param image:
    :param save_dir:
    :param name:
    :return:
    """
    if mean:
        image = unprocess_image(image, mean)
    misc.imsave(os.path.join(save_dir, name + ".png"), image)


def get_variable(weights, name):
    init = tf.constant_initializer(weights, dtype=tf.float32)
    var = tf.get_variable(name=name, initializer=init,  shape=weights.shape)
    return var


def weight_variable(shape, stddev=0.02, name=None):
    # print(shape)
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def get_tensor_size(tensor):
    from operator import mul
    return tf.reduce_all(mul, (d.value for d in tensor.get_shape()), 1)


def conv2d_basic(x, W, bias):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)


def conv2d_strided(x, W, b):
    conv = tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)


def conv2d_transpose_strided(x, W, b, output_shape=None, stride=2):
    # print x.get_shape()
    # print W.get_shape()
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]
    # print output_shape
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[
                                  1, stride, stride, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)


def leaky_relu(x, alpha=0.0, name=""):
    return tf.maximum(alpha * x, x, name)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def local_response_norm(x):
    return tf.nn.lrn(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75)


def batch_norm(x, n_out, phase_train, scope='bn', decay=0.9, eps=1e-5):
    """
    Code taken from http://stackoverflow.com/a/34634291/2267819
    """
    with tf.variable_scope(scope):
        beta = tf.get_variable(name='beta', shape=[
                               n_out], initializer=tf.constant_initializer(0.0), trainable=True)
        gamma = tf.get_variable(name='gamma', shape=[n_out], initializer=tf.random_normal_initializer(1.0, 0.02),
                                trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
    return normed


def process_image(image, mean_pixel):
    return image - mean_pixel


def unprocess_image(image, mean_pixel):
    return image + mean_pixel


def bottleneck_unit(x, out_chan1, out_chan2, down_stride=False, up_stride=False, name=None):
    """
    Modified implementation from github ry?!
    """

    def conv_transpose(tensor, out_channel, shape, strides, name=None):
        out_shape = tensor.get_shape().as_list()
        in_channel = out_shape[-1]
        kernel = weight_variable(
            [shape, shape, out_channel, in_channel], name=name)
        shape[-1] = out_channel
        return tf.nn.conv2d_transpose(x, kernel, output_shape=out_shape, strides=[1, strides, strides, 1],
                                      padding='SAME', name='conv_transpose')

    def conv(tensor, out_chans, shape, strides, name=None):
        in_channel = tensor.get_shape().as_list()[-1]
        kernel = weight_variable(
            [shape, shape, in_channel, out_chans], name=name)
        return tf.nn.conv2d(x, kernel, strides=[1, strides, strides, 1], padding='SAME', name='conv')

    def bn(tensor, name=None):
        """
        :param tensor: 4D tensor input
        :param name: name of the operation
        :return: local response normalized tensor - not using batch normalization :(
        """
        return tf.nn.lrn(tensor, depth_radius=5, bias=2, alpha=1e-4, beta=0.75, name=name)

    in_chans = x.get_shape().as_list()[3]

    if down_stride or up_stride:
        first_stride = 2
    else:
        first_stride = 1

    with tf.variable_scope('res%s' % name):
        if in_chans == out_chan2:
            b1 = x
        else:
            with tf.variable_scope('branch1'):
                if up_stride:
                    b1 = conv_transpose(x, out_chans=out_chan2, shape=1, strides=first_stride,
                                        name='res%s_branch1' % name)
                else:
                    b1 = conv(x, out_chans=out_chan2, shape=1,
                              strides=first_stride, name='res%s_branch1' % name)
                b1 = bn(b1, 'bn%s_branch1' % name, 'scale%s_branch1' % name)

        with tf.variable_scope('branch2a'):
            if up_stride:
                b2 = conv_transpose(x, out_chans=out_chan1, shape=1,
                                    strides=first_stride, name='res%s_branch2a' % name)
            else:
                b2 = conv(x, out_chans=out_chan1, shape=1,
                          strides=first_stride, name='res%s_branch2a' % name)
            b2 = bn(b2, 'bn%s_branch2a' % name, 'scale%s_branch2a' % name)
            b2 = tf.nn.relu(b2, name='relu')

        with tf.variable_scope('branch2b'):
            b2 = conv(b2, out_chans=out_chan1, shape=3,
                      strides=1, name='res%s_branch2b' % name)
            b2 = bn(b2, 'bn%s_branch2b' % name, 'scale%s_branch2b' % name)
            b2 = tf.nn.relu(b2, name='relu')

        with tf.variable_scope('branch2c'):
            b2 = conv(b2, out_chans=out_chan2, shape=1,
                      strides=1, name='res%s_branch2c' % name)
            b2 = bn(b2, 'bn%s_branch2c' % name, 'scale%s_branch2c' % name)

        x = b1 + b2
        return tf.nn.relu(x, name='relu')


def add_to_regularization_and_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name, var)
        tf.add_to_collection("reg_loss", tf.nn.l2_loss(var))


def add_activation_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name + "/activation", var)
        tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))


def add_gradient_summary(grad, var):
    if grad is not None:
        tf.summary.histogram(var.op.name + "/gradient", grad)
