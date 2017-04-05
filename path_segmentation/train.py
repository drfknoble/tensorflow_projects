'''path_segmentation'''

#%%

# pylint: disable=E0401
# pylint: disable=C0103
# pylint: disable=W0621

# Import. Here, we import tensorflow, which gives us access to the library.
import os
import tensorflow as tf

# Here, we define a function that reads a TFRecord file; parsing a single
# example.
def read_record(filename_queue):
    '''Read record'''

    # Here, the record contains examples derived from PNG images.

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

# Here, we extract data from a single example.
def extract_example_data(example):
    '''Extract example's data'''

    # feature = sequence_example['feature_list']

    feature_height = tf.cast(example['feature_height'], tf.int32)
    feature_width = tf.cast(example['feature_width'], tf.int32)
    feature_depth = tf.cast(example['feature_depth'], tf.int32)
    label_height = tf.cast(example['label_height'], tf.int32)
    label_width = tf.cast(example['label_width'], tf.int32)
    label_depth = tf.cast(example['label_depth'], tf.int32)
    feature_raw = tf.decode_raw(example['feature_raw'], tf.uint8)
    label_raw = tf.decode_raw(example['label_raw'], tf.uint8)

    # feature = tf.reshape(feature_raw, tf.stack([feature_height, feature_width, feature_depth]))
    # label = tf.reshape(label_raw, tf.stack([label_height, label_width, label_depth]))

    feature = tf.reshape(feature_raw, tf.stack([feature_height, feature_width, feature_depth]))
    label = tf.reshape(label_raw, tf.stack([label_height, label_width, label_depth]))

    return feature, label

# Here, we define a function that reads a TFRecord file.
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

# Here, we define important directories.
input_dir = './data/input/'
output_dir = './data/output/'

# Here, we define important file names.
record_file = output_dir + 'training_record.tfrecords'

# Here, we define the number of times we read a record file, and what size
# each batch is.
num_epochs = 1
batch_size = 1

# Here, we create handles for reading and writing TFRecord files.
record = input_pipeline([record_file], num_epochs, batch_size)

# Here, we define out input
with tf.name_scope('input'):

    x = tf.placeholder(tf.float32, shape=[None, None, None, None], name='x')
    y = tf.placeholder(tf.int32, shape=[None, None, None, None], name='y')

# Here, we define our graph.
with tf.name_scope('network'):

    token = x

    with tf.variable_scope('conv1_1'):

        kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(0, 0.2),
                                 shape=[3, 3, 3, 64], name='kernel')
        bias = tf.get_variable(initializer=tf.constant_initializer(0.1), shape=[64], name='bias')
        conv = tf.nn.conv2d(token, kernel, strides=[1, 1, 1, 1], padding='SAME')
        token = tf.nn.bias_add(conv, bias)

    with tf.variable_scope('relu1_1'):

        token = tf.nn.relu(token)

    with tf.variable_scope('conv1_2'):

        kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(0, 0.2),
                                 shape=[3, 3, 64, 64], name='kernel')
        bias = tf.get_variable(initializer=tf.constant_initializer(0.1), shape=[64], name='bias')
        conv = tf.nn.conv2d(token, kernel, strides=[1, 1, 1, 1], padding='SAME')
        token = tf.nn.bias_add(conv, bias)

    with tf.variable_scope('relu1_2'):

        token = tf.nn.relu(token)

    with tf.variable_scope('pool1'):

        token = tf.nn.avg_pool(token, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('conv2_1'):

        kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(0, 0.2),
                                 shape=[3, 3, 64, 128], name='kernel')
        bias = tf.get_variable(initializer=tf.constant_initializer(0.1), shape=[128], name='bias')
        conv = tf.nn.conv2d(token, kernel, strides=[1, 1, 1, 1], padding='SAME')
        token = tf.nn.bias_add(conv, bias)

    with tf.variable_scope('relu2_1'):

        token = tf.nn.relu(token)

    with tf.variable_scope('conv2_2'):

        kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(0, 0.2),
                                 shape=[3, 3, 128, 128], name='kernel')
        bias = tf.get_variable(initializer=tf.constant_initializer(0.1), shape=[128], name='bias')
        conv = tf.nn.conv2d(token, kernel, strides=[1, 1, 1, 1], padding='SAME')
        token = tf.nn.bias_add(conv, bias)

    with tf.variable_scope('relu2_2'):

        token = tf.nn.relu(token)

    with tf.variable_scope('pool2'):

        token = tf.nn.avg_pool(token, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('conv3_1'):

        kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(0, 0.2),
                                 shape=[3, 3, 128, 256], name='kernel')
        bias = tf.get_variable(initializer=tf.constant_initializer(0.1), shape=[256], name='bias')
        conv = tf.nn.conv2d(token, kernel, strides=[1, 1, 1, 1], padding='SAME')
        token = tf.nn.bias_add(conv, bias)

    with tf.variable_scope('relu3_1'):

        token = tf.nn.relu(token)

    with tf.variable_scope('conv3_2'):

        kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(0, 0.2),
                                 shape=[3, 3, 256, 256], name='kernel')
        bias = tf.get_variable(initializer=tf.constant_initializer(0.1), shape=[256], name='bias')
        conv = tf.nn.conv2d(token, kernel, strides=[1, 1, 1, 1], padding='SAME')
        token = tf.nn.bias_add(conv, bias)

    with tf.variable_scope('relu3_2'):

        token = tf.nn.relu(token)

    with tf.variable_scope('conv3_3'):

        kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(0, 0.2),
                                 shape=[3, 3, 256, 256], name='kernel')
        bias = tf.get_variable(initializer=tf.constant_initializer(0.1), shape=[256], name='bias')
        conv = tf.nn.conv2d(token, kernel, strides=[1, 1, 1, 1], padding='SAME')
        token = tf.nn.bias_add(conv, bias)

    with tf.variable_scope('relu3_3'):

        token = tf.nn.relu(token)

    with tf.variable_scope('conv3_4'):

        kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(0, 0.2),
                                 shape=[3, 3, 256, 256], name='kernel')
        bias = tf.get_variable(initializer=tf.constant_initializer(0.1), shape=[256], name='bias')
        conv = tf.nn.conv2d(token, kernel, strides=[1, 1, 1, 1], padding='SAME')
        token = tf.nn.bias_add(conv, bias)

    with tf.variable_scope('relu3_4'):

        token = tf.nn.relu(token)

    with tf.variable_scope('pool3'):

        token = tf.nn.avg_pool(token, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('conv4_1'):

        kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(0, 0.2),
                                 shape=[3, 3, 256, 512], name='kernel')
        bias = tf.get_variable(initializer=tf.constant_initializer(0.1), shape=[512], name='bias')
        conv = tf.nn.conv2d(token, kernel, strides=[1, 1, 1, 1], padding='SAME')
        token = tf.nn.bias_add(conv, bias)

    with tf.variable_scope('relu4_1'):

        token = tf.nn.relu(token)

    with tf.variable_scope('conv4_2'):

        kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(0, 0.2),
                                 shape=[3, 3, 512, 512], name='kernel')
        bias = tf.get_variable(initializer=tf.constant_initializer(0.1), shape=[512], name='bias')
        conv = tf.nn.conv2d(token, kernel, strides=[1, 1, 1, 1], padding='SAME')
        token = tf.nn.bias_add(conv, bias)

    with tf.variable_scope('relu4_2'):

        token = tf.nn.relu(token)

    with tf.variable_scope('conv4_3'):

        kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(0, 0.2),
                                 shape=[3, 3, 512, 512], name='kernel')
        bias = tf.get_variable(initializer=tf.constant_initializer(0.1), shape=[512], name='bias')
        conv = tf.nn.conv2d(token, kernel, strides=[1, 1, 1, 1], padding='SAME')
        token = tf.nn.bias_add(conv, bias)

    with tf.variable_scope('relu4_3'):

        token = tf.nn.relu(token)

    with tf.variable_scope('conv4_4'):

        kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(0, 0.2),
                                 shape=[3, 3, 512, 512], name='kernel')
        bias = tf.get_variable(initializer=tf.constant_initializer(0.1), shape=[512], name='bias')
        conv = tf.nn.conv2d(token, kernel, strides=[1, 1, 1, 1], padding='SAME')
        token = tf.nn.bias_add(conv, bias)

    with tf.variable_scope('relu4_4'):

        token = tf.nn.relu(token)

    with tf.variable_scope('pool4'):

        token = tf.nn.avg_pool(token, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('conv5_1'):

        kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(0, 0.2),
                                 shape=[3, 3, 512, 512], name='kernel')
        bias = tf.get_variable(initializer=tf.constant_initializer(0.1), shape=[512], name='bias')
        conv = tf.nn.conv2d(token, kernel, strides=[1, 1, 1, 1], padding='SAME')
        token = tf.nn.bias_add(conv, bias)

    with tf.variable_scope('relu5_1'):

        token = tf.nn.relu(token)

    with tf.variable_scope('conv5_2'):

        kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(0, 0.2),
                                 shape=[3, 3, 512, 512], name='kernel')
        bias = tf.get_variable(initializer=tf.constant_initializer(0.1), shape=[512], name='bias')
        conv = tf.nn.conv2d(token, kernel, strides=[1, 1, 1, 1], padding='SAME')
        token = tf.nn.bias_add(conv, bias)

    with tf.variable_scope('relu5_2'):

        token = tf.nn.relu(token)

    with tf.variable_scope('conv5_3'):

        kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(0, 0.2),
                                 shape=[3, 3, 512, 512], name='kernel')
        bias = tf.get_variable(initializer=tf.constant_initializer(0.1), shape=[512], name='bias')
        conv = tf.nn.conv2d(token, kernel, strides=[1, 1, 1, 1], padding='SAME')
        token = tf.nn.bias_add(conv, bias)

    with tf.variable_scope('relu5_3'):

        token = tf.nn.relu(token)

    with tf.variable_scope('conv5_4'):

        kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(0, 0.2),
                                 shape=[3, 3, 512, 512], name='kernel')
        bias = tf.get_variable(initializer=tf.constant_initializer(0.1), shape=[512], name='bias')
        conv = tf.nn.conv2d(token, kernel, strides=[1, 1, 1, 1], padding='SAME')
        token = tf.nn.bias_add(conv, bias)

    with tf.variable_scope('relu5_4'):

        token = tf.nn.relu(token)

    with tf.variable_scope('pool5'):

        token = tf.nn.avg_pool(token, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('conv6_1'):

        kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(0, 0.2),
                                 shape=[7, 7, 512, 4096], name='kernel')
        bias = tf.get_variable(initializer=tf.constant_initializer(0.1), shape=[4096], name='bias')
        conv = tf.nn.conv2d(token, kernel, strides=[1, 1, 1, 1], padding='SAME')
        token = tf.nn.bias_add(conv, bias)

    with tf.variable_scope('relu6_1'):

        token = tf.nn.relu(token)
        token = tf.nn.dropout(token, 0.5)

    with tf.variable_scope('conv7_1'):

        kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(0, 0.2),
                                 shape=[1, 1, 4096, 4096], name='kernel')
        bias = tf.get_variable(initializer=tf.constant_initializer(0.1), shape=[4096], name='bias')
        conv = tf.nn.conv2d(token, kernel, strides=[1, 1, 1, 1], padding='SAME')
        token = tf.nn.bias_add(conv, bias)

    with tf.variable_scope('relu7_1'):

        token = tf.nn.relu(token)
        token = tf.nn.dropout(token, 0.5)

    with tf.variable_scope('conv8_1'):

        kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(0, 0.2),
                                 shape=[1, 1, 4096, 4096], name='kernel')
        bias = tf.get_variable(initializer=tf.constant_initializer(0.1), shape=[4096], name='bias')
        conv = tf.nn.conv2d(token, kernel, strides=[1, 1, 1, 1], padding='SAME')
        token = tf.nn.bias_add(conv, bias)

    with tf.variable_scope('relu8_1'):

        token = tf.nn.relu(token)
        token = tf.nn.dropout(token, 0.5)

    with tf.variable_scope('reshape'):
        shape = tf.shape(x)
        token = tf.image.resize_images(token, [shape[1], shape[2]])

# Here, we define our output
with tf.name_scope('Output'):

    y_ = token

# cost = tf.losses.mean_squared_error(y, y_)
cost = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=y_,
    labels=tf.squeeze(y, squeeze_dims=[3]))))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)
tf.summary.scalar('Cost', cost)

# Initialisation commands
init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

# In this session, we read the TFRecord file and use its examples with our
# graph.
with tf.Session() as l:

    l.run(init)

    summary_writer = tf.summary.FileWriter('./logs', l.graph)

    merged = tf.summary.merge_all()

    saver = tf.train.Saver()

    model_dir = 'E:/Users/fknoble/My_FCN/' # for very large models.

    try:
        loader = tf.train.import_meta_graph(model_dir + 'model/main.meta')
        ckpt = tf.train.latest_checkpoint(model_dir + 'model/')
        loader.restore(l, ckpt)
    except IOError as e:
        saver.save(l, model_dir + '/model/main.ckpt', 0)
        saver.export_meta_graph(model_dir + 'model/main.meta')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        i = 0

        print('Step ' + str(i))

        while not coord.should_stop():

            feature, label = l.run(record)

            feed_dict = {x: feature, y: label}
            summary, c, _ = l.run([merged, cost, optimizer], feed_dict)

            summary_writer.add_summary(summary, i)

            if i % 1 == 0:
                print('Step ' + str(i))
                print('Cost: ' + c)

            if i % 50 == 0:
                saver.save(l, model_dir + 'model/main.ckpt', i)

            i += 1

    except tf.errors.OutOfRangeError:
        print('EoF')
    finally:
        coord.request_stop()

    coord.join(threads)

    test = l.run(y_, {x: feature})[0]

    f_file = l.run(tf.image.encode_png(test))
    W = open(output_dir + './test.png', 'wb+')
    W.write(f_file)
    W.close()
