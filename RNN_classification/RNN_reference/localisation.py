'''
'''
#%%%
#pylint: disable=C0103
#pylint: disable=E1101
import numpy as np
import tensorflow as tf

def generate_data():
    '''Generate data'''
    #cur_pos = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
    #next_pos = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    cur_pos = np.zeros([3, 4, 4])
    for i in range(3):
        cur_pos[i, i, i] = 1
    cur_pos = np.reshape(cur_pos, [3, 16])

    # print(cur_pos)
    # print(np.shape(cur_pos))

    next_pos = np.zeros([3, 4, 4])
    for i in range(3):
        next_pos[i, i+1, i+1] = 1
    next_pos = np.reshape(next_pos, [3, 16])

    # print(next_pos)
    # print(np.shape(next_pos))

    record = tf.python_io.TFRecordWriter('./data.tfrecords')

    for _, [features, labels] in enumerate(zip(cur_pos, next_pos)):

        input_features = [tf.train.Feature(float_list=tf.train.FloatList(value=features))]
        label_feature = [tf.train.Feature(float_list=tf.train.FloatList(value=labels))]
        feature_list = {
            'feature_list': tf.train.FeatureList(feature=input_features),
            'feature_list_labels': tf.train.FeatureList(feature=label_feature)
        }
        feature_lists = tf.train.FeatureLists(feature_list=feature_list)
        example = tf.train.SequenceExample(feature_lists=feature_lists)
        record.write(example.SerializeToString())

    record.close()

    return

def read_record(filename_queue):
    """Read record"""

    reader = tf.TFRecordReader()
    _, record_string = reader.read(filename_queue)

    _, example = tf.parse_single_sequence_example(
        record_string,
        None,
        sequence_features={
            'feature_list': tf.FixedLenSequenceFeature(16, tf.float32),
            'feature_list_labels': tf.FixedLenSequenceFeature(16, tf.float32)
        })

    feature = example['feature_list']
    label = example['feature_list_labels']

    return feature, label

def input_pipeline(filenames, batch_size=1, num_epochs=None):
    """Input pipeline"""

    filename_queue = tf.train.string_input_producer(
        filenames,
        num_epochs=num_epochs,
        shuffle=False
    )

    example, label = read_record(filename_queue)

    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.batch(
        [example, label],
        batch_size=batch_size,
        capacity=capacity,
        dynamic_pad=True,
        )

    return example_batch, label_batch

generate_data()

batch_size = 2
epochs = 1000
num_hidden = 20

feature_batch, label_batch = input_pipeline(['./data.tfrecords'], batch_size, epochs)

data = tf.placeholder(tf.float32, [None, 1, 16])
target = tf.placeholder(tf.float32, [None, 16])

weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.0, shape=[target.get_shape()[1]]))

cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)

with tf.variable_scope('vs'):
    try:
        output, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
        val = tf.transpose(output, [1, 0, 2])
        #last = tf.reshape(val, [-1, 1])
        last = tf.gather(val, int(val.get_shape()[0]) - 1)
    except BaseException:
        tf.get_variable_scope().reuse_variables()
        output, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
        val = tf.transpose(output, [1, 0, 2])
        #last = tf.reshape(val, [-1, 1])
        last = tf.gather(val, int(val.get_shape()[0]) - 1)

prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
#cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
#optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)
#mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
#error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

#prediction = tf.matmul(last, weight) + bias
cost = tf.reduce_sum(tf.square(tf.sub(prediction, target)))
optimizer = tf.train.AdamOptimizer().minimize(cost)

with tf.Session() as s:

    s.run(tf.local_variables_initializer())
    s.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(s, coord=coord)

    try:
        i = 0
        while not coord.should_stop():

            i += 1

            batch_feature, batch_label = s.run([feature_batch, label_batch])

            # print(np.shape(batch_label))
            batch_label = np.reshape(batch_label, [-1, 16])
            # print(batch_label)
            # print(np.shape(batch_label))

            s.run(optimizer, {data:batch_feature, target:batch_label})

    except tf.errors.OutOfRangeError:
        print('EoF')
    finally:
        coord.request_stop()
    coord.join(threads)

    x = np.zeros([1, 4, 4])
    x[0, 1, 1] = 1
    print(x)
    x = np.reshape(x, [1, 16])
    print('\n', np.reshape(s.run(prediction, {data: [x]}), [1, 1, 16]))

