"""
"""
#%%%
#pylint: disable=C0103
#pylint: disable=E1101
import numpy as np
import tensorflow as tf

def generate_data():
    """Generate data"""

    #train_input = ['{0:04b}'.format(i) for i in range(2**4)]
    #np.random.shuffle(train_input)
    #train_input = [map(float, i) for i in train_input]
    #ti = []
    #for i in train_input:
    #    temp_list = []
    #    for j in i:
    #        temp_list.append([j])
    #    ti.append(np.array(temp_list))
    #train_input = ti
    #print('1: ', train_input)
    #print(np.shape(train_input))

    #train_output = []
    #for i in train_input:
    #    count = 0
    #    for j in i:
    #        if j[0] == 1:
    #            count += 1
    #    temp_list = ([0]*5)
    #    temp_list[count] = 1
    #    train_output.append(temp_list)
    #print('2: ', train_input)
    #print(np.shape(train_input))

    #h, he, hel, hell
    #train_input = ['1000', '1100', '1110', '1111']
    train_input = ['a000', 'ab00', 'abc0', 'abcd']
    train_input = [map(ord, i) for i in train_input]
    #print(train_input)
    ti = []
    for st in train_input:
        temp_list = []
        for c in st:
            temp_list.append([c])
        ti.append(np.array(temp_list))
    train_input = ti

    #train_output = [[0, 1, 0, 0 ], [0, 0, 1, 0 ], [0, 0, 1, 0], [0, 0, 0, 1]]
    train_output = ['b', 'c', 'd', 'e']
    train_output = [map(ord, i) for i in train_output]

    #print('train_input: ', train_input)
    #print('\nshape: ', np.shape(train_input))

    #print('\ntrain_output: ', train_output)
    #print('\nshape: ', np.shape(train_output))

    #train_input = np.transpose(train_input, [0, 2, 1]).astype('float')

    writer = tf.python_io.TFRecordWriter('./RNN/data.tfrecords')

    for _, [features, labels] in enumerate(zip(train_input, train_output)):

        features = [float(i) for i in features]

        example = tf.train.SequenceExample()

        example.context.feature['length'].float_list.value.append(1.0)
        fl_features = example.feature_lists.feature_list['features']
        fl_labels = example.feature_lists.feature_list['labels']

        for token in features:
            fl_features.feature.add().float_list.value.append(token)
        for token_label in labels:
            fl_labels.feature.add().float_list.value.append(token_label)

        writer.write(example.SerializeToString())

        #print(example)

    writer.close()

    return

def read_record(filename_queue):
    """Read record"""

    reader = tf.TFRecordReader()
    _, record_string = reader.read(filename_queue)

    _, example = tf.parse_single_sequence_example(
        record_string,
        context_features={
            'length': tf.FixedLenFeature([], tf.float32)
        },
        sequence_features={
            'features': tf.FixedLenSequenceFeature([], tf.float32),
            'labels': tf.FixedLenSequenceFeature([], tf.float32)
        })

    feature = example['features']
    label = example['labels']

    return feature, label

def input_pipeline(filenames, batch_size=1, num_epochs=None):
    """Input pipeline"""

    filename_queue = tf.train.string_input_producer(
        filenames,
        num_epochs=num_epochs,
        shuffle=True
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

batch_size = 2
epochs = 10000
num_hidden = 24

generate_data()
feature, label = input_pipeline(['./RNN/data.tfrecords'], batch_size, epochs)

data = tf.placeholder(tf.float32, [None, 4, 1])
#Number of examples, number of input, dimension of each input
target = tf.placeholder(tf.float32, [None, 1])

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

#prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
#cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
#optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)
#mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
#error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

prediction = tf.matmul(last, weight) + bias
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

            batch_feature, batch_label = s.run([feature, label])

            #print('\nbatch_feature: ', batch_feature)
            #print('\nshape: ', np.shape(batch_feature))
            #print('\nbatch_label: ', batch_label)
            #print('\nshape: ', np.shape(batch_label))

            #print(np.reshape(batch_feature, [batch_size, 10, 1]))
            #print(np.shape(np.reshape(batch_feature, [batch_size, 10, 1])))

            batch_feature = np.reshape(batch_feature, [batch_size, 4, 1])

            #batch_feature = np.transpose(batch_feature, [0, 2, 1])
            #print('\nbatch_feature^T: ', s.run(batch_feature))
            #print('\nshape: ', s.run(tf.shape(s.run(batch_feature))))

            #batch_label = batch_label[:, 0]
            #print('\nbatch_label slice: ', batch_label)
            #print('\nshape: ', np.shape(batch_label))

            s.run(optimizer, {data: batch_feature, target: batch_label})

    except tf.errors.OutOfRangeError:
        print('EoF')
    finally:
        coord.request_stop()
    coord.join(threads)

    x = [[[ord('a')], [ord('b')], [ord('0')], [ord('0')]]]
    print(x)
    print('\n', s.run(prediction, {data: x}))
    #x = [[[1], [1], [0], [0]]]
    #print(x)
    #print('\n', s.run(prediction, {data: x}))
    #x = [[[1], [1], [1], [0]]]
    #print(x)
    #print('\n', s.run(prediction, {data: x}))
    #x = [[[1], [1], [1], [1]]]
    #print(x)
    #print('\n', s.run(prediction, {data: x}))

    s.close()
