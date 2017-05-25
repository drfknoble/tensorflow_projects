
'''
'''
#%%%
#pylint: disable=C0103
#pylint: disable=E1101
import numpy as np
import tensorflow as tf

def generate_paths(p=2, n=4, x_o=0, y_o=0, x_t=1, y_t=1):
    '''Generates an array of paths from (x_o,y_o) to (x_t,y_t)'''

    path_array = []

    for i in range(p):

        x = x_o
        y = y_o

        h = abs(x_t - x) + abs(y_t - y)

        local_map = np.zeros([n, n], 'int')

        local_map[x, y] = 1

        path_history = [local_map]

        while True:

            viable = False

            delta = np.random.choice([-1, 1])

            direction = np.random.choice([0, 1])
            if direction == 0:
                if abs(x_t - (x + delta)) < abs(x_t - x):
                    x = x + delta
                    viable = True
            else:
                if abs(y_t - (y + delta)) < abs(y_t - y):
                    y = y + delta
                    viable = True

            if viable:

                local_map = np.zeros([n, n], 'int')
                local_map[x, y] = 1
                path_history.append(local_map)

                h = abs(x_t - x) + abs(y_t - y)

            if h == 0:
                break

        path_array.append(path_history)

        path_history = np.reshape(path_history, [-1, 16])
        # print(path_history)
        # print(np.shape(path_history))

        write_sequence_record(path_history[0:6], [path_history[6]])

    return path_array

def write_sequence_record(sequence, sequence_label):
    '''Write path to record'''

    record = tf.python_io.TFRecordWriter('./data.tfrecords')

    input_features = [
        tf.train.Feature(float_list=tf.train.FloatList(value=token))
        for token in sequence]
    label_feature = [
        tf.train.Feature(float_list=tf.train.FloatList(value=token_label))
        for token_label in sequence_label]
    feature_list = {
        'feature_list': tf.train.FeatureList(feature=input_features),
        'feature_list_labels': tf.train.FeatureList(feature=label_feature)
    }
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    example = tf.train.SequenceExample(feature_lists=feature_lists)
    record.write(example.SerializeToString())

    record.close()

    return

def read_sequence_record(filename_queue):
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

    example, label = read_sequence_record(filename_queue)

    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.batch(
        [example, label],
        batch_size=batch_size,
        capacity=capacity,
        dynamic_pad=True,
        )

    return example_batch, label_batch

paths = generate_paths(100, 4, 0, 0, 3, 3)

batch_size = 10
epochs = 1000
num_hidden = 20

feature_batch, label_batch = input_pipeline(['./data.tfrecords'], batch_size, epochs)

data = tf.placeholder(tf.float32, [None, 6, 16])
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

tf.summary.scalar('cost', cost)
merged = tf.summary.merge_all()

with tf.Session() as s:

    s.run(tf.local_variables_initializer())
    s.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(s, coord=coord)

    summary_writer = tf.train.SummaryWriter('./logs/', s.graph)

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

            if i%10 == 0:
                cost_p = s.run(cost, {data:batch_feature, target:batch_label})
                print(cost_p)
                summary = s.run(merged, {data: batch_feature, target: batch_label})
                summary_writer.add_summary(summary, i)
                # print(summary)

    except tf.errors.OutOfRangeError:
        print('EoF')
    finally:
        coord.request_stop()
    coord.join(threads)

    saver = tf.train.Saver()
    saver.save(s, './models/localisation_model')

with tf.Session() as f:

    loader = tf.train.import_meta_graph('./models/localisation_model.meta')
    loader.restore(f, tf.train.latest_checkpoint('./models/'))

    paths = generate_paths(1, 4, 0, 0, 3, 3)

    for i in paths[0][0:6]:
        print(i, '\n')
    x = np.reshape(paths[0], [-1, 16])
    print(f.run(prediction, {data: [x[0:6]]}))
    