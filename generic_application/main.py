'''generic_application'''

# pylint: disable=E0401
# pylint: disable=C0103

import tensorflow as tf 

def float_feature(value):
    '''Create float_list-based feature'''
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def int_feature(value):
    '''Create float_list-based feature'''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
    '''Create bytes_list-based feature'''
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def make_example(feature, label=None):
    '''Make example from feature'''

    example = tf.train.Example(features=tf.train.Features(feature={
        'x': float_feature(feature[0]),
        'y': float_feature(feature[1]),
    }))

    return example

def read_record(filename_queue):
    '''Read record'''

    reader = tf.TFRecordReader()
    _, serialised_example = reader.read(filename_queue)

    example = tf.parse_single_example(
        serialised_example,
        features={
            'x': tf.FixedLenFeature([], tf.float32),
            'y': tf.FixedLenFeature([], tf.float32),
        }
    )

    return example

def input_pipeline(filenames, num_epochs=1):
    """Read a TFRecord"""

    filename_queue = tf.train.string_input_producer(
        filenames,
        num_epochs=num_epochs,
        shuffle=False
    )

    record = read_record(filename_queue)

    return record

def output_pipeline(filenames, num_epochs=1):
    """Write a TFRecord"""

    filename_queue = tf.train.string_input_producer(
        filenames,
        num_epochs=num_epochs,
        shuffle=False
    )

    reader = tf.TextLineReader()
    _, value = reader.read(filename_queue)

    x, y = tf.decode_csv(value, record_defaults=[[0.0], [0.0]])

    return [x, y]

input_dir = './data/input/'
output_dir = './data/output/'

csv_file = input_dir + 'csv_data.csv'
record_file = output_dir + 'csv_record.tfrecords'

csv_data = output_pipeline([csv_file], 1)
record = input_pipeline([record_file], 1)

with tf.name_scope('input'):
    A = tf.placeholder(tf.float32, shape=None, name='A')
    B = tf.placeholder(tf.float32, shape=None, name='B')

    tf.summary.scalar('A', A)
    tf.summary.scalar('B', B)

with tf.name_scope('output'):

    C = tf.Variable(0, name='C')

    C = A + B

    tf.summary.scalar('C', C)

init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

with tf.Session() as s:

    s.run(init)

    writer = tf.python_io.TFRecordWriter(record_file)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        while not coord.should_stop():

            feature = s.run(csv_data)

            example = make_example(feature)

            print(example)

            writer.write(example.SerializeToString())

    except tf.errors.OutOfRangeError:
        print('EoF')
    finally:
        coord.request_stop()

    coord.join(threads)

    writer.close()

with tf.Session() as l:

    l.run(init)

    summary_writer = tf.summary.FileWriter('./logs', s.graph)

    merged = tf.summary.merge_all()

    saver = tf.train.Saver()
    saver.save(l, './model/main.ckpt', 0)
    saver.export_meta_graph('./model/main.meta')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        i = 0

        while not coord.should_stop():

            r = l.run(record)

            x = l.run(tf.cast(r['x'], tf.int64))
            y = l.run(tf.cast(r['y'], tf.int64))

            feed_dict = {A: x, B: y}
            summary, ans = l.run([merged, C], feed_dict)

            summary_writer.add_summary(summary, i)

            print(ans)

            # if i%4 == 0:
            saver.save(l, './model/main.ckpt', i)

            i += 1

    except tf.errors.OutOfRangeError:
        print('EoF')
    finally:
        coord.request_stop()

    coord.join(threads)

with tf.Session() as f:

    loader = tf.train.import_meta_graph('./model/main.meta')
    ckpt = tf.train.latest_checkpoint('./model/')
    loader.restore(f, ckpt)

    ans = f.run(C, {A: 2.0, B: 5.0})

    print(ans)
