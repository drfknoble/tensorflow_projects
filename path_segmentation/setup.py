'''path_segmentation'''

#%%

# pylint: disable=E0401
# pylint: disable=C0103

# Imports.
import tensorflow as tf
import utilities as utils

# Here, we define important directories.
default_dir = 'E:/users/fknoble/My_FCN/'

logs = './logs/'
model = './model/'
input_dir = './data/input/'  # './data/test/ to learn the identity kernel
output_dir = './data/output/'
training_features_dir = input_dir + 'features/training/'
training_labels_dir = input_dir + 'labels/training/'
validation_features_dir = input_dir + 'features/validation/'
validation_labels_dir = input_dir + 'labels/validation/'

# Here, we define important file names.
training_features_txt_file = input_dir + 'training_features.txt'
training_labels_txt_file = input_dir + 'training_labels.txt'
validation_features_txt_file = input_dir + 'validation_features.txt'
validation_labels_txt_file = input_dir + 'validation_labels.txt'

training_record_file = default_dir + 'data/training_record.tfrecords'
validation_record_file = default_dir + 'data/validation_record.tfrecords'

# Here, we populate the feature_file and label_file with name of feature
# and label images.
utils.populate_file(training_features_txt_file, training_features_dir)
utils.populate_file(training_labels_txt_file, training_labels_dir)
utils.populate_file(validation_features_txt_file, validation_features_dir)
utils.populate_file(validation_labels_txt_file, validation_labels_dir)

training_features_list = utils.parse_file(training_features_txt_file)
training_labels_list = utils.parse_file(training_labels_txt_file)
validation_features_list = utils.parse_file(validation_features_txt_file)
validation_labels_list = utils.parse_file(validation_labels_txt_file)

# Here, we create handles for reading and writing TFRecord files.
training_features = utils.output_pipeline(training_features_list, 1)
training_labels = utils.output_pipeline(training_labels_list, 1)
validation_features = utils.output_pipeline(validation_features_list, 1)
validation_labels = utils.output_pipeline(validation_labels_list, 1)

# Initialisation commands
init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

# In this session, we read our raw data and create a TFRecord file.
with tf.Session() as s:

    s.run(init)

    training_writer = tf.python_io.TFRecordWriter(training_record_file)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    print('Starting...')

    try:

        i = 0

        while not coord.should_stop():

            single_training_feature = s.run(training_features)
            single_training_label = s.run(training_labels)

            training_example = utils.make_example(
                single_training_feature, single_training_label)

            training_writer.write(training_example.SerializeToString())

            print(i)

            i += 1

    except tf.errors.OutOfRangeError:
        print('Done!')
    finally:
        coord.request_stop()

    coord.join(threads)

    training_writer.close()

    validation_writer = tf.python_io.TFRecordWriter(validation_record_file)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    print('Starting...')

    try:

        i = 0

        while not coord.should_stop():

            single_validation_feature = s.run(validation_features)
            single_validation_label = s.run(validation_labels)

            validation_example = utils.make_example(
                single_validation_feature, single_validation_label)

            validation_writer.write(validation_example.SerializeToString())

            print(i)

            i += 1

    except tf.errors.OutOfRangeError:
        print('Done!')
    finally:
        coord.request_stop()

    coord.join(threads)

    validation_writer.close()
