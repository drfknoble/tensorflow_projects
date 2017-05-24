'''utilities'''

#%%

# pylint: disable=E0401
# pylint: disable=C0103
# pylint: disable=W0621

import os
import tensorflow as tf

def generate_file_list(directory, file_type, file_list):
    '''parse_folder(). Parses a directory for a specific file type
    and writes a list of the files found.'''

    F = open(file_list, 'w')

    lst = os.listdir(directory)
    lst = lst.sort

    for file in sorted(os.listdir(directory)):
        if file.endswith(file_type):
            F.write(directory + '/' + file + '\n')

    F.close()

def parse_file_list(file_list, num_epochs=1):
    '''generate_records(). Generate a tfrecord of examples from a
    list of file names.'''

    filename_queue = tf.train.string_input_producer(
        file_list,
        num_epochs=num_epochs,
        shuffle=False
    )

    reader = tf.TextLineReader()
    _, value = reader.read(filename_queue)

    print(value)

    return value
