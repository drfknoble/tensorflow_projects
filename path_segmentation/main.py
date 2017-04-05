'''path_segmentation'''

#%%

# pylint: disable=E0401
# pylint: disable=C0103
# pylint: disable=W0621

import numpy as np
import scipy.io
import tensorflow as tf

data = scipy.io.loadmat('E:/users/fknoble/FCN/Model_zoo/imagenet-vgg-verydeep-19')
weights = np.squeeze(data['layers'])

init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

with tf.Session() as f:

    f.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        i = 0

        while not coord.should_stop():

            kernel, bias = weights[i][0][0][0][0]

            print(np.shape(kernel))

            i += 1

            break

    except tf.errors.OutOfRangeError:
        print('EoF')
    finally:
        coord.request_stop()

    coord.join(threads)
