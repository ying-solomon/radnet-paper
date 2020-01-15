import fnmatch
import os
import random
import re
import threading
import json

import tensorflow as tf
from netCDF4 import Dataset
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.interpolate import splev, splrep

"""
Data v7: statistical values
ST:  min 100.0000000000 max 333.1499946801 mean 268.1406929063 std 37.5368706325
C:  min 0.0000000000 max 0.0099999904 mean 0.0017284657 std 0.0023850203
R:  min -255.9600440474 max 78.3198382662 mean -1.3758656541 std 6.1112494507
T:  min 100.0000000000 max 355.5721906214 mean 230.1788102309 Td 46.5063403685
H:  min -2720.3344538111 max 1848.3667831706 mean 4.2050377031 Hd 13.4852605066
"""

#
#minT = 178.87485
minT = 150
#maxT = 310.52261
maxT = 340

minC = 0
maxC = 0.01

#minH = 6.324828e-08
minH = 0
#maxH = 0.021951281
maxH = 0.1

minR = -59.08844
maxR = 14.877947

minP = 0.0
maxP = 103000


epoch = 0

lock = threading.Lock()

standard_x = np.linspace(1, 10000, 10)
standard_x = np.append(standard_x, np.linspace(11000, 80000, 25))
standard_x = np.append(standard_x, np.linspace(82760, 103000, 25))
standard_x = standard_x.tolist()

level_size = len(standard_x)

def cal_air_pressure(air_pressure_interface):
    air_pressure = np.empty(60)
    for level in range(len(air_pressure_interface) - 1):
        air_pressure[level] = (air_pressure_interface[level] + air_pressure_interface[level+1])*0.5
    return air_pressure

def normalizeT(t):
    return normalize(t, minT, maxT)


def normalizeH(h):
    return normalize(h, minH, maxH)


def normalizeC(c):
    return normalize(c, minC, maxC)


def normalizeR(r):
    return normalize(r, minR, maxR)


def normalizeP(p):
    return normalize(p, minP, maxP)


def normalize(x, min, max, mean=1, std=1):
    # TODO: add option to choose between min-max of zero-mean normalization
    return (x - min) / (max - min)  # min max normalization
    # return (x - mean) / std  # standardization - zero-mean normalization
    # return x+100


def denormalize(x, mean, std):
    return x * std + mean


def get_category_cardinality(files):
    """ Deprecated: function used before for identifying the samples based in its name
    and calculate the minimum and maximum sample id
    :param files: array of root paths.
    :return: min_id, max_id: int
    """
    file_pattern = r'([0-9]+)\.csv'
    id_reg_expression = re.compile(file_pattern)
    min_id = None
    max_id = None
    for filename in files:
        id = int(id_reg_expression.findall(filename)[0])
        if min_id is None or id < min_id:
            min_id = id
        if max_id is None or id > max_id:
            max_id = id

    return min_id, max_id


def randomize_files(files):
    """ Function that randomizes a list of filePaths.

    :param files: list of path files
    :return: iterable of random files
    """
    for file in files:
        file_index = random.randint(0, (len(files) - 1))
        yield files[file_index]


def find_files(directory, pattern='*.csv'):
    """ Recursively finds all files matching the pattern.

    :param directory:  directory path
    :param pattern: reggex
    :return: list of files
    """

    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

def interpolate(x, y, standard_x, t_or_h='r'):
    x = np.flip(x,0)
    y = np.flip(y,0)


    level_size = len(standard_x)
    spl = splrep(x, y)
    for start_index, start in enumerate(standard_x):
        if x[0] <= start:
            start_index = start_index - 1
            break

    for end_index, end in enumerate(standard_x):
        if x[-1] <= end:
            end_index = end_index + 1
            break
    #valid intervals [start_index, end_index)


    interpolated = splev(standard_x[start_index:end_index], spl)
    interpolated[0] = interpolated[1]
    interpolated[-1] = interpolated[-2]

    if t_or_h == 't':
        standard_y = np.append(np.zeros(start_index), normalizeT(interpolated))
        standard_y = np.append(standard_y, np.zeros(level_size - end_index))
    elif t_or_h == 'h':
        standard_y = np.append(np.zeros(start_index), normalizeH(interpolated))
        standard_y = np.append(standard_y, np.zeros(level_size - end_index))
    else:
        standard_y = np.append(np.zeros(start_index), interpolated)
        standard_y = np.append(standard_y, np.zeros(level_size - end_index))

    assert len(standard_y) == level_size

    return standard_y


def load_data_samples(files):
    """ Generator that yields samples from the directory.

    In the latest versions, the files are files where each line is a sample
    in json format. This function basically read each sample of the file and
    normalizes it and generates the data for the model in the desired format.

    :param files: list of files
    :return: iterable that contains the data, the label and the identifier of the sample.
    """


    for filename in files:

        with lock:
            f = Dataset(filename, mode='r')
            v = f.variables['radiation_data'][:]
            f.close()
        #ids = np.random.choice(np.arange(v.shape[0]), size = 10, replace=False)
        for id in range(v.shape[0]):
            #data = np.append(v[id,0:122],v[id, 182:243])
            #data = np.append(v[id,0:2],normalizeT(v[id, 2:62]))
            #data = np.append(data, normalizeH(v[id,62:122]))
            #data = np.append(data, normalizeP(v[id,182:243]))
            '''
            data = []

            for i in range(60):
                data.append(normalizeC(v[id, 0]))
                data.append(normalizeT(v[id, 1]))
                data.append(normalizeT(v[id, i+2]))
                data.append(normalizeH(v[id, i + 62]))
                data.append(normalizeP(v[id, i + 182]))
            '''
            #    data = np.append(data, normalizeC(v[id, 0]))
            #    data = np.append(data, normalizeT(v[id, 1]))
            #    data = np.append(data, normalizeT(v[id, i+2]))
            #    data = np.append(data, normalizeH(v[id, i + 62]))
            #    data = np.append(data, normalizeP(v[id, i + 182]))

            air_pressure = cal_air_pressure(v[id, 182:243])
            inter_air_temperature = interpolate(air_pressure, v[id, 2:62], standard_x, 't')
            inter_humidity = interpolate(air_pressure, v[id, 62:122], standard_x, 'h')
            inter_radiation = interpolate(air_pressure, v[id, 122:182], standard_x)

            data = np.append(normalizeC(v[id, 0]), normalizeT(v[id, 1]))
            data = np.append(data, np.zeros(level_size - 2))
            data = np.append(data, inter_air_temperature)
            data = np.append(data, inter_humidity)
            #data = np.append(data, normalizeP(v[id, 182:243]))

            label = np.array(inter_radiation)

            '''
            data.append(v[id,0])
            data.append(v[id,1])


            num_levels = int((v.shape[1]-2)/3)

            for i in range(num_levels):
                data.append(v[id,i+2])
                data.append(v[id,i+98])
                label.append(v[id,i+194])

            for _ in range(0, 196 - 194):
                data.append(np.float32(0.0))

            '''
            if np.isnan(data.sum()) or np.isnan(label.sum()):
                print("NaN found!!!!!")
                continue

            yield data, label, [id]


class FileReader(object):
    """ Background reader that pre-processes radiation files
    and enqueues them into a TensorFlow queue.

    """

    def __init__(self,
                 data_dir,
                 test_dir,
                 coord,
                 n_input=180,
                 n_output=60,
                 queue_size=5000000,
                 test_percentage=0.2):

        # TODO: Implement a option that enables the usage of a test queue, by default it is
        # enabled here. For implementing this, the flag should be propagated to the several
        # functions that operate with both queues.

        self.data_dir = data_dir
        self.test_dir = test_dir
        self.coord = coord
        self.n_input = n_input
        self.n_output = n_output
        self.threads = []
        self.sample_placeholder_train = tf.placeholder(tf.float32, [n_input])
        self.result_placeholder_train = tf.placeholder(tf.float32, [n_output])
        self.sample_placeholder_test = tf.placeholder(tf.float32, [n_input])
        self.result_placeholder_test = tf.placeholder(tf.float32, [n_output])
        self.idFile_placeholder_test = tf.placeholder(tf.int32, [1])
        self.idFile_placeholder_train = tf.placeholder(tf.int32, [1])

        self.queue_train = tf.PaddingFIFOQueue(queue_size, [tf.float32, tf.float32, tf.int32],
                                               shapes=[[n_input], [n_output], [1]])
        self.queue_test = tf.PaddingFIFOQueue(queue_size, [tf.float32, tf.float32, tf.int32],
                                              shapes=[[n_input], [n_output], [1]])
        self.enqueue_train = self.queue_train.enqueue(
            [self.sample_placeholder_train, self.result_placeholder_train, self.idFile_placeholder_train])
        self.enqueue_test = self.queue_test.enqueue(
            [self.sample_placeholder_test, self.result_placeholder_test, self.idFile_placeholder_test])

        # https://github.com/tensorflow/tensorflow/issues/2514
        # https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/rmGu1HAyPw4
        # Use of a flag that changes the input queue to another one, this way the model can
        # be tested using the test queue when required.
        self.select_q = tf.placeholder(tf.int32, [])
        self.queue = tf.QueueBase.from_list(
            self.select_q, [self.queue_train, self.queue_test])

        # Find any file as the reggex is *
        self.files = find_files(data_dir, '*')
        if not self.files:
            raise ValueError("No data files found in '{}'.".format(data_dir))

        print("training files length: {}".format(len(self.files)))

        self.test_files = find_files(test_dir, '*')
        if not self.test_files:
            raise ValueError(
                "No test data files found in '{}'.".format(test_dir))

        print("test files length: {}".format(len(self.test_files)))

        # Split the data into test and train datasets
        # range = int(len(self.files) * test_percentage)
        self.test_dataset = self.test_files
        self.train_dataset = self.files

    def dequeue(self, num_elements):
        """ Function for dequeueing a mini-batch

        :param num_elements: int size of minibatch
        :return:
        """
        data, label, id = self.queue.dequeue_many(num_elements)

        return data, label, id

    def queue_switch(self):
        return self.select_q

    def thread_main(self, sess, id, n_thread, test):
        """ Thread function to be launched as many times as required for loading the data
        from several files into the Tensorflow's queue.

        :param sess: Tensorflow's session
        :param id: thread ID
        :param test: bool for choosing between the queue to feed the data, True for test queue
        :return: void
        """
        global epoch
        stop = False
        # Go through the dataset multiple times
        if test:
            files = self.test_dataset
        else:
            files = self.train_dataset

        # while tensorflows coordinator doesn't want to stop, continue.
        while not stop:

            epoch += 1
            if not test:
                print("Number of epochs: {}".format(epoch))
            randomized_files = randomize_files(files)

            '''
            file_partitions = []
            for index, i in enumerate(files):
                if (index)%(n_thread-1)+1 == id:
                    file_partitions.append(i)
            randomized_files = randomize_files(file_partitions)
            '''
            iterator = load_data_samples(randomized_files)

            for data, label, id_file in iterator:
                # update coordinator's state
                if self.coord.should_stop():
                    stop = True
                    break

                if test:  # in train range and test thread
                    sess.run(self.enqueue_test,
                             feed_dict={self.sample_placeholder_test: data,
                                        self.result_placeholder_test: label,
                                        self.idFile_placeholder_test: id_file})
                else:  # below the rage -> train
                    sess.run(self.enqueue_train,
                             feed_dict={self.sample_placeholder_train: data,
                                        self.result_placeholder_train: label,
                                        self.idFile_placeholder_train: id_file})

    def start_threads(self, sess, n_threads=2):
        """ Reader threads' launcher, uses the first thread for feeding into the test queue
        and the rest for feeding into the train queue.

        :param sess:
        :param n_threads:
        :return: void
        """
        for id in range(n_threads):
            if id == 0:
                thread = threading.Thread(
                    target=self.thread_main, args=(sess, id, n_threads, True))
            else:
                thread = threading.Thread(
                    target=self.thread_main, args=(sess, id, n_threads, False))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)

        return self.threads

    # not used anymore
    def decompose_data(self, data):

        levels = 96
        CO2 = data[0]
        surface_temperature = data[1]
        air_temperature = []
        humidity = []
        for i in range(2, levels * 2 + 2):
            if (i % 2) == 0:  # even
                air_temperature.append(denormalize(data[i], meanT, stdT))
            else:
                humidity.append(denormalize(data[i], meanH, stdH))

        input_dic = {
            "surface_temperature": surface_temperature,
            "co2": CO2,
            "air_temperature": air_temperature,
            "humidity": humidity
        }

        return input_dic
