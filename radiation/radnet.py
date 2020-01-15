# https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc#.dykqbzqek
# https://medium.com/@hamedmp/exporting-trained-tensorflow-models-to-c-the-right-way-cf24b609d183#.2yffldwf7
# https://github.com/tensorflow/tensorflow/issues/616


from tensorflow.core.framework import graph_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
import numpy as np
import scipy.interpolate
from datetime import datetime
import tensorflow as tf
from scipy.interpolate import splev, splrep

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

def normalize(x, min, max, mean=1, std=1):
    # TODO: add option to choose between min-max of zero-mean normalization
    # min max normalization
    # return (x - mean) / std  # standardization - zero-mean normalization
    # return x+100
    
    result = (x - min) / (max - min)
    
    #if type(result) is np.ndarray:
    #    for index, value in enumerate(result):
    #        if value < 0:
    #            result[index] = 0.0
    
    return result

def denormalize(x, min, max, mean=1, std=1):
   
    return x * (max - min) + min

def denormalizeT(t):
    return denormalize(t, minT, maxT)


def denormalizeH(h):
    return denormalize(h, minH, maxH)


def denormalizeC(c):
    return denormalize(c, minC, maxC)


def denormalizeR(r):
    return denormalize(r, minR, maxR)


def denormalizeP(p):
    return denormalize(p, minP, maxP)


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


def interpolate(x, y, standard_x, t_or_h='r'):
    x = np.flip(x,0)
    y = np.flip(y,0)
    
    #for index, value in enumerate(y):
    #    if value < 0:
    #        y[index] = 0

    spl = splrep(x, y)
    for start_index, start in enumerate(standard_x):
        if x[0] < start:
            start_index = start_index - 1
            break
        elif x[0] == start:
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



def check_interpolate(x, y, standard_x):
    x = np.flip(x,0)
    y = np.flip(y,0)

    spl = splrep(x, y)
    for start_index, start in enumerate(standard_x):
        if x[0] < start:
            start_index = start_index - 1
            break
        elif x[0] == start:
            break

    for end_index, end in enumerate(standard_x):
        if x[-1] <= end:
            end_index = end_index + 1
            break

    return start_index, end_index


class RadNet:
    """ Class for loading and fetching the model for getting predictions.

    """

    # TODO: the names of the items of this dict should be changed to match the names of the
    # variables used in the program data calls this class.

    HUMIDITY = "humidity"
    AIR_TEMPERATURE = "air_temperature"
    SURFACE_TEMPERATURE = "surface_temperature"
    CO2 = "CO2"
    PRESSURE = "pressure"


    def __init__(self, frozen_graph_path):
        """ Initializes the needed variables for making prediction

        Loads the graph from a frozen graph in protobbuf format file generated after training the model.
        Then starts a Tensorflow session with that graph and gets the needed variables for
        fetching the model in further calls. The session is then initialized once.

        :param frozen_graph_path: path to the protobuf (.pb) file containing the graph and the value
        of the variables.
        """
        self.input_size = level_size

        with ops.Graph().as_default():
            output_graph_def = graph_pb2.GraphDef()

            with open(frozen_graph_path, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                _ = importer.import_graph_def(output_graph_def, name="")



            # We are initializing the session with the default graph which already has the
            # frozen graph loaded.
            self.sess = session.Session()

            # Important loop for printing the nodes of the graph and debugging
            #for op in self.sess.graph.get_operations():
            #    print(op.name)

            # The input and output nodes are gotten into a variable
            self.input_node = self.sess.graph.get_tensor_by_name("create_model/radnet_1/input_node:0")
            self.output_node = self.sess.graph.get_tensor_by_name("create_model/radnet_1/out/output_node:0")
            # Getting the train_flag_node is also required as it is like that and we need to indicate the graph
            # we are just fetching a parameter.
            #self.train_flag_node = self.sess.graph.get_tensor_by_name("create_model/train_bool_node:0")
    #@profile
    def predict(self, sample, output_size=60, preprocess=False):
        """ Method for fetching the model

        :param sample: array[196] with the correct input of the model
        :return prediction: array[96] with the radiation level for each of the 26 layers
        """

        # Line to be uncommented with the climt integration
        #processed_sample = self.__pre_process(sample)
        if preprocess:
            processed_sample = self.__pre_process(sample)
            prepared = [np.array(processed_sample).reshape((level_size, 3, 1))] * 1
            # prepared = [np.array(sample).reshape((level_size, 4, 1))] * 128
            prediction = self.sess.run(
                self.output_node,
                feed_dict={
                    self.input_node: prepared
                    # self.input_node: [np.array(processed_sample).reshape((level_size, 3, 1))]
                    # self.train_flag_node: False
                })
            prediction = self.__post_process(sample, prediction.squeeze())


        else:
            
            
            sample = self.__pre_process_new(sample)
            prepared = [np.array(sample).reshape((level_size, 4, 1))] * 1
            #prepared = [np.array(sample).reshape((level_size, 4, 1))] * 128
            prediction = self.sess.run(
                    self.output_node,
                    feed_dict={
                         self.input_node: prepared
                         #self.input_node: [np.array(processed_sample).reshape((level_size, 3, 1))]
                         #self.train_flag_node: False
                    })

    #        print((b-a).microseconds)
            # Interpolates de output to the wished value size
            #prediction = self.__interpolate(prediction.squeeze(), output_size)


        return prediction


    def __post_process(self, inputs, prediction):


        #air_pressure = cal_air_pressure(inputs[self.PRESSURE])
        air_pressure = inputs[self.PRESSURE]
        start_index, end_index = check_interpolate(air_pressure, inputs[self.AIR_TEMPERATURE], standard_x)

        spl = splrep(standard_x[start_index:end_index], prediction[start_index:end_index])
        interpolated = splev(np.flip(air_pressure,0), spl)

        return np.flip(interpolated, 0)

    
    def __pre_process_new(self, inputs):
 
        air_pressure = inputs[self.PRESSURE]
        #air_pressure = cal_air_pressure(inputs[self.PRESSURE])
        data = np.append(normalizeC(inputs[self.CO2]), normalizeT(inputs[self.SURFACE_TEMPERATURE]))
        data = np.append(data, np.zeros(level_size - 2))
        data = np.append(data, normalizeT(inputs[self.AIR_TEMPERATURE]))
        data = np.append(data, normalizeH(inputs[self.HUMIDITY]))
        data = np.append(data, normalizeP(air_pressure))
        

        return data

    def __pre_process(self, inputs):
        """ Method that prepares the data for fetching the model.

        :param inputs:
        :return:
        """
        '''
        data = {}

        for key, value in inputs.items():
            if isinstance(value, list):
                value = self.__normalize(
                    value,
                    self.STATISTIC_PARAMS[key]['mean'],
                    self.STATISTIC_PARAMS[key]['std'])

                # Interpolate the data into the x parameters per layer.
                print(key)
                value = self.__interpolate(value.squeeze())

            data[key] = value
        '''
        '''
        # Transform the matrix into the expected input for the model
        data = inputs
        input = []
        input.append(data[self.CO2])
        input.append(data[self.SURFACE_TEMPERATURE])
        for i in range(0, self.input_size):
            input.append(data[self.AIR_TEMPERATURE][i])
            input.append(data[self.HUMIDITY][i])
        for i in range(0, self.input_size+1):
            input.append(data[self.PRESSURE][i])

        # fill last 2 values with 0
        # this numbers can be calculated based in the input_size, but I didn't want to do
        # so for simplicity of the code
        for _ in range(13):
            input.append(0.0)
        

        data = inputs
        input = []
        for i in range(60):
            input.append(normalizeC(data[self.CO2]))
            input.append(normalizeT(data[self.SURFACE_TEMPERATURE]))
            input.append(normalizeT(data[self.AIR_TEMPERATURE][i]))
            input.append(normalizeH(data[self.HUMIDITY][i]))
            input.append(normalizeP(data[self.PRESSURE][i]))
        '''


        #air_pressure = cal_air_pressure(inputs[self.PRESSURE])
        air_pressure = inputs[self.PRESSURE]
        inter_air_temperature = interpolate(air_pressure, inputs[self.AIR_TEMPERATURE], standard_x, 't')
        inter_humidity = interpolate(air_pressure, inputs[self.HUMIDITY], standard_x, 'h')

        data = np.append(normalizeC(inputs[self.CO2]), normalizeT(inputs[self.SURFACE_TEMPERATURE]))
        data = np.append(data, np.zeros(level_size-2))
        data = np.append(data, inter_air_temperature)
        data = np.append(data, inter_humidity)

        '''


        air_pressure = cal_air_pressure(inputs[self.PRESSURE])
        data = np.append(normalizeC(inputs[self.CO2]), normalizeT(inputs[self.SURFACE_TEMPERATURE]))
        data = np.append(data, np.zeros(level_size - 2))
        data = np.append(data, normalizeT(inputs[self.AIR_TEMPERATURE]))
        data = np.append(data, normalizeH(inputs[self.HUMIDITY]))
        data = np.append(data, normalizeP(air_pressure))
        '''

        return data

    @staticmethod
    def __normalize(x, mean, std):
        if isinstance(x, list):
            x = np.array(x)
        if std == 0:
            return x
        # return  (x - min) / (max - min) # min max normalization
        return (x - mean) / std  # standardization - zero-mean normalization
