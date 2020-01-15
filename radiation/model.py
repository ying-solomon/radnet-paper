import tensorflow as tf

BN_EPSILON = 0.001


def conv2d(x, W, b, strides=1, padding='SAME'):
    """ Wrapper for using the applying a 2d conv with the tensorflow library
     and then add a bias.
    """
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
#    x = tf.nn.bias_add(x, b)
    return x


def pool2d(x, k=2, l=2):
    """ Wrapper for a max pool filter
    """
    return tf.nn.max_pool(x, ksize=[1, k, l, 1], strides=[1, k, l, 1], padding='SAME')


def ReLU(x, alpha):
    """ Wrapper for using a ReLU activation function
    TODO: Implement an option for choosing between relu, leaky relu and prelu.
    """
    # return tf.nn.relu(x)
    # return leakyReLU(x, 0.001)
    # return parametricReLU(x, alpha)
    return tf.nn.elu(x)
    # return tf.tanh(x)


def weightInitilization5(a, b, c, d, wstddev):
    """ Inits the weights for the convolutional layers. The initialization is using a
    normal distribution with a given standard deviation.

    :param a: height of filter
    :param b: width of filter
    :param c: in size
    :param d: out size
    :param wstddev: standard deviation
    :return:
    """
    return tf.get_variable("weight", shape=[a, b, c, d],
                           initializer=tf.contrib.layers.xavier_initializer_conv2d())
    # return tf.Variable(tf.random_normal([a, b, c, d], stddev=wstddev))


def weightInitilization3(a, b, wstddev):
    """ Inits the wights for the fully connected layers.
    The the initialization is using xavier initialization that improves the convergence
    considerably.

    :param a: In size
    :param b: Out size
    :return:
    """
    # xavier initialization improves the starting of the training
    # http://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    return tf.get_variable("weight", shape=[a, b],
                           initializer=tf.contrib.layers.xavier_initializer())

    # earlier solution with random initialization.
    # return tf.Variable(tf.random_normal([a, b], stddev=wstddev))


def biasInitialization(a, bstddev):
    """ Initialization of the bias
    In the lecture 5 slide 38 set b to small value i.e. 0.1 of Jim's slides.

    :param a:
    :param bstddev:
    :return: x
    """
    return tf.Variable(tf.random_normal([a], stddev=bstddev, mean=0.1))
    # return tf.Variable(tf.zeros([a]))


def parametricReLU(x, alpha):
    """ PReLU
    Ref.: http://stackoverflow.com/questions/39975676/how-to-implement-prelu-activation-in-tensorflow
    Ref.: https://github.com/tflearn/tflearn/blob/4ba8c8d78bf1bbdfc595bf547bad30580cb4c20b/tflearn/activations.py#L191

    :param x:
    :param alpha: trainable variable for the negative values
    :return:
    """
    pos = tf.nn.relu(x)
    neg = alpha * (x - tf.abs(x)) * 0.5

    return pos + neg


def leakyReLU(x, alpha=0., max_value=None):
    """ leakyReLU

    # Ref.: https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/V6aeBw4nlaE

    :param alpha: slope of negative section. Hyperparameter
    :param max_value: saturation threshold.
    :return: x
    """

    if alpha != 0.:
        negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        max_value = tf.cast(max_value, x.dtype.base_dtype)
        zero = tf.cast(0., x.dtype.base_dtype)
        x = tf.clip_by_value(x, zero, max_value)
    if alpha != 0.:
        x -= alpha * negative_part
    return x


def bnInitialization(n_out):
    """ Batch normalization initialization of the variables of the layer.

    :param n_out:
    :return:
    """
    current = dict()
    with tf.variable_scope('bn'):
        current['beta'] = tf.Variable(
            tf.constant(0.0, shape=[n_out]), name='beta')

        current['gamma'] = tf.Variable(
            tf.constant(1.0, shape=[n_out]), name='gamma')

        current['mean'] = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                      trainable=False)
        current['var'] = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                     trainable=False)

        return current


def preluInitialization(n_out):
    """ Initialization of Alpha for the PReLU

    :param n_out:
    :return:
    """
    return tf.Variable(tf.constant(0.0, shape=[n_out]), name='alpha', dtype=tf.float32)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
    #tf.summary.scalar('max', tf.reduce_max(var))
    #tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def conv_summaries(self, conv, conv_activation, conv_name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(conv_name):
        with tf.name_scope('layer'):
            variable_summaries(conv)
        with tf.name_scope('activation'):
            variable_summaries(conv_activation)
 #       with tf.name_scope('normalization'):
 #           variable_summaries(conv_normalization)
        with tf.name_scope('weights'):
            variable_summaries(self.vars[conv_name]['w'])
        with tf.name_scope('bias'):
            variable_summaries(self.vars[conv_name]['b'])


def batchNorm_old(x, axes, vars, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Ref.: https://gist.github.com/tomokishii/0ce3bdac1588b5cca9fa5fbdf6e1c412
    Ref.: http://r2rt.com/implementing-batch-normalization-in-tensorflow.html

    :param x: Tensor, 2d input maps
    :param n_out: integer, depth of input maps
    :param phase_train: boolean tf.Varialbe, true indicates training phase
    :param scope: string, variable scope

    :return normed: batch-normalized maps
    """
    mean, var = tf.nn.moments(x, axes, name='moments')

    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    with tf.name_scope('bn'):

        def mean_var_with_update():
            ema_apply_op = ema.apply([vars['mean'], vars['var']])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(vars['mean']), tf.identity(vars['var'])

        mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (
            ema.average(vars['mean']), ema.average(vars['var'])))

        normed = tf.nn.batch_normalization(
            x, mean, var, vars['beta'], vars['gamma'], 1e-3)
    return normed


def batchNorm(input_layer, axes, vars, phase_train):
    '''
    Helper function to do batch normalziation
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :return: the 4D tensor after being normalized
    '''

    mean, variance = tf.nn.moments(input_layer, axes=axes)
    beta = vars['beta']
    gamma = vars['gamma']
    bn_layer = tf.nn.batch_normalization(
        input_layer, mean, variance, beta, gamma, BN_EPSILON)

    return bn_layer


c0_size = 128
c1_size = 256
c2_size = 512
c3_size = 1024
c34_size = 1024
c4_size = 1024

fc1_size = 512
fc2_size = 256
#out_size = 60

pre_size = 1024
#fc1_size = 2048
#fc2_size = 256
out_size = 60
weight_stddev = 1.5
bias_stddev = 1


class RadNetModel(object):
    '''Implements the Radiation model for Climate Science

    TODO: Usage...

    '''

    def __init__(self):
        ''' Initializes the RadNet Model. '''
        self.vars = self._create_variables()
        self.phase_train = tf.placeholder(tf.bool, name="train_bool_node")

    def train_phase(self):
        return self.phase_train

    def _create_variables(self):
        var = dict()

        with tf.variable_scope('radnet'):
            with tf.variable_scope('conv0'):
                current = dict()
                current['w'] = weightInitilization5(
                    6, 3, 1, c0_size, weight_stddev)
                current['b'] = biasInitialization(c0_size, bias_stddev)
                current['bn'] = bnInitialization(c0_size)
                current['pr'] = preluInitialization(c0_size)
                var['conv0'] = current
            with tf.variable_scope('conv00'):
                current = dict()
                current['w'] = weightInitilization5(
                    3, 3, c0_size, c1_size, weight_stddev)
                current['b'] = biasInitialization(c1_size, bias_stddev)
                current['bn'] = bnInitialization(c1_size)
                current['pr'] = preluInitialization(c1_size)
                var['conv00'] = current
            with tf.variable_scope('conv1'):
                current = dict()
                current['w'] = weightInitilization5(
                    3, 3, c1_size, c2_size, weight_stddev)
                current['b'] = biasInitialization(c1_size, bias_stddev)
                current['bn'] = bnInitialization(c1_size)
                current['pr'] = preluInitialization(c1_size)
                var['conv1'] = current
            with tf.variable_scope('conv11'):
                current = dict()
                current['w'] = weightInitilization5(
                    3, 3, c1_size, c1_size, weight_stddev)
                current['b'] = biasInitialization(c1_size, bias_stddev)
                current['bn'] = bnInitialization(c1_size)
                current['pr'] = preluInitialization(c1_size)
                var['conv11'] = current
            with tf.variable_scope('conv2'):
                current = dict()
                current['w'] = weightInitilization5(
                    3, 3, c1_size, c2_size, weight_stddev)
                current['b'] = biasInitialization(c2_size, bias_stddev)
                current['bn'] = bnInitialization(c2_size)
                current['pr'] = preluInitialization(c2_size)
                var['conv2'] = current
            with tf.variable_scope('conv22'):
                current = dict()
                current['w'] = weightInitilization5(
                    3, 3, c2_size, c2_size, weight_stddev)
                current['b'] = biasInitialization(c2_size, bias_stddev)
                current['bn'] = bnInitialization(c2_size)
                current['pr'] = preluInitialization(c2_size)
                var['conv22'] = current
            with tf.variable_scope('conv3'):
                current = dict()
                current['w'] = weightInitilization5(
                    3, 3, c2_size, c3_size, weight_stddev)
                current['b'] = biasInitialization(c3_size, bias_stddev)
                current['bn'] = bnInitialization(c3_size)
                current['pr'] = preluInitialization(c3_size)
                var['conv3'] = current
            with tf.variable_scope('conv33'):
                current = dict()
                current['w'] = weightInitilization5(
                    3, 3, c3_size, c3_size, weight_stddev)
                current['b'] = biasInitialization(c3_size, bias_stddev)
                current['bn'] = bnInitialization(c3_size)
                current['pr'] = preluInitialization(c3_size)
                var['conv33'] = current

            with tf.variable_scope('conv4'):
                current = dict()
                current['w'] = weightInitilization5(
                    3, 3, c3_size, c3_size, weight_stddev)
                current['b'] = biasInitialization(c3_size, bias_stddev)
                current['bn'] = bnInitialization(c3_size)
                current['pr'] = preluInitialization(c3_size)
                var['conv4'] = current

            with tf.variable_scope('conv44'):
                current = dict()
                current['w'] = weightInitilization5(
                    3, 3, c3_size, c3_size, weight_stddev)
                current['b'] = biasInitialization(c3_size, bias_stddev)
                current['bn'] = bnInitialization(c3_size)
                current['pr'] = preluInitialization(c3_size)
                var['conv44'] = current

            with tf.variable_scope('pre'):
                current = dict()
                current['w'] = weightInitilization3(
                    183, pre_size, weight_stddev)
                current['b'] = biasInitialization(pre_size, bias_stddev)
                current['bn'] = bnInitialization(pre_size)
                current['pr'] = preluInitialization(pre_size)
                var['pre'] = current

            with tf.variable_scope('fc1'):
                current = dict()
                #current['w'] = weightInitilization3( 4 * fc2_size , fc1_size, weight_stddev)
                #current['w'] = weightInitilization3(2 * 2 * c3_size, fc1_size, weight_stddev)
                current['w'] = weightInitilization3(
                    1024, fc1_size, weight_stddev)
                current['b'] = biasInitialization(fc1_size, bias_stddev)
                current['bn'] = bnInitialization(fc1_size)
                current['pr'] = preluInitialization(fc1_size)
                var['fc1'] = current
            with tf.variable_scope('fc2'):
                current = dict()
                current['w'] = weightInitilization3(
                    fc1_size, fc2_size, weight_stddev)
                current['b'] = biasInitialization(fc2_size, bias_stddev)
                current['bn'] = bnInitialization(fc2_size)
                current['pr'] = preluInitialization(fc2_size)
                var['fc2'] = current
            with tf.variable_scope('out'):
                current = dict()
                current['w'] = weightInitilization3(
                    fc1_size, out_size, weight_stddev)
                current['b'] = biasInitialization(out_size, bias_stddev)
                var['out'] = current

        return var

    def _create_network(self, input_batch):
        ''' Construct the network.'''

        print("input shape: ")
        print(input_batch.get_shape())
        # Pre-process the input
        # x is 64 x 1 tensor with padding at the end
        input_batch = tf.reshape(
            input_batch, shape=[-1, 60, 3, 1], name="input_node")

        with tf.name_scope('conv0'):
            # 1x1 conv layer https://www.quora.com/What-is-a-1X1-convolution
            conv = conv2d(input_batch, self.vars['conv0']['w'],
                          self.vars['conv0']['b'], strides=2, padding="SAME")
            conv_pool = pool2d(conv, k=2, l=2)
            conv_activation = ReLU(conv_pool, self.vars['conv0']['pr'])
            # conv_normalization = batchNorm(
            #    conv_activation, [0, 1, 2], self.vars['conv0']['bn'], self.phase_train)
            conv_summaries(self, conv, conv_activation,
                           'conv0')

        with tf.name_scope('conv00'):
            # 1x1 conv layer https://www.quora.com/What-is-a-1X1-convolution
            conv = conv2d(
                conv_activation, self.vars['conv00']['w'], self.vars['conv00']['b'], strides=2)
            conv_pool = pool2d(conv, k=2, l=2)
            conv_activation = ReLU(conv_pool, self.vars['conv00']['pr'])
            # conv_normalization = batchNorm(
            #    conv_activation, [0, 1, 2], self.vars['conv00']['bn'], self.phase_train)
            conv_summaries(self, conv, conv_activation,
                           'conv00')

        with tf.name_scope('conv1'):

            conv = conv2d(
                conv_activation, self.vars['conv1']['w'], self.vars['conv1']['b'], strides=2)
            conv_activation = ReLU(conv, self.vars['conv1']['pr'])
            #conv_normalization = batchNorm(conv_activation, [0, 1, 2], self.vars['conv1']['bn'], self.phase_train)
            conv_summaries(self, conv, conv_activation,
                           'conv1')

        '''
            with tf.name_scope('conv11'):
                conv = conv2d(conv_normalization, self.vars['conv11']['w'], self.vars['conv11']['b'], strides=1)
                conv_pool = pool2d(conv, k=2, l=2)
                conv_activation = ReLU(conv_pool, self.vars['conv11']['pr'])
                conv_normalization = batchNorm(conv_activation, [0, 1, 2], self.vars['conv11']['bn'], self.phase_train)
                conv_summaries(self, conv, conv_activation, conv_normalization, 'conv11')
                print(conv_normalization.shape)

        with tf.name_scope('conv2'):
            conv = conv2d(
                conv_normalization, self.vars['conv2']['w'], self.vars['conv2']['b'], strides=1)
            conv_activation = ReLU(conv, self.vars['conv2']['pr'])
            conv_normalization = batchNorm(
                conv_activation, [0, 1, 2], self.vars['conv2']['bn'], self.phase_train)
            conv_summaries(self, conv, conv_activation,
                           conv_normalization, 'conv2')
            print(conv_normalization.shape)
            
	    with tf.name_scope('conv22'):
                conv = conv2d(conv_normalization, self.vars['conv22']['w'], self.vars['conv22']['b'], strides=1)
                conv_pool = pool2d(conv, k=2, l=2)
                conv_activation = ReLU(conv_pool, self.vars['conv22']['pr'])
                conv_normalization = batchNorm(conv_activation, [0, 1, 2], self.vars['conv22']['bn'], self.phase_train)
                conv_summaries(self, conv, conv_activation, conv_normalization, 'conv22')
                print(conv_normalization.shape)


        with tf.name_scope('conv3'):
            conv = conv2d(
                conv_normalization, self.vars['conv3']['w'], self.vars['conv3']['b'], strides=1)
            conv_pool = pool2d(conv, k=2, l=2)
            conv_activation = ReLU(conv_pool, self.vars['conv3']['pr'])
            conv_normalization = batchNorm(
                conv_activation, [0, 1, 2], self.vars['conv3']['bn'], self.phase_train)
            conv_summaries(self, conv, conv_activation,
                           conv_normalization, 'conv3')
            print(conv_normalization.shape)

            with tf.name_scope('conv33'):
                conv = conv2d(conv_normalization, self.vars['conv33']['w'], self.vars['conv33']['b'], strides=1)
                conv_pool = pool2d(conv, k=2, l=2)
                conv_activation = ReLU(conv_pool, self.vars['conv33']['pr'])
                conv_normalization = batchNorm(conv_activation, [0, 1, 2], self.vars['conv33']['bn'], self.phase_train)
                conv_summaries(self, conv, conv_activation, conv_normalization, 'conv33')

            with tf.name_scope('conv4'):
                conv = conv2d(conv_normalization, self.vars['conv4']['w'], self.vars['conv4']['b'], strides=1)
                conv_activation = ReLU(conv, self.vars['conv4']['pr'])
                conv_normalization = batchNorm(conv_activation, [0, 1, 2], self.vars['conv4']['bn'], self.phase_train)
                conv_summaries(self, conv, conv_activation, conv_normalization, 'conv4')

            with tf.name_scope('conv44'):
                conv = conv2d(conv_normalization, self.vars['conv44']['w'], self.vars['conv44']['b'], strides=1)
                conv_pool = pool2d(conv, k=2, l=2)
                conv_activation = ReLU(conv_pool, self.vars['conv44']['pr'])
                conv_normalization = batchNorm(conv_activation, [0, 1, 2], self.vars['conv44']['bn'], self.phase_train)
                conv_summaries(self, conv, conv_activation, conv_normalization, 'conv44')

        '''

        with tf.name_scope('fc1'):
            # Reshape conv3 output to fit fully connected layer input
            fc1 = tf.reshape(
                conv_activation, [-1, self.vars['fc1']['w'].get_shape().as_list()[0]])
            #fc1 = tf.reshape(conv_normalization, [-1, self.vars['fc1']['w'].get_shape().as_list()[0]])
            fc1 = tf.add(
                tf.matmul(fc1, self.vars['fc1']['w']), self.vars['fc1']['b'])
            fc1 = ReLU(fc1, self.vars['fc1']['pr'])
            #fc1 = batchNorm(fc1, [0], self.vars['fc1']['bn'], self.phase_train)
            variable_summaries(self.vars['fc1']['w'])

            print(fc1.get_shape())

        '''
        with tf.name_scope('fc2'):
            fc2 = tf.add(
                tf.matmul(fc1, self.vars['fc2']['w']), self.vars['fc2']['b'])
            fc2 = ReLU(fc2, self.vars['fc2']['pr'])
            fc2 = batchNorm(fc2, [0], self.vars['fc2']['bn'], self.phase_train)
            variable_summaries(self.vars['fc2']['w'])
            print(fc2.get_shape())
	'''
        with tf.name_scope('out'):
            out = tf.add(tf.matmul(
                fc1, self.vars['out']['w']), self.vars['out']['b'], name="output_node")
            variable_summaries(self.vars['out']['w'])
            print(out.get_shape())

        return out

    def loss(self, input_batch, real_output):
        ''' Creates a RadNet network and returns the autoencoding loss.
            The variables are all scoped to the given name.

            TODO: Implement an option to choose between mse loss and huber loss.
        '''
        with tf.name_scope('radnet'):
            output = self._create_network(input_batch)
            with tf.name_scope('loss'):
                # Huber loss
                loss = tf.reduce_mean(self.huber_loss(real_output, output))

                tf.summary.scalar('loss', loss)

                return loss

    def predict(self, input, real_output, id_file):
        """ Function for calculating prediction without backpropagating the error

        :param input:
        :param real_output:
        :param id_file:
        :return:
        """

        with tf.name_scope('radnet'):
            pred_output = self._create_network(input)

            # the loss here is mse because this call is made online for testing, not for training
            mse = tf.reduce_mean(
                tf.squared_difference(pred_output, real_output))
            mape = tf.multiply(100.0, tf.reduce_mean(
                tf.abs(tf.divide((real_output - pred_output), real_output))))

            return id_file, real_output, pred_output, mse, mape, input

    def huber_loss(self, y_true, y_pred, max_grad=1.):

        return tf.losses.huber_loss(y_true, y_pred, delta=1.)
