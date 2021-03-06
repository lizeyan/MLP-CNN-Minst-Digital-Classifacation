import theano.tensor as T
import numpy as np
from utils import sharedX
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.pool import pool_2d


class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable

    def forward(self, inputs):
        pass

    def params(self):
        pass


class Relu(Layer):
    def __init__(self, name):
        super(Relu, self).__init__(name)

    def forward(self, inputs):
        # Your codes here
        return T.nnet.relu(inputs)


class Sigmoid(Layer):
    def __init__(self, name):
        super(Sigmoid, self).__init__(name)

    def forward(self, inputs):
        # Your codes here
        return T.nnet.sigmoid(inputs)


class Softmax(Layer):
    def __init__(self, name):
        super(Softmax, self).__init__(name)

    def forward(self, inputs):
        # Your codes here
        return T.nnet.softmax(inputs)


class Linear(Layer):
    def __init__(self, name, inputs_dim, num_output, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.inputs_dim = inputs_dim;
        self.W = sharedX(np.random.randn(inputs_dim, num_output) * init_std, name=name + '/W')
        self.b = sharedX(np.zeros(num_output), name=name + '/b')

    def forward(self, inputs):
        # Your codes here
        inputs_shape = T.reshape(inputs, [inputs.shape[0], self.inputs_dim])
        return T.dot(inputs_shape, self.W) + self.b

    def params(self):
        return [self.W, self.b]


class Convolution(Layer):
    def __init__(self, name, kernel_size, num_input, num_output, init_std):
        super(Convolution, self).__init__(name, trainable=True)
        # Determine ? in W_shape
        W_shape = (num_output, num_input, kernel_size, kernel_size)
        self.W = sharedX(np.random.randn(*W_shape) * init_std, name=name + '/W')
        self.b = sharedX(np.zeros(num_output), name=name + '/b')

    def forward(self, inputs):
        # Your codes here
        # hint: note how to add bias to a 4-D tensor?
        return conv2d(inputs, self.W) + self.b.dimshuffle('x', 0, 'x', 'x')

    def params(self):
        return [self.W, self.b]


class Pooling(Layer):
    def __init__(self, name, kernel_size):
        super(Pooling, self).__init__(name)
        self.kernel_size = kernel_size

    def forward(self, inputs):
        # Your coders here
        # hint: enable ignore border mode
        return pool_2d(inputs, [self.kernel_size, self.kernel_size], ignore_border=True)
