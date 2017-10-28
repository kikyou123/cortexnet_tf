import os
import tensorflow as tf
from math import ceil
import numpy as np
from op import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class Model01(object):
    def __init__(self, network_size, input_spatial_size):

        self.hidden_layers = len(network_size) - 2
        self.network_size = network_size

        print('\n{:-^80}'.format(' Building model '))
        print('Hidden layers:', self.hidden_layers)
        print('Net sizing:', network_size)
        print('Input spatial size: {} x {}'.format(network_size[0], input_spatial_size))
        self.activation_size = [input_spatial_size]

        for layer in range(0, self.hidden_layers):
            print('{:<80}'.format('Layer ' + str(layer + 1) + ' '))
            print('Bottom size: {} x {}'.format(network_size[layer], self.activation_size[-1]))
            self.activation_size.append(tuple(ceil(s / 2) for s in self.activation_size[layer]))
            print('Top size:{} x {}'.format(network_size[layer + 1], self.activation_size[-1]))
        print('{:<80}'.format('Classifier '))
        print(network_size[-2], '-->', network_size[-1])
        print(80 * '-')
        print('\n\n')

    def forward(self, x, state, reuse=False):
        with tf.variable_scope('AE') as scope:
            if reuse:
                scope.reuse_variables()
            residuals = list()
            state = state or [None] * (self.hidden_layers - 1)
            for layer in range(0, self.hidden_layers): # connect discriminative blocks
                if layer:
                    if state[layer - 1] is not None:
                        s = state[layer - 1]
                    else:
                        s = tf.zeros(x.get_shape().as_list())
                    x = tf.concat(values = [x, s], axis = -1)
                x = conv2d(x, self.network_size[layer + 1], name='D_' + str(layer + 1))
                residuals.append(x)
                x = relu(x)
                x = batch_norm(x, name='BN_D_' + str(layer + 1))
            for layer in reversed(range(0, self.hidden_layers)):
                x = deconv2d(x, self.network_size[layer], name='G_' + str(layer + 1))
                if layer: 
                    state[layer - 1] = x
                    x += residuals[layer - 1]
                x = relu(x)
                x = batch_norm(x, name='BN_G_' + str(layer + 1))
            x_mean = mean(residuals[-1])
            video_index = linear(x_mean, self.network_size[-1], 'linear')
            
            return (x, state), (x_mean, video_index)


def _test_model():
    T = 2
    x = tf.placeholder(tf.float32, [T + 1, 1, 4 * 2**3, 6 * 2**3, 3])
    K = 10
    y = tf.placeholder(tf.float32, [T, 1, K])
    model_01 = Model01(network_size=(3, 6, 12, 18, K), input_spatial_size=x[0].get_shape().as_list()[1: 3])
    state = None
    (x_hat, state), (emb, idx) = model_01.forward(x[0], state)

    print('Input size:', tuple(x.get_shape().as_list()))
    print('Output size:', tuple(x_hat.get_shape().as_list()))
    print('Video index size:', tuple(idx.get_shape()))
    for i, s in enumerate(state):
        print('State', i + 1, 'has size:', tuple(s.get_shape().as_list()))
    print('Embedding has size:', emb.get_shape().as_list())

def _test_training():

    K = 10  # number of training videos
    network_size = (3, 6, 12, 18, K)
    T = 6  # sequence length
    max_epoch = 100 # number of epochs
    lr = 1e-3 # learning rate
    state = None

    print('\n{:-^80}'.format(' Train a ' + str(network_size[:-1]) + ' layer network '))
    print('Sequence length T:', T)
    print('Create the input image and target sequences')
    x = tf.placeholder(tf.float32, [T + 1, 1, 4 * 2**3, 6 * 2**3, 3])
    y = tf.placeholder(tf.float32, [T, 1, K])
    print('Input has size', tuple(x.get_shape().as_list()))
    print('Target index has size', tuple(y.get_shape().as_list()))

    print('Define model')
    model = Model01(network_size=network_size, input_spatial_size=x[0].get_shape().as_list()[1: 3])
    loss = []
    reuse = False
    for t in range(0, T):
        (x_hat, state), (emb, idx) = model.forward(x[t], state, reuse=reuse)
        mse = tf.reduce_mean(tf.square(x_hat - x[t + 1]))
        nll = tf.nn.softmax_cross_entropy_with_logits(labels=y[t], logits=idx)
        nll = tf.reduce_mean(nll)
        loss.append(mse + nll)
        reuse = True
    loss = tf.reduce_mean(loss)
    
    t_vars = tf.trainable_variables()
    num_param = 0.0
    for var in t_vars:
        num_param += int(np.prod(var.get_shape()))
    print("Number of paramers: %d" % num_param)

    optim = tf.train.GradientDescentOptimizer(lr).minimize(loss, var_list=t_vars)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    print('Run for', max_epoch, 'iterations')
    x_train = np.random.rand(T + 1, 1, 4 * 2**3, 6 * 2**3, 3)
    y_train = np.random.rand(T, 1, K)
    for epoch in range(0, max_epoch):
        err, _ = sess.run([loss, optim], feed_dict={x: x_train, y:y_train})
        print(' > Epoch {:2d} loss: {:.3f}'.format((epoch + 1), err))

if __name__ == '__main__':
    _test_training()

