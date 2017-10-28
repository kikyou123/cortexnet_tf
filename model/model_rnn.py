import os
import tensorflow as tf
from math import ceil
import numpy as np
from op import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Model02RG(object):
    def __init__(self, network_size, input_spatial_size, batch_size, T, c_dim = 3):
        self.hidden_layers = len(network_size) - 2
        self.network_size = network_size
        self.input_spatial_size = input_spatial_size
        self.batch_size = batch_size
        self.T = T
        self.K = network_size[-1]
        self.c_dim = c_dim

        print('\n{:-^80}'.format(' Building model '))
        print('Hidden layers:', self.hidden_layers)
        print('Net sizing:', network_size)
        print('Input spatial size: {} x {}'.format(network_size[0], input_spatial_size))
        self.activation_size = [input_spatial_size]

        for layer in range(0, self.hidden_layers):
            print('{:<80}'.format('Layer ' + str(layer + 1) + ' '))
            print('Bottom size: {} x {}'.format(network_size[layer], self.activation_size[-1]))
            self.activation_size.append(tuple(ceil(s / 2) for s in self.activation_size[layer]))
            print('Top size: {} x {}'.format(network_size[layer + 1], self.activation_size[-1]))
        print('{:<80}'.format('Classifier '))
        print(network_size[-2], '-->', network_size[-1])
        print(80 * '-')
        print('\n\n')

        self.input_shape_x = [T + 1, batch_size, input_spatial_size[0],   input_spatial_size[1], c_dim]
        self.input_shape_y = [T, batch_size, K]
        self.build_model()

    def build_model(self):
        self.x = tf.placeholder(tf.float32, self.input_shape_x)
        self.y = tf.placeholder(tf.float32, self.input_shape_y)
        frame_predictions, frame_logits = self.forward(self.x)

        self.frame_predictions = tf.concat(axis=0, values=frame_predictions) #[T, B, H, W, C]
        self.frame_logits = tf.concat(axis=0, values=frame_logits) #[T, B, K]
        
        self.t_vars = tf.trainable_variables()
        num_param = 0.0
        for var in self.t_vars:
            num_param += int(np.prod(var.get_shape()))
        print("Number of paramers: %d" % num_param)

        self.saver = tf.train.Saver(max_to_keep = 10)


    def forward(self, x):
        T = self.T
        K = self.K
        frame_predictions = []
        frame_logits = []
        reuse = False
        state = None
        for t in range(0, T):
            (x_hat, state), (emb, idx) = self.step(x[t], state, reuse=reuse)
            frame_predictions.append(tf.reshape(x_hat, [1, self.batch_size, self.input_spatial_size[0], self.input_spatial_size[1], self.c_dim]))
            frame_logits.append(tf.reshape(idx, [1, self.batch_size, K]))
            reuse = True
        
        return frame_predictions, frame_logits


    def step(self, x, state, reuse=False):
        with tf.variable_scope('one_step') as scope:
            if reuse:
                scope.reuse_variables()
            residuals = list()
            state = state or [[None] * (self.hidden_layers - 1), [None] * self.hidden_layers]
            for layer in range(0, self.hidden_layers): # connect discriminative blocks
                if layer:
                    if state[0][layer - 1] is not None:
                        s = state[0][layer - 1] 
                    else:
                        s = tf.zeros(x.get_shape().as_list())
                    x = tf.concat(values=[x, s], axis = -1)
                x = conv2d(x, self.network_size[layer + 1], name='D_' + str(layer + 1))
                residuals.append(x)
                x = relu(x)
                x = batch_norm(x, name='BN_D_' + str(layer + 1))

            for layer in reversed(range(0, self.hidden_layers)):
                x = deconv2d(x, self.network_size[layer], name='G_' + str(layer + 1))
                if state[1][layer] is None:
                    get_init_variables(x, self.network_size[layer], name='S_' + str(layer + 1))
                if state[1][layer] is not None:
                    s = conv2d(state[1][layer], self.network_size[layer], d_h = 1, d_w = 1, bias = False, name='S_' + str(layer + 1))
                    x += s
                state[1][layer] = x
                if layer: 
                    state[0][layer - 1] = x
                    x += residuals[layer - 1]
                x = relu(x)
                x = batch_norm(x, name='BN_G_' + str(layer + 1))
            x_mean = mean(residuals[-1])
            video_index = linear(x_mean, self.network_size[-1], 'linear')
            
            return (x, state), (x_mean, video_index)


def _test_model():
    T = 2
    input_spatial_size = [4 * 2**3, 6 * 2**3]
    c_dim = 3
    batch_size = 1
    K = 10
    network_size = (3, 6, 12, 18, K)
    model_01 = Model02RG(network_size, input_spatial_size, batch_size, T, K, c_dim)

    print('Input size:', tuple(model_01.x.get_shape().as_list()))
    print('Output size:', tuple(model_01.frame_predictions.get_shape().as_list()))
    print('Video index size:', tuple(model_01.frame_logits.get_shape().as_list()))

def _test_training():

    K = 10  # number of training videos
    network_size = (3, 6, 12, 18, K)
    T = 6  # sequence length
    max_epoch = 100 # number of epochs
    lr = 1e-3 # learning rate
    input_spatial_size = [4 * 2**3, 6 * 2**3]
    c_dim = 3
    batch_size = 1

    print('\n{:-^80}'.format(' Train a ' + str(network_size[:-1]) + ' layer network '))
    print('Sequence length T:', T)

    print('Define model')
    model_01 = Model02RG(network_size, input_spatial_size, batch_size, T, K, c_dim)
    print('Create the input image and target sequences')
    print('Input has size', tuple(model_01.x.get_shape().as_list()))
    print('Target index has size', tuple(model_01.y.get_shape().as_list()))

    loss = []
    x = model_01.x
    y = model_01.y
    pred_img = model_01.frame_predictions
    pred_logits = model_01.frame_logits
    for t in range(0, T):
        mse = tf.reduce_mean(tf.square(pred_img[t] - x[t + 1]))
        nll = tf.nn.softmax_cross_entropy_with_logits(labels=y[t], logits=pred_logits[t])
        nll = tf.reduce_mean(nll)
        loss.append(mse + nll)
    loss = tf.reduce_mean(loss)
    
    optim = tf.train.GradientDescentOptimizer(lr).minimize(loss, var_list=model_01.t_vars)
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
#    _test_model()
    _test_training()

