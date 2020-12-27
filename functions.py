import tensorflow as tf
import numpy as np

def resblock(input, IC, OC, name, reuse=tf.AUTO_REUSE):

    l1 = tf.nn.relu(input, name=name + 'relu1')

    l1 = tf.layers.conv2d(inputs=l1, filters=np.minimum(IC, OC), kernel_size=3, strides=1, padding='same',
                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name=name + 'l1', reuse=reuse)

    l2 = tf.nn.relu(l1, name='relu2')

    l2 = tf.layers.conv2d(inputs=l2, filters=OC, kernel_size=3, strides=1, padding='same',
                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name=name + 'l2', reuse=reuse)

    if IC != OC:
        input = tf.layers.conv2d(inputs=input, filters=OC, kernel_size=1, strides=1, padding='same',
                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name=name + 'map', reuse=reuse)

    return input + l2


def MC_RLVC(input, reuse=tf.AUTO_REUSE):

    m1 = tf.layers.conv2d(inputs=input, filters=64, kernel_size=3, strides=1, padding='same',
                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc1', reuse=reuse)

    m2 = resblock(m1, 64, 64, name='mc2', reuse=reuse)

    m3 = tf.layers.average_pooling2d(m2, pool_size=2, strides=2, padding='same')

    m4 = resblock(m3, 64, 64, name='mc4', reuse=reuse)

    m5 = tf.layers.average_pooling2d(m4, pool_size=2, strides=2, padding='same')

    m6 = resblock(m5, 64, 64, name='mc6', reuse=reuse)

    m7 = resblock(m6, 64, 64, name='mc7', reuse=reuse)

    m8 = tf.image.resize_images(m7, [2 * tf.shape(m7)[1], 2 * tf.shape(m7)[2]])

    m8 = m4 + m8

    m9 = resblock(m8, 64, 64, name='mc9', reuse=reuse)

    m10 = tf.image.resize_images(m9, [2 * tf.shape(m9)[1], 2 * tf.shape(m9)[2]])

    m10 = m2 + m10

    m11 = resblock(m10, 64, 64, name='mc11', reuse=reuse)

    m12 = tf.layers.conv2d(inputs=m11, filters=64, kernel_size=3, strides=1, padding='same',
                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc12', reuse=reuse)

    m12 = tf.nn.relu(m12, name='relu12')

    m13 = tf.layers.conv2d(inputs=m12, filters=3, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc13', reuse=reuse)

    return m13


def cnn_layers(tensor, layer, num_filters, out_filters, kernel, stride=2, uni=True, act=tf.nn.relu, act_last=None, reuse=tf.AUTO_REUSE):

    for l in range(layer-1):

       tensor = tf.layers.conv2d(inputs=tensor, filters=num_filters, kernel_size=kernel, padding='same',
                       reuse=reuse, activation=act, strides=stride,
                       kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=uni), name='cnn_' + str(l + 1))

    tensor = tf.layers.conv2d(inputs=tensor, filters=out_filters, kernel_size=kernel, padding='same',
                    reuse=reuse, activation=act_last, strides=stride,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=uni), name='cnn_' + str(layer))

    return tensor


def dnn_layers(tensor, layer, num_filters, out_filters, kernel, stride=2, uni=True, act=tf.nn.relu, act_last=None, reuse=tf.AUTO_REUSE):

    for l in range(layer-1):

       tensor = tf.layers.conv2d_transpose(inputs=tensor, filters=num_filters, kernel_size=kernel, padding='same',
                       reuse=reuse, activation=act, strides=stride,
                       kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=uni), name='dnn_' + str(l + 1))

    tensor = tf.layers.conv2d_transpose(inputs=tensor, filters=out_filters, kernel_size=kernel, padding='same',
                    reuse=reuse, activation=act_last, strides=stride,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=uni), name='dnn_' + str(layer))

    return tensor


def recurrent_cnn(tensor, step, layer, num_filters, out_filters, kernel, stride=2, uni=True, act=tf.nn.relu, act_last=None, reuse=tf.AUTO_REUSE):

    for i in range(step):

        tensor_i = tensor[:, i, :, :, :]
        tensor_i = cnn_layers(tensor_i, layer, num_filters, out_filters, kernel, stride, uni, act, act_last, reuse)

        if i == 0:
            tensor_out = tf.expand_dims(tensor_i, 1)
        else:
            tensor_out = tf.concat([tensor_out, tf.expand_dims(tensor_i, 1)], axis=1)

    return tensor_out


def recurrent_dnn(tensor, step, layer, num_filters, out_filters, kernel, stride=2, uni=True, act=tf.nn.relu, act_last=None, reuse=tf.AUTO_REUSE):

    for i in range(step):

        tensor_i = tensor[:, i, :, :, :]
        tensor_i = dnn_layers(tensor_i, layer, num_filters, out_filters, kernel, stride, uni, act, act_last, reuse)

        if i == 0:
            tensor_out = tf.expand_dims(tensor_i, 1)
        else:
            tensor_out = tf.concat([tensor_out, tf.expand_dims(tensor_i, 1)], axis=1)

    return tensor_out


class ConvLSTMCell(tf.nn.rnn_cell.RNNCell):
  """A LSTM cell with convolutions instead of multiplications.
  Reference:
    Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." Advances in Neural Information Processing Systems. 2015.
  """

  def __init__(self, shape, filters, kernel, forget_bias=1.0, activation=tf.tanh, normalize=False, peephole=False, data_format='channels_last', reuse=None):
    super(ConvLSTMCell, self).__init__(_reuse=reuse)
    self._kernel = kernel
    self._filters = filters
    self._forget_bias = forget_bias
    self._activation = activation
    self._normalize = normalize
    self._peephole = peephole
    if data_format == 'channels_last':
        self._size = tf.TensorShape(shape + [self._filters])
        self._feature_axis = self._size.ndims
        self._data_format = None
    elif data_format == 'channels_first':
        self._size = tf.TensorShape([self._filters] + shape)
        self._feature_axis = 0
        self._data_format = 'NC'
    else:
        raise ValueError('Unknown data_format')

  @property
  def state_size(self):
    return tf.nn.rnn_cell.LSTMStateTuple(self._size, self._size)

  @property
  def output_size(self):
    return self._size

  def call(self, x, state):
    c, h = state

    x = tf.concat([x, h], axis=self._feature_axis)
    n = x.shape[-1].value
    m = 4 * self._filters if self._filters > 1 else 4
    W = tf.get_variable('kernel', self._kernel + [n, m])
    y = tf.nn.convolution(x, W, 'SAME', data_format=self._data_format)
    if not self._normalize:
      y += tf.get_variable('bias', [m], initializer=tf.zeros_initializer())
    j, i, f, o = tf.split(y, 4, axis=self._feature_axis)

    if self._peephole:
      i += tf.get_variable('W_ci', c.shape[1:]) * c
      f += tf.get_variable('W_cf', c.shape[1:]) * c

    if self._normalize:
      j = tf.contrib.layers.layer_norm(j)
      i = tf.contrib.layers.layer_norm(i)
      f = tf.contrib.layers.layer_norm(f)

    f = tf.sigmoid(f + self._forget_bias)
    i = tf.sigmoid(i)
    c = c * f + i * self._activation(j)

    if self._peephole:
      o += tf.get_variable('W_co', c.shape[1:]) * c

    if self._normalize:
      o = tf.contrib.layers.layer_norm(o)
      c = tf.contrib.layers.layer_norm(c)

    o = tf.sigmoid(o)
    h = o * self._activation(c)

    state = tf.nn.rnn_cell.LSTMStateTuple(c, h)

    return h, state


