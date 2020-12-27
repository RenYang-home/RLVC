import tensorflow as tf
import tensorflow_compression as tfc
import functions

def one_step_rnn(tensor, state_c, state_h, Height, Width, num_filters, scale, kernal, act):

    tensor = tf.expand_dims(tensor, axis=1)

    cell = functions.ConvLSTMCell(shape=[Height // scale, Width // scale], activation=act,
                                 filters=num_filters, kernel=kernal)
    state = tf.nn.rnn_cell.LSTMStateTuple(state_c, state_h)
    tensor, state = tf.nn.dynamic_rnn(cell, tensor, initial_state=state, dtype=tensor.dtype)
    state_c, state_h = state

    tensor = tf.squeeze(tensor, axis=1)

    return tensor, state_c, state_h


def MV_analysis(tensor, num_filters, out_filters, Height, Width, c_state, h_state, act):
  """Builds the analysis transform."""

  with tf.variable_scope("MV_analysis", reuse=tf.AUTO_REUSE):
    with tf.variable_scope("layer_0"):
      layer = tfc.SignalConv2D(
          num_filters, (3, 3), corr=True, strides_down=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN())
      tensor = layer(tensor)

    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
          num_filters, (3, 3), corr=True, strides_down=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN())
      tensor = layer(tensor)

    with tf.variable_scope("recurrent"):
      tensor, c_state_out, h_state_out = one_step_rnn(tensor, c_state, h_state,
                                              Height, Width, num_filters,
                                              scale=4, kernal=[3, 3], act=act)

    with tf.variable_scope("layer_2"):
      layer = tfc.SignalConv2D(
          num_filters, (3, 3), corr=True, strides_down=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN())
      tensor = layer(tensor)

    with tf.variable_scope("layer_3"):
      layer = tfc.SignalConv2D(
          out_filters, (3, 3), corr=True, strides_down=2, padding="same_zeros",
          use_bias=True, activation=None)
      tensor = layer(tensor)

    return tensor, c_state_out, h_state_out


def MV_synthesis(tensor, num_filters, Height, Width, c_state, h_state, act):
  """Builds the synthesis transform."""

  with tf.variable_scope("MV_synthesis", reuse=tf.AUTO_REUSE):
    with tf.variable_scope("layer_0"):
      layer = tfc.SignalConv2D(
          num_filters, (3, 3), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN(inverse=True))
      tensor = layer(tensor)

    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
          num_filters, (3, 3), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN(inverse=True))
      tensor = layer(tensor)

    with tf.variable_scope("recurrent"):
      tensor, c_state_out, h_state_out = one_step_rnn(tensor, c_state, h_state,
                                              Height, Width, num_filters,
                                              scale=4, kernal=[3, 3], act=act)

    with tf.variable_scope("layer_2"):
      layer = tfc.SignalConv2D(
          num_filters, (3, 3), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN(inverse=True))
      tensor = layer(tensor)

    with tf.variable_scope("layer_3"):
      layer = tfc.SignalConv2D(
          2, (3, 3), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=None)
      tensor = layer(tensor)

    return tensor, c_state_out, h_state_out


def Res_analysis(tensor, num_filters, out_filters, Height, Width, c_state, h_state, act):
  """Builds the analysis transform."""

  with tf.variable_scope("analysis", reuse=tf.AUTO_REUSE):
    with tf.variable_scope("layer_0"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN())
      tensor = layer(tensor)

    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN())
      tensor = layer(tensor)

    with tf.variable_scope("recurrent"):
      tensor, c_state_out, h_state_out = one_step_rnn(tensor, c_state, h_state,
                                              Height, Width, num_filters,
                                              scale=4, kernal=[5, 5], act=act)

    with tf.variable_scope("layer_2"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN())
      tensor = layer(tensor)

    with tf.variable_scope("layer_3"):
      layer = tfc.SignalConv2D(
          out_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
          use_bias=True, activation=None)
      tensor = layer(tensor)

    return tensor, c_state_out, h_state_out


def Res_synthesis(tensor, num_filters, Height, Width, c_state, h_state, act):
  """Builds the synthesis transform."""

  with tf.variable_scope("synthesis", reuse=tf.AUTO_REUSE):
    with tf.variable_scope("layer_0"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN(inverse=True))
      tensor = layer(tensor)

    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN(inverse=True))
      tensor = layer(tensor)

    with tf.variable_scope("recurrent"):
      tensor, c_state_out, h_state_out = one_step_rnn(tensor, c_state, h_state,
                                              Height, Width, num_filters,
                                              scale=4, kernal=[5, 5], act=act)

    with tf.variable_scope("layer_2"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN(inverse=True))
      tensor = layer(tensor)

    with tf.variable_scope("layer_3"):
      layer = tfc.SignalConv2D(
          3, (5, 5), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=None)
      tensor = layer(tensor)

    return tensor, c_state_out, h_state_out


def rec_prob(tensor, num_filters, Height, Width, c_state, h_state, k=3, act=tf.tanh):

  with tf.variable_scope("CNN_input"):
      tensor = tf.expand_dims(tensor, axis=1)
      y1 = functions.recurrent_cnn(tensor, 1, layer=4, num_filters=num_filters, stride=1,
                                   out_filters=num_filters, kernel=[k, k], act=tf.nn.relu, act_last=None)
      y1 = tf.squeeze(y1, axis=1)

  with tf.variable_scope("RNN"):
      y2, c_state_out, h_state_out = one_step_rnn(y1, c_state, h_state,
                                                      Height, Width, num_filters,
                                                      scale=16, kernal=[k, k], act=act)

  with tf.variable_scope("CNN_output"):
      y2 = tf.expand_dims(y2, axis=1)
      y3 = functions.recurrent_cnn(y2, 1, layer=4, num_filters=num_filters, stride=1,
                                   out_filters=2 * num_filters, kernel=[k, k], act=tf.nn.relu, act_last=None)
      y3 = tf.squeeze(y3, axis=1)

  return y3, c_state_out, h_state_out


def bpp_est(x_target, sigma_mu, num_filters, tiny=1e-10):

    sigma, mu = tf.split(sigma_mu, [num_filters, num_filters], axis=-1)

    half = tf.constant(.5, dtype=tf.float32)

    upper = tf.math.add(x_target, half)
    lower = tf.math.add(x_target, -half)

    sig = tf.maximum(sigma, -7.0)
    upper_l = tf.sigmoid(tf.multiply((upper - mu), (tf.exp(-sig) + tiny)))
    lower_l = tf.sigmoid(tf.multiply((lower - mu), (tf.exp(-sig) + tiny)))
    p_element = upper_l - lower_l
    p_element = tf.clip_by_value(p_element, tiny, 1 - tiny)

    ent = -tf.log(p_element) / tf.log(2.0)
    bits = tf.math.reduce_sum(ent)

    return bits, sigma, mu