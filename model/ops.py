# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : ops.py
# Description :base operators.
# --------------------------------------

import tensorflow as tf

def mish(inputs):
    """
    mish activation function.
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    :param inputs: inputs data
    :return: same shape as the input.
    """
    with tf.variable_scope('mish'):
        return inputs * tf.math.tanh(tf.math.softplus(inputs))

def leaky_relu(inputs, alpha):
    """
    leaky relu activation function.
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    :param inputs: inputs data
    :return: same shape as the input.
    """
    with tf.variable_scope('leaky_relu'):
        f1 = 0.5 * (1 + alpha)
        f2 = 0.5 * (1 - alpha)
        return f1 * inputs + f2 * tf.abs(inputs)

def conv2d(inputs, filters_shape, trainable, downsample=False, activate='mish', bn=True, scope='conv2d'):
    with tf.variable_scope(scope):
        if downsample:
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(inputs, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"
            input_data = inputs

        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True, shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)

        if bn:
            conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(), training=trainable)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True, dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)

        if activate == 'leaky':
            conv = leaky_relu(conv, alpha=0.1)
        elif activate == 'mish':
            conv = mish(conv)
        else:
            return conv
    return conv

def csp_block(inputs, trans_channels, res_channels, left_channels, right_channels, output_channels, iter_num, trainable, scope):
    input_channels = inputs.get_shape().as_list()[3]
    concat_channels = left_channels + right_channels
    with tf.variable_scope(scope):
        inputs_data = conv2d(inputs, filters_shape=(3, 3, input_channels, trans_channels), trainable=trainable, downsample=True, scope='in_conv')
        short_cut = inputs_data
        residual = residual_block(inputs_data, res_channels, iter_num, trainable, scope='residual_block')
        inputs_left = conv2d(residual, filters_shape=(1, 1, res_channels, left_channels), trainable=trainable, scope='left_conv')
        inputs_right = conv2d(short_cut, filters_shape=(1, 1, trans_channels, right_channels), trainable=trainable, scope='right_conv')
        outputs = route(inputs_left, inputs_right, scope='concat')
        outputs = conv2d(outputs, filters_shape=(1, 1, concat_channels, output_channels), trainable=trainable, scope='output_conv')

    return outputs

def residual_block(inputs, output_channels, iter_num, trainable, scope):
    input_channels = inputs.get_shape().as_list()[3]
    with tf.variable_scope(scope):
        inputs = conv2d(inputs, filters_shape=(1, 1, input_channels, output_channels), trainable=trainable, scope='conv')
        short_cut = inputs
        for i in range(iter_num):
            inputs_data = conv2d(inputs, filters_shape=(1, 1, output_channels, output_channels), trainable=trainable, scope='conv_1_'+str(i))
            inputs_data = conv2d(inputs_data, filters_shape=(3, 3, output_channels, output_channels), trainable=trainable, scope='conv_2_'+str(i))
            outputs = tf.add(inputs_data, short_cut)

    return outputs

def spp_block(inputs, filter_num1=512, filter_num2=1024, trainable=True, scope='spp_block'):
    input_channels = inputs.get_shape().as_list()[3]
    with tf.variable_scope(scope):
        inputs_data = conv2d(inputs, filters_shape=(1, 1, input_channels, filter_num1), trainable=trainable, activate='leaky', scope='conv_1')
        inputs_data = conv2d(inputs_data, filters_shape=(3, 3, filter_num1, filter_num2), trainable=trainable, activate='leaky', scope='conv_2')
        inputs_data = conv2d(inputs_data, filters_shape=(1, 1, filter_num2, filter_num1), trainable=trainable, activate='leaky', scope='conv_3')
        spp = spatial_pyramid_pooling(inputs_data, 5, 9, 13, scope='spp')
        inputs_data = conv2d(spp, filters_shape=(1, 1, filter_num1 * 4, filter_num1), trainable=trainable, activate='leaky',scope='conv_4')
        inputs_data = conv2d(inputs_data, filters_shape=(3, 3, filter_num1, filter_num2), trainable=trainable, activate='leaky',scope='conv_5')
        outputs = conv2d(inputs_data, filters_shape=(1, 1, filter_num2, filter_num1), trainable=trainable, activate='leaky',scope='conv_6')

    return outputs

def spatial_pyramid_pooling(inputs, pool_size_1, pool_size_2, pool_size_3, scope):
    with tf.variable_scope(scope):
        pool_1 = maxpool(inputs, pool_size_1, stride=1, scope='pool_1')
        pool_2 = maxpool(inputs, pool_size_2, stride=1, scope='pool_2')
        pool_3 = maxpool(inputs, pool_size_3, stride=1, scope='pool_3')
        outputs = tf.concat([pool_1, pool_2, pool_3, inputs], axis=-1)

    return outputs

def maxpool(inputs, size=2, stride=2, scope='maxpool'):
    with tf.variable_scope(scope):
         pool = tf.layers.max_pooling2d(inputs, pool_size=size, strides=stride, padding='SAME')

    return pool

def route(previous_output, current_output, scope='concat'):
    with tf.variable_scope(scope):
        outputs = tf.concat([current_output, previous_output], axis=-1)

    return outputs

def upsample_block(inputs_1, inputs_2, filter_num1, filter_num2, trainable=True, scope='upsample_block'):
    input_channels_1 = inputs_1.get_shape().as_list()[3]
    input_channels_2 = inputs_2.get_shape().as_list()[3]
    input_channels = filter_num1 + filter_num1
    with tf.variable_scope(scope):
        inputs_data_1 = conv2d(inputs_1, filters_shape=(1, 1, input_channels_1, filter_num1), trainable=trainable, activate='leaky', scope='conv_1')
        inputs_data_2 = conv2d(inputs_2, filters_shape=(1, 1, input_channels_2, filter_num1), trainable=trainable, activate='leaky', scope='conv_2')
        inputs_data_2 = upsample(inputs_data_2, "resize")
        inputs_data = route(inputs_data_1, inputs_data_2)
        inputs_data = conv2d(inputs_data, filters_shape=(1, 1, input_channels, filter_num1), trainable=trainable, activate='leaky', scope='conv_3')
        inputs_data = conv2d(inputs_data, filters_shape=(3, 3, filter_num1, filter_num2), trainable=trainable, activate='leaky', scope='conv_4')
        inputs_data = conv2d(inputs_data, filters_shape=(1, 1, filter_num2, filter_num1), trainable=trainable, activate='leaky', scope='conv_5')
        inputs_data = conv2d(inputs_data, filters_shape=(3, 3, filter_num1, filter_num2), trainable=trainable, activate='leaky', scope='conv_6')
        outputs = conv2d(inputs_data, filters_shape=(1, 1, filter_num2, filter_num1), trainable=trainable, activate='leaky', scope='conv_7')

    return outputs

def upsample(inputs, method="deconv", scope="upsample"):
    assert method in ["resize", "deconv"]

    if method == "resize":
        with tf.variable_scope(scope):
            input_shape = tf.shape(inputs)
            outputs = tf.image.resize_nearest_neighbor(inputs, (input_shape[1] * 2, input_shape[2] * 2))

    if method == "deconv":
        numm_filter = inputs.shape.as_list()[-1]
        outputs = tf.layers.conv2d_transpose(inputs, numm_filter, kernel_size=2, padding='same', strides=(2,2), kernel_initializer=tf.random_normal_initializer())

    return outputs

def downsample_block(inputs_1, inputs_2, filter_num1, filter_num2, trainable=True, scope='downsample_block'):
    input_channels_1 = inputs_1.get_shape().as_list()[3]
    input_channels_2 = inputs_2.get_shape().as_list()[3]
    input_channels = filter_num1 + input_channels_2
    with tf.variable_scope(scope):
        inputs_data_1 = conv2d(inputs_1, filters_shape=(3, 3, input_channels_1, filter_num1), trainable=trainable, downsample=True, activate='leaky', scope='conv_1')
        inputs_data = route(inputs_data_1, inputs_2)
        inputs_data = conv2d(inputs_data, filters_shape=(1, 1, input_channels, filter_num1), trainable=trainable, activate='leaky', scope='conv_2')
        inputs_data = conv2d(inputs_data, filters_shape=(3, 3, filter_num1, filter_num2), trainable=trainable, activate='leaky', scope='conv_3')
        inputs_data = conv2d(inputs_data, filters_shape=(1, 1, filter_num2, filter_num1), trainable=trainable, activate='leaky', scope='conv_4')
        inputs_data = conv2d(inputs_data, filters_shape=(3, 3, filter_num1, filter_num2), trainable=trainable, activate='leaky', scope='conv_5')
        outputs = conv2d(inputs_data, filters_shape=(1, 1, filter_num2, filter_num1), trainable=trainable, activate='leaky', scope='conv_6')

    return outputs