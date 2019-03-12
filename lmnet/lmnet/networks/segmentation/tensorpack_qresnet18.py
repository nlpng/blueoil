# -*- coding: utf-8 -*-
# File: resnet_model.py

import functools
import tensorflow as tf

from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.models import GlobalAvgPooling, FullyConnected

from linear import linear_mid_tread_half_quantizer
from binary import binary_mean_scaling_quantizer


def fix_bn(name, l, training, data_format):
    return tf.layers.batch_normalization(
        l,
        axis=-1 if data_format in ['NHWC', 'channels_last'] else 1,
        momentum=0.997,
        epsilon=0.001,
        center=True,
        scale=True,
        beta_initializer=tf.zeros_initializer(),
        gamma_initializer=tf.ones_initializer(),
        moving_mean_initializer=tf.zeros_initializer(),
        moving_variance_initializer=tf.ones_initializer(),
        training=training,
        trainable=True,
        name=name,
        reuse=None,
        renorm=False,
        renorm_clipping=None,
        renorm_momentum=0.99,
        fused=True)


def fix_conv(name, x, filters, kernel_size, strides, data_format):
    if strides == 2:
        x = tf.space_to_depth(x, block_size=2, data_format=data_format, name=name + '_pool')

    return tf.layers.conv2d(
        x, filters, kernel_size,
        padding="SAME",
        data_format='channels_last' if data_format == 'NHWC' else 'channels_first',
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        use_bias=False,
        name=name)


def resnet_basicblock(name, x, ch_out, stride, training, data_format, activation):
    ch_in = x.get_shape().as_list()[-1 if data_format in ['NHWC', 'channels_last'] else 1]
    shortcut = x

    with tf.variable_scope(name):
        x = fix_bn('bn1', x, training, data_format)
        with tf.variable_scope('relu1'):
            x = activation(x, 2, 2.0)

        x = fix_conv('conv1', x, ch_out, 3, stride, data_format)
        x = fix_bn('bn2', x, training, data_format)
        with tf.variable_scope('relu2'):
            x = activation(x, 2, 2.0)

        x = fix_conv('conv2', x, ch_out, 3, 1, data_format)

        if ch_in != ch_out:
            shortcut = tf.nn.avg_pool(shortcut, ksize=[1, stride, stride, 1],
                                      strides=[1, stride, stride, 1], padding='VALID')
            shortcut = tf.pad(shortcut, [[0, 0], [0, 0], [0, 0], [(ch_out - ch_in) // 2, (ch_out - ch_in) // 2]])

    return x + shortcut


def get_quantize_var(getter, name, weight_quantization=None, *args, **kwargs):
    assert callable(weight_quantization)
    var = getter(name, *args, **kwargs)
    with tf.variable_scope(name):
        # Apply weight quantize to variable whose last word of name is "kernel".
        if "kernel" == var.op.name.split("/")[-1]:
            return weight_quantization(var)
    return var


def weight_quantizer(quantized):
    if quantized:
        return functools.partial(
            get_quantize_var,
            weight_quantization=binary_mean_scaling_quantizer())
    else:
        return None


def activation_quantizer(quantized):
    if quantized:
        return linear_mid_tread_half_quantizer(bit=2, max_value=2.0)
    else:
        return tf.nn.relu


def resnet_backbone(image, ch_out, training, data_format, activation, kernel_setter):
    x = fix_conv('Conv0', image, ch_out, 1, 2, data_format)
    x = activation(fix_bn('bn0', x, training, data_format), 2, 2.0)
    with tf.variable_scope('qresidual', custom_getter=kernel_setter):
        x = fix_conv('Conv1', x, ch_out, 3, 2, data_format)
        x = resnet_basicblock('block_1_1', x, ch_out, 1, training, data_format, activation)
        x = resnet_basicblock('block_1_2', x, ch_out, 1, training, data_format, activation)
        x = resnet_basicblock('block_2_1', x, ch_out * 2, 2, training, data_format, activation)
        x = resnet_basicblock('block_2_2', x, ch_out * 2, 1, training, data_format, activation)
        x = resnet_basicblock('block_3_1', x, ch_out * 4, 2, training, data_format, activation)
        x = resnet_basicblock('block_3_2', x, ch_out * 4, 1, training, data_format, activation)
        x = resnet_basicblock('block_4_1', x, ch_out * 8, 2, training, data_format, activation)
        x = resnet_basicblock('block_4_2', x, ch_out * 8, 1, training, data_format, activation)
    x = tf.nn.relu(fix_bn('gbn', x, training, data_format))
    x = GlobalAvgPooling('gap', x)
    logits = FullyConnected('linear', x, 1000,
                            kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    return logits
