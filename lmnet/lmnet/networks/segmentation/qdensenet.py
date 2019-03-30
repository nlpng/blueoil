# -*- coding: utf-8 -*-
# File: resnet_model.py

import functools
import tensorflow as tf

from tensorpack.tfutils.argscope import argscope, get_arg_scope

from lmnet.quantizations import (
    binary_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer,
)


def fix_bn(name, l, data_format):
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
        training=False,
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


def basicblock(name, x, ch_out, data_format, activation):
    ch_in = x.get_shape().as_list()[1 if data_format in ['NCHW', 'channels_first'] else 3]
    skip = x
    with tf.variable_scope(name):
        x = fix_conv('conv1', x, ch_out, 3, 1, data_format)
        x = fix_bn('bn1', x, data_format)
        with tf.variable_scope('relu1'):
            x = activation(x)

        x = fix_conv('conv2', x, ch_out, 3, 1, data_format)
        x = fix_bn('bn2', x, data_format)
        with tf.variable_scope('relu2'):
            x = activation(x)

        if ch_in != ch_out:
            skip = x = fix_conv('conv3', skip, ch_out, 1, 1, data_format)
            skip = activation(fix_bn('bn3', skip, data_format))

    return tf.concat([x, skip], axis=-1)


def group(name, x, out_ch, data_format, down, activation, kernel_setter):
    with tf.variable_scope(name, custom_getter=kernel_setter):
        if down:
            x = tf.space_to_depth(x, block_size=2, name=name + '_pool')
        x = fix_conv(name + '_conv', x, out_ch, 1, 1, data_format)
        x = activation(fix_bn(name + '_bn', x, data_format))
        x = basicblock(name + '_block', x, out_ch, data_format, activation)
    return x


def densenet_backbone(image, ch_out, data_format, activation, kernel_setter):
    x = fix_conv('conv0', image, ch_out, 3, 2, data_format)
    x = activation(fix_bn('bn0', x, data_format))

    x = group('group1', x, ch_out, data_format, True, activation, kernel_setter)
    x = group('group2', x, ch_out * 2, data_format, True, activation, kernel_setter)
    x = group('group3', x, ch_out * 4, data_format, True, activation, kernel_setter)

    x16_down = x
    with tf.variable_scope('x16_down', custom_getter=kernel_setter):
        x16_down = fix_conv('conv_x16', x16_down, ch_out * 4, 1, 1, data_format)
        x16_down = tf.nn.relu(fix_bn('bn_x16', x16_down, data_format))

    x = group('group4', x, ch_out * 8, data_format, True, activation, kernel_setter)
    x32_down = x
    with tf.variable_scope('x32_down', custom_getter=kernel_setter):
        x32_down = fix_conv('conv_x32', x32_down, ch_out * 8, 1, 1, data_format)
        x32_down = tf.nn.relu(fix_bn('bn_x32', x32_down, data_format))
    logits = tf.reduce_mean(x32_down, [1, 2], keep_dims=True, name='gap')

    return logits, x32_down, x16_down
