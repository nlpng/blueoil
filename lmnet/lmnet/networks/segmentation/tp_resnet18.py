# -*- coding: utf-8 -*-
# File: resnet_model.py

import tensorflow as tf

from tensorpack.models import BatchNorm, BNReLU, Conv2D, FullyConnected, GlobalAvgPooling, MaxPooling
from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.tfutils.varreplace import custom_getter_scope


def rename_get_variable(mapping):
    """
    Args:
        mapping(dict): an old -> new mapping for variable basename. e.g. {'kernel': 'W'}
    Returns:
        A context where the variables are renamed.
    """
    def custom_getter(getter, name, *args, **kwargs):
        splits = name.split('/')
        basename = splits[-1]
        if basename in mapping:
            basename = mapping[basename]
            splits[-1] = basename
            name = '/'.join(splits)
        return getter(name, *args, **kwargs)
    return custom_getter_scope(custom_getter)


def bnrelu(x, name=None):
    x = tf.layers.batch_normalization(x, name='bn', training=False)
    x = tf.nn.relu(x, name=name)
    return x


def resnet_shortcut(x, n_out, stride):
    n_in = x.get_shape().as_list()[3]
    if n_in != n_out:   # change dimension when channel is not the same
        with tf.variable_scope('convshortcut'):
            x = tf.layers.conv2d(x, n_out, 1, strides=stride, padding='SAME', use_bias=False,
                                 kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out'),
                                 name=tf.get_variable_scope())
            x = tf.layers.batch_normalization(x, name='bn', training=False)
        return x
    else:
        return x


def resnet_basicblock(x, ch_out, stride):
    shortcut = x
    x = tf.layers.conv2d(x, ch_out, 3, strides=stride, padding='SAME', use_bias=False,
                         kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out'),
                         activation=bnrelu, name='conv1')
    with tf.variable_scope('conv2'):
        x = tf.layers.conv2d(x, ch_out, 3, padding='SAME', use_bias=False,
                             kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out'),
                             name=tf.get_variable_scope())
        x = tf.layers.batch_normalization(x, gamma_initializer=tf.zeros_initializer(), name='bn', training=False)
    out = x + resnet_shortcut(shortcut, ch_out, stride)
    return tf.nn.relu(out)


def resnet_group(name, x, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                x = block_func(x, features, stride if i == 0 else 1)
    return x


def resnet_backbone(image, num_blocks, group_func, block_func):
    # Note that this pads the image by [2, 3] instead of [3, 2].
    # Similar things happen in later stride=2 layers as well.
    with rename_get_variable({'kernel': 'W', 'bias': 'b'}):
        x = tf.layers.conv2d(image, 64, 7, strides=2, padding='SAME', use_bias=False,
                             kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out'),
                             activation=bnrelu, name='conv0')
        layer = tf.layers.MaxPooling2D(pool_size=3, strides=2, padding='SAME', name='pool0')
        x = layer.apply(x, scope=tf.get_variable_scope())
        x = group_func('group0', x, block_func, 64, num_blocks[0], 1)
        x = group_func('group1', x, block_func, 128, num_blocks[1], 2)
        x = group_func('group2', x, block_func, 256, num_blocks[2], 2)
        x16_down = x
        x = group_func('group3', x, block_func, 512, num_blocks[3], 2)
        x32_down = x
        x = tf.reduce_mean(x, [1, 2], keep_dims=True, name='gap')
    return x, x32_down, x16_down
