# -*- coding: utf-8 -*-
# Copyright 2018 The Blueoil Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import os
import functools
import tensorflow as tf
import tensorflow.contrib.slim as slim

from lmnet.networks.segmentation.base import Base
# from lmnet.networks.segmentation.tp_resnet18 import resnet_backbone, resnet_basicblock, resnet_group
from lmnet.networks.segmentation.tp_qresnet18 import resnet_backbone, weight_quantizer, activation_quantizer

from tensorpack.models import *
from tensorpack.tfutils import varmanip, argscope, get_model_loader, tower, SaverRestore


class BiSeNet(Base):
    """bilateral semantic segmentation network."""

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

        self.activation = tf.nn.relu
        self.custom_getter = None

        self.w_quantizer = weight_quantizer(True)
        self.a_quantizer = activation_quantizer(True)

    @staticmethod
    def _batch_norm(inputs, training):
        return tf.contrib.layers.batch_norm(
            inputs,
            decay=0.997,
            updates_collections=None,
            is_training=training,
            activation_fn=None,
            center=True,
            scale=True)

    @staticmethod
    def _conv2d(x, filters, kernel_size, strides):
        if strides == 2:
            x = tf.space_to_depth(x, block_size=2, data_format="NHWC")

        return tf.layers.conv2d(
            x, filters, kernel_size,
            strides=1,
            padding="SAME",
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            use_bias=False)

    def _conv_block(self, name,  x, out_ch, ksize, strides, is_training,
                    batchnorm=True, quantizer=None, activation=tf.nn.relu):
        """
        Basic conv block for Encoder-Decoder
        Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
        """

        with tf.variable_scope(name, custom_getter=quantizer):
            x = self._conv2d(x, out_ch, ksize, strides)
            if batchnorm:
                x = self._batch_norm(x, is_training)
            x = activation(x)
        return x

    def _attention_refinement(self, name, x, out_ch, is_training):
        """ Attention Refinement Module (ARM) """

        x = self._conv_block(name, x, out_ch, 3, 1, is_training, batchnorm=False)

        # Global average pooling
        net = tf.reduce_mean(x, [1, 2], keep_dims=True)

        net = self._conv2d(net, out_ch, 1, 1)
        # net = self._batch_norm(net, is_training)
        net = tf.sigmoid(net)

        output = tf.multiply(x, net)

        return output

    def _feature_fusion(self, name, input_1, input_2, out_ch, is_training):
        """ Feature Fusion Module (FFM) """

        inputs = tf.concat([input_1, input_2], axis=-1)
        inputs = self._conv_block(name, inputs, out_ch, 1, 1, is_training, batchnorm=False)

        # Global average pooling
        net = tf.reduce_mean(inputs, [1, 2], keep_dims=True)

        net = self._conv2d(net, out_ch, 1, 1)
        net = tf.nn.relu(net)
        net = self._conv2d(net, out_ch, 1, 1)
        net = tf.sigmoid(net)

        net = tf.multiply(inputs, net)

        net = tf.add(inputs, net)

        return net

    def _bisenet_head(self, name, inputs, out_ch, is_training, scale, is_aux=False):
        """ The output heads of BiSeNet """
        if is_aux:
            net = self._conv_block(name, inputs, 256, 3, 1, is_training)
        else:
            net = self._conv_block(name, inputs, 64, 3, 1, is_training)

        net = self._conv2d(net, out_ch, 1, 1)
        if scale > 1:
            h = net.get_shape()[1].value * scale
            w = net.get_shape()[2].value * scale
            net = tf.image.resize_bilinear(net, size=[h, w], align_corners=True)
        return net

    def base(self, images, is_training, *args, **kwargs):

        self.images = images

        # spatial path
        spatial = self._conv_block('spatial0', images, 64, 3, 2, is_training,
                                   batchnorm=True, quantizer=None, activation=self.a_quantizer)
        spatial = self._conv_block('spatial1', spatial, 64, 3, 2, is_training,
                                   batchnorm=True, quantizer=self.w_quantizer, activation=self.a_quantizer)
        spatial = self._conv_block('spatial2', spatial, 128, 3, 2, is_training,
                                   batchnorm=True, quantizer=self.w_quantizer)

        # context path
        logits, down_x32, down_x16 = resnet_backbone(images, 64, 'NHWC', self.a_quantizer, self.w_quantizer)

        global_context = down_x32

        arm_1 = self._attention_refinement('arm_1', down_x32, 128, is_training)
        arm_1_concat = tf.concat([global_context, arm_1], axis=-1)
        global_context = tf.depth_to_space(arm_1_concat, block_size=2)
        global_context = self._conv_block('context_arm_1', global_context, 128, 3, 1, is_training, batchnorm=False)

        self.aux0 = self._bisenet_head('aux0', global_context, self.num_classes, is_training, 16, is_aux=True)

        arm_2 = self._attention_refinement('arm_2', down_x16, 128, is_training)
        arm_2 += global_context
        global_context = tf.depth_to_space(arm_2, block_size=2)
        global_context = self._conv_block('context_arm_2', global_context, 128, 3, 1, is_training, batchnorm=False)

        self.aux1 = self._bisenet_head('aux1', global_context, self.num_classes, is_training, 8, is_aux=True)

        context = global_context

        x = self._feature_fusion('ffm', spatial, context, 256, is_training)
        x = self._bisenet_head('logit', x, self.num_classes, is_training, 8)
        return x

    def _weight_decay_loss(self):
        """L2 weight decay (regularization) loss."""
        losses = []
        # print("apply l2 loss these variables")
        for var in tf.trainable_variables():
            # exclude batch norm variable
            if "kernel" in var.name or 'weights' in var.name or 'W' in var.name:
                # print(var.name)
                losses.append(tf.nn.l2_loss(var))

        return tf.add_n(losses) * self.weight_decay_rate

    def loss(self, output, labels):
        """Loss

        Args:
           output: Tensor of network output. shape is (batch_size, output_height, output_width, num_classes).
           labels: Tensor of grayscale imnage gt labels. shape is (batch_size, height, width).
        """
        if self.data_format == 'NCHW':
            output = tf.transpose(output, perm=[0, 2, 3, 1])
        with tf.name_scope("loss"):
            # calculate loss weights for each batch.
            loss_weight = []
            all_size = tf.to_float(tf.reduce_prod(tf.shape(labels)))
            for class_index in range(self.num_classes):
                num_label = tf.reduce_sum(tf.to_float(tf.equal(labels, class_index)))
                weight = (all_size - num_label) / all_size
                # TODO(wakisaka): 3 is masic number. ratio setting
                # weight = weight ** 3
                loss_weight.append(weight)

            reshape_main = tf.reshape(output, (-1, self.num_classes))
            reshape_aux0 = tf.reshape(self.aux0, (-1, self.num_classes))
            reshape_aux1 = tf.reshape(self.aux1, (-1, self.num_classes))

            label_flat = tf.reshape(labels, (-1, 1))
            labels = tf.reshape(tf.one_hot(label_flat, depth=self.num_classes), (-1, self.num_classes))

            reshape_main = tf.nn.softmax(reshape_main)
            cross_entropy_main = -tf.reduce_sum(
                (labels * tf.log(tf.clip_by_value(reshape_main, 1e-10, 1.0))) * loss_weight,
                axis=[1]
            )
            cross_entropy_main_mean = tf.reduce_mean(cross_entropy_main, name="cross_entropy_main_mean")

            # cross_entropy_main_mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=reshape_main,
            #                                                                                  labels=labels))

            reshape_aux0 = tf.nn.softmax(reshape_aux0)
            cross_entropy_aux0 = -tf.reduce_sum(
                (labels * tf.log(tf.clip_by_value(reshape_aux0, 1e-10, 1.0))) * loss_weight,
                axis=[1]
            )
            cross_entropy_aux0_mean = tf.reduce_mean(cross_entropy_aux0, name="cross_entropy_aux0_mean")

            # cross_entropy_aux0_mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=reshape_aux0,
            #                                                                                  labels=labels))

            reshape_aux1 = tf.nn.softmax(reshape_aux1)
            cross_entropy_aux1 = -tf.reduce_sum(
                (labels * tf.log(tf.clip_by_value(reshape_aux1, 1e-10, 1.0))) * loss_weight,
                axis=[1]
            )
            cross_entropy_aux1_mean = tf.reduce_mean(cross_entropy_aux1, name="cross_entropy_aux1_mean")

            # cross_entropy_aux1_mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=reshape_aux1,
            #                                                                                  labels=labels))

            cross_entropy_mean = cross_entropy_main_mean + cross_entropy_aux0_mean + cross_entropy_aux1_mean

            # cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy_mean")

            loss = cross_entropy_mean
            if self.weight_decay_rate:
                weight_decay_loss = self._weight_decay_loss()
                tf.summary.scalar("weight_decay", weight_decay_loss)
                loss = loss + weight_decay_loss

            tf.summary.scalar("loss", loss)

            return loss
