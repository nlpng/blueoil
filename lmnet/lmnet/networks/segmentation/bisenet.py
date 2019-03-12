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
from lmnet.networks.segmentation.tensorpack_resnet18 import (
    resnet_backbone, resnet_basicblock, resnet_group)

from tensorpack.models import *
from tensorpack.tfutils import varmanip, argscope, get_model_loader, tower


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

    @staticmethod
    def _build_keras_xception(inputs, is_training=True):
        tf.keras.backend.set_learning_phase(is_training)
        keras_xception = tf.keras.applications.xception.Xception(include_top=False, weights='imagenet',
                                                                 input_tensor=inputs,
                                                                 input_shape=inputs.get_shape()[-3:],
                                                                 pooling=None)
        keras_xception.summary()
        logits = keras_xception.output
        x32_endpoint = keras_xception.get_layer('block13_pool').output
        x16_endpoint = keras_xception.get_layer('block4_pool').output
        return logits, x32_endpoint, x16_endpoint

    @staticmethod
    def _build_keras_resnet50(inputs, is_training=True):
        tf.keras.backend.set_learning_phase(is_training)
        keras_resnet50 = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet',
                                                                 input_tensor=inputs,
                                                                 input_shape=inputs.get_shape()[-3:],
                                                                 pooling=None)
        keras_resnet50.summary()
        logits = keras_resnet50.output
        x32_endpoint = keras_resnet50.get_layer('activation_49').output
        x16_endpoint = keras_resnet50.get_layer('activation_40').output
        # logits, x32_endpoint, x16_endpoint = keras_resnet50(inputs)
        return logits, x32_endpoint, x16_endpoint

    @staticmethod
    def _build_tensorpack_resnet18(inputs, training=True):
        saved_model = '/storage/neil/saved/train_log/imagenet-resnet-d18-batch256/model-500400'
        var = varmanip.load_chkpt_vars(saved_model)
        # print(var.keys())
        logits, x32_endpoint, x16_endpoint = resnet_backbone(inputs, [2, 2, 2, 2], resnet_group, resnet_basicblock)
        # get_model_loader(saved_model)
        return logits, x32_endpoint, x16_endpoint

    @staticmethod
    def _build_tensorpack_qresnet18(inputs, training=True):
        saved_model = '/storage/neil/saved/train_log/imagenet-resnet-d18-batch256/model-500400'
        var = varmanip.load_chkpt_vars(saved_model)
        # print(var.keys())
        logits, x32_endpoint, x16_endpoint = resnet_backbone(inputs, [2, 2, 2, 2], resnet_group, resnet_basicblock)
        get_model_loader(saved_model)
        return logits, x32_endpoint, x16_endpoint

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
    def _conv2d(inputs, filters, kernel_size, strides):
        return tf.layers.conv2d(
            inputs, filters, kernel_size,
            strides=strides,
            padding="SAME",
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            use_bias=False)

    def _conv_block(self, x, out_ch, ksize, strides, is_training):
        """
        Basic conv block for Encoder-Decoder
        Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
        """
        x = self._conv2d(x, out_ch, ksize, strides)
        x = self._batch_norm(x, is_training)
        x = tf.nn.relu(x)
        return x

    def _attention_refinement(self, x, out_ch, is_training):
        """ Attention Refinement Module (ARM) """

        x = self._conv_block(x, out_ch, 3, 1, is_training)

        # Global average pooling
        net = tf.reduce_mean(x, [1, 2], keep_dims=True)

        net = self._conv2d(net, out_ch, 1, 1)
        net = self._batch_norm(net, is_training)
        net = tf.sigmoid(net)

        output = tf.multiply(x, net)

        return output

    def _feature_fusion(self, input_1, input_2, out_ch, is_training):
        """ Feature Fusion Module (FFM) """

        inputs = tf.concat([input_1, input_2], axis=-1)
        inputs = self._conv_block(inputs, out_ch, 1, 1, is_training)

        # Global average pooling
        net = tf.reduce_mean(inputs, [1, 2], keep_dims=True)

        net = self._conv2d(net, out_ch, 1, 1)
        net = tf.nn.relu(net)
        net = self._conv2d(net, out_ch, 1, 1)
        net = tf.sigmoid(net)

        net = tf.multiply(inputs, net)

        net = tf.add(inputs, net)

        return net

    def _bisenet_head(self, inputs, out_ch, is_training, scale, is_aux=False):
        """ The output heads of BiSeNet """
        if is_aux:
            net = self._conv_block(inputs, 256, 3, 1, is_training)
        else:
            net = self._conv_block(inputs, 64, 3, 1, is_training)

        net = self._conv2d(net, out_ch, 1, 1)
        if scale > 1:
            h = net.get_shape()[1].value * scale
            w = net.get_shape()[2].value * scale
            net = tf.image.resize_bilinear(net, size=[h, w], align_corners=True)
        return net

    def base(self, images, is_training, *args, **kwargs):

        self.images = images

        # spatial path
        spatial = self._conv_block(images, 64, 3, 2, is_training)
        spatial = self._conv_block(spatial, 64, 3, 2, is_training)
        spatial = self._conv_block(spatial, 128, 3, 2, is_training)
        # print('Spatial out: ', spatial_out.get_shape())

        # context path
        # frontend use ResNet-50 of imagenet pre-trained model
        # logits, end_points, frontend_scope, init_fn = self._build_frontend(images,
        #                                                                    is_training=is_training,
        #                                                                    pretrained_dir="pretrained")
        # logits, down_x32, down_x16 = self._build_keras_resnet50(images)
        # logits, down_x32, down_x16 = self._build_keras_xception(images)
        logits, down_x32, down_x16 = self._build_tensorpack_resnet18(images)

        global_context = tf.reduce_mean(logits, [1, 2], keep_dims=True)
        global_context = self._conv_block(global_context, 128, 1, 1, is_training)
        global_context = tf.image.resize_bilinear(global_context, size=[11, 15], align_corners=True)

        # pred_out = []

        arm_1 = self._attention_refinement(down_x32, 128, is_training)
        arm_1 += global_context
        global_context = tf.image.resize_bilinear(arm_1, size=[22, 30], align_corners=True)
        global_context = self._conv_block(global_context, 128, 3, 1, is_training)

        self.aux0 = self._bisenet_head(global_context, self.num_classes, is_training, 16, is_aux=True)
        # print('aux 0 out: ', self.aux0.get_shape())
        # pred_out.append(self.aux0)

        arm_2 = self._attention_refinement(down_x16, 128, is_training)
        arm_2 += global_context
        global_context = tf.image.resize_bilinear(arm_2, size=[44, 60], align_corners=True)
        global_context = self._conv_block(global_context, 128, 3, 1, is_training)

        self.aux1 = self._bisenet_head(global_context, self.num_classes, is_training, 8, is_aux=True)
        # print('aux 1 out: ', self.aux1.get_shape())
        # pred_out.append(self.aux1)

        context = global_context
        # print('Context out: ', context_out.get_shape())

        x = self._feature_fusion(spatial, context, 256, is_training)
        x = self._bisenet_head(x, self.num_classes, is_training, 8)
        # pred_out.append(x)
        # print('base output: ', x.get_shape())
        # for var in tf.trainable_variables():
        #     print(var)
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


# class LmSegnetV1Quantize(LmSegnetV1):
#     """LM original quantize semantic segmentation network.
#
#     Following `args` are used for inference: ``activation_quantizer``, ``activation_quantizer_kwargs``,
#     ``weight_quantizer``, ``weight_quantizer_kwargs``.
#
#     Args:
#         activation_quantizer (callable): Weight quantizater. See more at `lmnet.quantizations`.
#         activation_quantizer_kwargs (dict): Kwargs for `activation_quantizer`.
#         weight_quantizer (callable): Activation quantizater. See more at `lmnet.quantizations`.
#         weight_quantizer_kwargs (dict): Kwargs for `weight_quantizer`.
#     """
#
#     def __init__(
#             self,
#             activation_quantizer=None,
#             activation_quantizer_kwargs=None,
#             weight_quantizer=None,
#             weight_quantizer_kwargs=None,
#             *args,
#             **kwargs
#     ):
#         super().__init__(
#             *args,
#             **kwargs
#         )
#
#         assert weight_quantizer
#         assert activation_quantizer
#
#         activation_quantizer_kwargs = activation_quantizer_kwargs if activation_quantizer_kwargs is not None else {}
#         weight_quantizer_kwargs = weight_quantizer_kwargs if weight_quantizer_kwargs is not None else {}
#
#         self.activation = activation_quantizer(**activation_quantizer_kwargs)
#         weight_quantization = weight_quantizer(**weight_quantizer_kwargs)
#         self.custom_getter = functools.partial(self._quantized_variable_getter,
#                                                weight_quantization=weight_quantization)
#
#     @staticmethod
#     def _quantized_variable_getter(getter, name, weight_quantization=None, *args, **kwargs):
#         """Get the quantized variables.
#
#         Use if to choose or skip the target should be quantized.
#
#         Args:
#             getter: Default from tensorflow.
#             name: Default from tensorflow.
#             weight_quantization: Callable object which quantize variable.
#             args: Args.
#             kwargs: Kwargs.
#         """
#         assert callable(weight_quantization)
#         var = getter(name, *args, **kwargs)
#         with tf.variable_scope(name):
#             # Apply weight quantize to variable whose last word of name is "kernel".
#             if "kernel" == var.op.name.split("/")[-1]:
#                 return weight_quantization(var)
#         return var
