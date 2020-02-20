# -*- coding: utf-8 -*-
# Copyright 2019 The Blueoil Authors. All Rights Reserved.
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
from easydict import EasyDict
import tensorflow as tf

from lmnet.common import Tasks
from lmnet.networks.segmentation.lm_bisenet import LMBiSeNetQuantize
from lmnet.datasets.cityscapes import Cityscapes
from lmnet.data_processor import Sequence
from lmnet.pre_processor import (
    Resize,
    PerImageStandardization,
)
from lmnet.post_processor import (
    Bilinear,
    Softmax,
)
from lmnet.data_augmentor import (
    Brightness,
    Color,
    Contrast,
    FlipLeftRight,
    Hue,
)
from lmnet.quantizations import (
    binary_channel_wise_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer,
)

IS_DEBUG = False

NETWORK_CLASS = LMBiSeNetQuantize
DATASET_CLASS = Cityscapes

IMAGE_SIZE = [160, 320]
BATCH_SIZE = 8
DATA_FORMAT = "NHWC"
TASK = Tasks.SEMANTIC_SEGMENTATION
CLASSES = DATASET_CLASS.classes

MAX_EPOCHS = 500
SAVE_CHECKPOINT_STEPS = 1000
KEEP_CHECKPOINT_MAX = 5
TEST_STEPS = 1000
SUMMARISE_STEPS = 1000


# pretrain
IS_PRETRAIN = False
PRETRAIN_VARS = []
PRETRAIN_DIR = ""
PRETRAIN_FILE = ""

# for debug
# BATCH_SIZE = 2
# SUMMARISE_STEPS = 1
# IS_DEBUG = True

PRE_PROCESSOR = Sequence([
    Resize(size=IMAGE_SIZE),
    PerImageStandardization(),
])
POST_PROCESSOR = Sequence([
    Bilinear(size=IMAGE_SIZE, data_format=DATA_FORMAT, compatible_tensorflow_v1=True),
    Softmax(),
])

steps_per_epoch = 2975 // BATCH_SIZE
max_steps = steps_per_epoch * MAX_EPOCHS


def warmup_cosine_scheduler(learning_rate, global_step, warmup_learning_rate, warmup_step, decay_step):
    import tensorflow as tf
    import numpy as np
    base_lr = tf.convert_to_tensor(learning_rate, tf.float32)
    warm_lr = tf.convert_to_tensor(warmup_learning_rate, tf.float32)
    warm_stp = tf.convert_to_tensor(warmup_step, tf.int32)
    tot_stp = tf.convert_to_tensor(decay_step, tf.int32)

    def lr_scheduler(global_step):

        slope = (base_lr - warm_lr) / tf.cast(warm_stp, tf.float32)
        warmup_rate = slope * tf.cast(global_step, tf.float32) + warm_lr

        # cosine_rate = tf.compat.v1.train.cosine_decay(learning_rate=base_lr,
        #                                               global_step=global_step,
        #                                               decay_steps=tot_stp)
        cosine_rate = 0.5 * base_lr * (1 + tf.cos(
            np.pi * (tf.cast(global_step, tf.float32) - tf.cast(warm_stp, tf.float32)) / (tf.cast(tot_stp, tf.float32) - tf.cast(warm_stp, tf.float32))))

        return tf.where(global_step < warm_stp, warmup_rate, cosine_rate)

    return lr_scheduler(global_step)


NETWORK = EasyDict()
NETWORK.OPTIMIZER_CLASS = tf.train.AdamOptimizer
NETWORK.OPTIMIZER_KWARGS = {}
NETWORK.LEARNING_RATE_FUNC = warmup_cosine_scheduler
NETWORK.LEARNING_RATE_KWARGS = {
    "learning_rate": 0.01,
    "warmup_learning_rate": 0.0001,
    "warmup_step": steps_per_epoch * 5,
    "decay_step": max_steps
}
NETWORK.IMAGE_SIZE = IMAGE_SIZE
NETWORK.BATCH_SIZE = BATCH_SIZE
NETWORK.DATA_FORMAT = DATA_FORMAT
NETWORK.WEIGHT_DECAY_RATE = 0.
NETWORK.AUXILIARY_LOSS_WEIGHT = 0.5
NETWORK.USE_FEATURE_FUSION = True
NETWORK.USE_ATTENTION_REFINEMENT = True
NETWORK.USE_TAIL_GAP = True
NETWORK.ACTIVATION_QUANTIZER = linear_mid_tread_half_quantizer
NETWORK.ACTIVATION_QUANTIZER_KWARGS = {
    'bit': 2,
    'max_value': 2
}
NETWORK.WEIGHT_QUANTIZER = binary_channel_wise_mean_scaling_quantizer
NETWORK.WEIGHT_QUANTIZER_KWARGS = {}

DATASET = EasyDict()
DATASET.BATCH_SIZE = BATCH_SIZE
DATASET.DATA_FORMAT = DATA_FORMAT
DATASET.PRE_PROCESSOR = PRE_PROCESSOR
DATASET.AUGMENTOR = Sequence([
    Resize(size=IMAGE_SIZE),
    Brightness((0.75, 1.25)),
    Color((0.75, 1.25)),
    Contrast((0.75, 1.25)),
    FlipLeftRight(),
    Hue((-10, 10)),
])
DATASET.ENABLE_PREFETCH = True
