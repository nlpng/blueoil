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
from easydict import EasyDict
import tensorflow as tf

from lmnet.common import Tasks
from lmnet.networks.classification.lmnet_v1 import LmnetV1Quantize
from lmnet.datasets.cifar10 import Cifar10
from lmnet.data_processor import Sequence
from lmnet.pre_processor import (
    Resize,
    DivideBy255,
)
from lmnet.data_augmentor import (
    Crop,
    FlipLeftRight,
    Pad,
)
from lmnet.quantizations import (
    binary_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer,
)

IS_DEBUG = False

NETWORK_CLASS = LmnetV1Quantize
DATASET_CLASS = Cifar10

IMAGE_SIZE = [32, 32]
BATCH_SIZE = 256
DATA_FORMAT = "NHWC"
TASK = Tasks.CLASSIFICATION
CLASSES = DATASET_CLASS.classes

MAX_STEPS = 20000
SAVE_STEPS = 1000
TEST_STEPS = 1000
SUMMARISE_STEPS = 100
# pretrain
USE_PRETRAIN = False
PRETRAIN_VARS = []
PRETRAIN_DIR = ""
PRETRAIN_FILE = ""
# distributed training
IS_DISTRIBUTION = False

# for debug
# MAX_STEPS = 10
# BATCH_SIZE = 31
# SAVE_STEPS = 2
# TEST_STEPS = 10
# SUMMARISE_STEPS = 2
# IS_DEBUG = True

PRE_PROCESSOR = Sequence([
    Resize(size=IMAGE_SIZE),
    DivideBy255()
])
POST_PROCESSOR = None


step_per_epoch = int(50000 / BATCH_SIZE)

# lr_values = []
# lr_boundaries = []
# initial_lr = 0.01
# end_percentage = 0.1
# scale_percentage = None
# num_iterations = MAX_STEPS
# mid_cycle_id = int(num_iterations * (1. - end_percentage) / float(2))
# scale = float(scale_percentage) if scale_percentage is not None else float(end_percentage)
# for step in range(MAX_STEPS):
#     if step > 2 * mid_cycle_id:
#         current_percentage = (step - 2 * mid_cycle_id)
#         current_percentage /= float((num_iterations - 2 * mid_cycle_id))
#         new_lr = initial_lr * (1. + (current_percentage * (1. - 100.) / 100.)) * scale
#     elif step > mid_cycle_id:
#         current_percentage = 1. - (step - mid_cycle_id) / mid_cycle_id
#         new_lr = initial_lr * (1. + current_percentage * (scale * 100 - 1.)) * scale
#     else:
#         current_percentage = step / mid_cycle_id
#         new_lr = initial_lr * (1. + current_percentage * (scale * 100 - 1.)) * scale
#     lr_values.append(new_lr)
#     if step != 0:
#         lr_boundaries.append(step)


NETWORK = EasyDict()
NETWORK.OPTIMIZER_CLASS = tf.train.MomentumOptimizer
NETWORK.OPTIMIZER_KWARGS = {"momentum": 0.9}
NETWORK.LEARNING_RATE_FUNC = 'one_cycle_policy'
NETWORK.LEARNING_RATE_KWARGS = {}
NETWORK.IMAGE_SIZE = IMAGE_SIZE
NETWORK.BATCH_SIZE = BATCH_SIZE
NETWORK.DATA_FORMAT = DATA_FORMAT
NETWORK.WEIGHT_DECAY_RATE = 0.0001
NETWORK.ACTIVATION_QUANTIZER = linear_mid_tread_half_quantizer
NETWORK.ACTIVATION_QUANTIZER_KWARGS = {
    'bit': 2,
    'max_value': 2
}
NETWORK.WEIGHT_QUANTIZER = binary_mean_scaling_quantizer
NETWORK.WEIGHT_QUANTIZER_KWARGS = {}

# dataset
DATASET = EasyDict()
DATASET.BATCH_SIZE = BATCH_SIZE
DATASET.DATA_FORMAT = DATA_FORMAT
DATASET.PRE_PROCESSOR = PRE_PROCESSOR
DATASET.AUGMENTOR = Sequence([
    Pad(2),
    Crop(size=IMAGE_SIZE),
    FlipLeftRight(),
])
# DATASET.TRAIN_VALIDATION_SAVING_SIZE = 5000
