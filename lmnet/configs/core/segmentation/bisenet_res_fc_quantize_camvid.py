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
from lmnet.networks.segmentation.bisenet_resnet_fc import BiSeNet
from lmnet.datasets.camvid import Camvid
from lmnet.data_processor import Sequence
from lmnet.pre_processor import (
    Resize,
    DivideBy255,
    PerImageStandardization,
)
from lmnet.data_augmentor import (
    RandomCrop,
    FlipLeftRight,
    FlipTopBottom,
    Brightness,
    Color,
    Contrast,
    FlipLeftRight,
    Hue,
    RandomResize,
    CropOrPad,
)
from lmnet.quantizations import (
    binary_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer,
)

IS_DEBUG = False

NETWORK_CLASS = BiSeNet
DATASET_CLASS = Camvid

# Training data 367
# Validation data 101
IMAGE_SIZE = [352, 480]
BATCH_SIZE = 16
DATA_FORMAT = "NHWC"
TASK = Tasks.SEMANTIC_SEGMENTATION
CLASSES = DATASET_CLASS.classes

MAX_STEPS = 150000
SAVE_STEPS = 10000
TEST_STEPS = 2000
SUMMARISE_STEPS = 2000

# distributed training
IS_DISTRIBUTION = False

# pretrain
USE_PRETRAIN = True
PRETRAIN_FILE = "/storage/neil/saved/train_log/imagenet-qresnet-w64-batch256-nomaxpool-reduced1stlayer3x3/model-525420"

PRE_PROCESSOR = Sequence([
    Resize(size=IMAGE_SIZE),
    # DivideBy255(),
    PerImageStandardization(),
])
POST_PROCESSOR = None


NETWORK = EasyDict()
NETWORK.OPTIMIZER_CLASS = tf.train.AdamOptimizer
NETWORK.OPTIMIZER_KWARGS = {}
NETWORK.LEARNING_RATE_FUNC = tf.train.piecewise_constant
NETWORK.LEARNING_RATE_KWARGS = {
    "values": [0.001, 0.0009, 0.0001],
    "boundaries": [37000, 57000],
}
# NETWORK.OPTIMIZER_CLASS = tf.train.RMSPropOptimizer
# NETWORK.OPTIMIZER_KWARGS = {"momentum": 0.9}
# NETWORK.LEARNING_RATE_FUNC = tf.train.polynomial_decay
# NETWORK.LEARNING_RATE_KWARGS = {
#     'learning_rate': 0.0025,
#     'power': 0.9,
#     'decay_steps': MAX_STEPS,
# }
NETWORK.IMAGE_SIZE = IMAGE_SIZE
NETWORK.BATCH_SIZE = BATCH_SIZE
NETWORK.DATA_FORMAT = DATA_FORMAT
NETWORK.WEIGHT_DECAY_RATE = 1e-4

DATASET = EasyDict()
DATASET.BATCH_SIZE = BATCH_SIZE
DATASET.DATA_FORMAT = DATA_FORMAT
DATASET.PRE_PROCESSOR = PRE_PROCESSOR
DATASET.AUGMENTOR = Sequence([
    RandomResize(min_scale=0.25, max_scale=1.75),
    CropOrPad(size=IMAGE_SIZE),
    # RandomCrop(crop_size=IMAGE_SIZE),
    Hue((-10, 10)),
    Brightness((0.75, 1.25)),
    Color((0.75, 1.25)),
    Contrast((0.75, 1.25)),
    FlipLeftRight(),
    # FlipTopBottom(),
])
DATASET.ENABLE_PREFETCH = True
