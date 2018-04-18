#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm
@file: DSB_train_shapes.py.py
@time: 2018/3/20 9:51
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random
import math
import re
import time
import numpy as np
from mrcnn.config import Config
import mrcnn.utils as utils
import mrcnn.model as modellib
from mrcnn.model import log
from PIL import Image

# Root directory of the project
ROOT_DIR = os.getcwd()
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
BASE_DIR = '/home/zsy/datasets/dsb/'
IMAGE_IDS = os.listdir(BASE_DIR + 'stage1_train')
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 classes
    USE_MINI_MASK = False
    # MINI_MASK_SHAPE=(8,8)

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)  # anchor side in pixels

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 200
    # ROI_POSITIVE_RATIO = 0.5

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 100

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50


config = ShapesConfig()
config.display()


class DSBDataset(utils.Dataset):
    def __init__(self, image_size, **kwargs):
        super(DSBDataset, self).__init__(**kwargs)

        self.image_size = image_size

        # image dict and mask dict
        self.images_list = []
        self.masks_list = []
        self.class_list = []

    def load_DSB(self, image_ids):
        self.add_class('shapes', 1, 'cell')
        for i, idx in enumerate(image_ids):
            self.add_image('shapes', idx, BASE_DIR + 'stage1_train/%s/images/%s.png' % (idx, idx))

        # # load image from dict to memory
        # for i, image_id in enumerate(image_ids):
        #     # images
        #     image = Image.open(self.image_info[i]['path'])
        #     image = image.convert('RGB')
        #     # 之前注释了resize
        #     image = image.resize(self.image_size)
        #     image = np.array(image).astype(np.uint8)
        #     self.images_list.append(image)
        #
        #     # masks
        #     mask_ids = os.listdir(BASE_DIR + 'stage1_train/%s/masks' % self.image_info[i]['id'])
        #     shape = (self.image_size[0], self.image_size[1], len(mask_ids))
        #     masks = np.zeros(shape=shape)
        #     class_ids = []
        #     for k, idx in enumerate(mask_ids):
        #         mask = Image.open(BASE_DIR + 'stage1_train/%s/masks/%s' % (self.image_info[i]['id'], idx))
        #         # 之前注释了resize
        #         mask = mask.resize(self.image_size)
        #         mask = np.array(mask)
        #         mask[mask > 0] = 1
        #         # 是不是写错了，还有这里没有去处mask的重叠
        #         # masks[:, :, 1] = mask
        #         masks[:, :, k] = mask
        #         class_ids.append(1)
        #     self.masks_list.append(masks)
        #     self.class_list.append(np.array(class_ids))

    def load_image(self, image_id):
        image = Image.open(self.image_info[image_id]['path'])
        image = image.convert('RGB')
        # 之前注释了resize
        image = image.resize(self.image_size)
        image = np.array(image).astype(np.uint8)
        return image
        # return self.images_list[image_id]

    def load_mask(self, image_id):
        mask_ids = os.listdir(BASE_DIR + 'stage1_train/%s/masks' % self.image_info[image_id]['id'])
        shape = (self.image_size[0], self.image_size[1], len(mask_ids))
        masks = np.zeros(shape=shape)
        class_ids = []
        for i, idx in enumerate(mask_ids):
            mask = Image.open(BASE_DIR + 'stage1_train/%s/masks/%s' % (self.image_info[image_id]['id'], idx))
            # 之前注释了resize
            mask = mask.resize(self.image_size)
            mask = np.array(mask)
            mask[mask > 0] = 1
            # 是不是写错了，还有这里没有去处mask的重叠
            # masks[:, :, 1] = mask
            masks[:, :, i] = mask
            class_ids.append(1)
        return masks, np.array(class_ids)
        # return self.masks_list[image_id], self.class_list[image_id]


class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def inference():
    if not os.path.exists('./result'):
        os.mkdir('./result')
    image_ids = os.listdir(BASE_DIR + 'stage1_test')
    images = []
    for idx in image_ids:
        image = Image.open(BASE_DIR + 'stage1_test/%s/images/%s.png' % (idx, idx))
        image = image.convert('RGB')  # 转换成RGB三通道模式
        images.append(np.array(image))

    inference_config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    model_path = model.find_last()[1]

    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    # results = model.detect(images, verbose=1)
    # for i, idx in enumerate(image_ids):
    #     if not os.path.exists('./result/%s' % idx):
    #         os.mkdir('./result/%s' % idx)
    #     masks = results['masks'][i]
    #     for j in range(masks.shape[-1]):
    #         # i换成j？
    #         # mask = masks[:, :, i]
    #         mask = masks[:, :, j]
    #         mask = mask * 255
    #         mask = Image.fromarray(mask)
    #         mask.save('./result/%s/%s_%d.png' % (idx, idx, j))
    # return results['masks']

    for image, idx in zip(images, image_ids):
        results = model.detect([image], verbose=1)
        r = results[0]
        if not os.path.exists('./result/%s' % idx):
            os.mkdir('./result/%s' % idx)
        masks = r['masks']
        for j in range(masks.shape[-1]):
            # i换成j？
            # mask = masks[:, :, i]
            mask = masks[:, :, j]
            mask = mask.astype(np.uint8)
            mask = mask * 255
            mask = Image.fromarray(mask)
            mask.save('./result/%s/%s_%d.png' % (idx, idx, j))


if __name__ == '__main__':
    dataset_train = DSBDataset(image_size=(512, 512))
    dataset_train.load_DSB(IMAGE_IDS)
    dataset_train.prepare()
    dataset_val = DSBDataset(image_size=(512, 512))
    dataset_val.load_DSB(IMAGE_IDS[500:])
    dataset_val.prepare()
    # print(dataset_train.image_ids)
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
    for i in range(2):
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=1,
                    layers='heads')
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=2,
                    layers='all')

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=20,
                layers='all')
    # inference()
