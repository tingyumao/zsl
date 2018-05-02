import os
import sys
import glob
import random
import math
import datetime
import itertools
import json
import re
import logging
from collections import OrderedDict
import numpy as np
import scipy.misc
import cv2
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.initializers as KI
import keras.engine as KE
import keras.models as KM

import utils
from module import *

###########################################
# RCNN Models
###########################################


def yolov3_image_feature_graph(input_image):
    # feature extractor: darknet53
    darknet_feature = darknet_graph(input_image, architecture='darknet53')
    feature_map = KL.Conv2D(1024, (1, 1), strides=(1, 1), padding="SAME", name='darknet_output_conv', use_bias=True)(darknet_feature)

    return feature_map


def yolo_predict(detection, anchors):
    pass


def yolo_loss(detection, target_object, target_bbox, target_bbox_object):
    """

    :param detection: None, 192, 5
    :param target_bbox: None, 192, 4
    :param target_object: None, 192
    :return:
    """
    to = detection[..., 0]
    ty = K.expand_dims(detection[..., 1], axis=-1)
    tx = K.expand_dims(detection[..., 2], axis=-1)
    th = K.expand_dims(detection[..., 3], axis=-1)
    tw = K.expand_dims(detection[..., 4], axis=-1)

    to = KL.Activation('sigmoid')(to)
    ty = KL.Activation('sigmoid')(ty)
    tx = KL.Activation('sigmoid')(tx)

    detection = KL.Concatenate(axis=-1)([ty, tx, th, tw])

    # sum of squared error loss
    object_loss = K.mean(K.binary_crossentropy(to, target_object))
    bbox_loss = K.sum(K.sum(K.square(detection - target_bbox), axis=-1) * target_bbox_object)

    loss = object_loss + bbox_loss

    return loss


#####################################################################
# Data Generator
#####################################################################


def data_generator(dataset, config, shuffle=True, augment=True, random_rois=0,
                   batch_size=1, detection_targets=False):
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0

    anchors = []
    for i, scale in enumerate(config.ANCHOR_SCALES):
        anchors.append(utils.generate_anchors(scales=scale, ratios=config.ANCHOR_RATIOS, shape=[8*2**i, 8*2**i],
                                              feature_stride=config.FEATURE_STRIDE/(2**i), anchor_stride=1))
    anchors = np.concatenate(anchors, axis=0)

    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]
            image, image_shape = dataset.load_image(image_id)
            image_bbox, image_class_id, attribute, class_attribute = dataset.load_bbox_class_attr(image_id)

            image_bbox = translate_bbox(image_bbox, input_shape=image_shape[:2], output_shape=config.IMAGE_SHAPE[:2])
            image = cv2.resize(image, tuple(config.IMAGE_SHAPE[:2]))
            gt_target_object, gt_target_bbox, gt_target_bbox_object = assign_bbox_to_anchors(config, image_bbox, anchors, image_id)

            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            # if not np.any(gt_class_ids > 0):
            #     continue

            # Init batch arrays
            if b == 0:
                batch_images = np.zeros((batch_size,) + image.shape, dtype=np.float32)
                batch_gt_class_ids = np.zeros((batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                batch_gt_object = np.zeros((batch_size, anchors.shape[0]), dtype=np.int32)
                batch_gt_boxes = np.zeros((batch_size, anchors.shape[0], 4), dtype=np.float32)
                batch_gt_boxes_object = np.zeros((batch_size, anchors.shape[0]), dtype=np.int32)
                batch_gt_attribute = np.zeros((batch_size, config.NUM_ATTRIBUTE))

            # Add to batch
            batch_images[b] = image.astype(np.float32)  # mold_image(image.astype(np.float32), config)
            batch_gt_class_ids[b, 0] = image_class_id
            batch_gt_object[b] = gt_target_object
            batch_gt_boxes[b, :gt_target_bbox.shape[0]] = gt_target_bbox
            batch_gt_boxes_object[b] = gt_target_bbox_object
            batch_gt_attribute[b] = attribute

            b += 1

            # Batch full?
            if b >= batch_size:
                inputs = [batch_images, batch_gt_attribute, batch_gt_object, batch_gt_boxes, batch_gt_boxes_object]
                outputs = []
                yield inputs, outputs

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(
                dataset.print_image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise


def translate_bbox(bbox, input_shape, output_shape):
    x1, y1, x2, y2 = bbox

    iw, ih = input_shape
    ow, oh = output_shape

    x1 = float(x1)*ow/iw
    y1 = float(y1)*oh/ih
    x2 = float(x2)*ow/iw
    y2 = float(y2)*oh/ih

    return x1, y1, x2, y2


def assign_bbox_to_anchors(config, image_bbox, anchors, image_id):
    feature_stride = float(config.FEATURE_STRIDE)

    gt_target_object = np.zeros(anchors.shape[0]).astype("float32")
    gt_target_bbox = np.zeros(anchors.shape).astype("float32")
    gt_target_bbox_object = np.zeros(anchors.shape[0]).astype("float32")

    y1, x1, y2, x2 = image_bbox
    assert x2 > x1 and y2 > y1
    x, y, w, h = x1, y1, x2 - x1, y2 - y1
    xc, yc = x + w / 2, y + h / 2

    image_bbox = [x1, y1, x2, y2]

    cnt = 0
    iou_test = []
    max_iou = 0.0
    max_t = []
    max_index = 0
    for i, anchor in enumerate(anchors):
        py, px, ph, pw = anchor
        py += feature_stride / 2
        px += feature_stride / 2
        px1, py1, px2, py2 = px - pw / 2, py - ph / 2, px + pw / 2, py + ph / 2

        if iou(image_bbox, [px1, py1, px2, py2]) > 0.5:
            # calculate tx, ty, tw, th
            tx = (xc - px + feature_stride / 2) / feature_stride
            ty = (yc - py + feature_stride / 2) / feature_stride
            tw = np.log(w / float(pw))
            th = np.log(h / float(ph))

            iou_test.append(iou(image_bbox, [px1, py1, px2, py2]))
            gt_target_object[i] = 1.0
            cnt += 1

            if 0 <= tx <= 1 and 0 <= ty <= 1:
                if iou_test[-1] > max_iou:
                    max_t = [tx, ty, tw, th]
                    max_index = i
                    max_iou = iou_test[-1]

    assert cnt > 0, print(image_id)
    assert max_t != list(), print(image_id)

    gt_target_bbox_object[max_index] = 1.0
    gt_target_bbox[max_index] = max_t

    return gt_target_object, gt_target_bbox, gt_target_bbox_object


def iou(bb, bbgt):
    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
    iw = bi[2] - bi[0] + 1
    ih = bi[3] - bi[1] + 1

    if iw > 0 and ih > 0:
        # compute overlap as area of intersection / area of union
        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
        ov = iw * ih / ua
    else:
        ov = 0.0

    return ov

###########################################
# ZSL Model
###########################################
class ZSL():
    """
    The design of ZSL is referred to MaskRCNN (https://github.com/matterport/Mask_RCNN).
    """
    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):

        # Define the input layers
        input_image = KL.Input(shape=config.IMAGE_SHAPE, name="input_image")
        input_attribute = KL.Input(shape=[config.NUM_ATTRIBUTE], name="input_features")

        # darknet53
        S1, S2, S3, S4, S5 = darknet_graph(input_image, architecture='darknet53')

        def output_detection_layers(x, num_filters, out_filters, name):
            for i in range(2):
                conv_name_base = "last_conv_" + name + "_" + str(i)
                x = KL.Conv2D(num_filters, (1, 1), padding="SAME", name=conv_name_base + '_a', use_bias=True)(x)
                # x = BatchNorm(axis=3, name=bn_name_base + 'b')(x)
                x = KL.LeakyReLU(alpha=0.1)(x)

                x = KL.Conv2D(num_filters*2, (3, 3), padding="SAME", name=conv_name_base + '_b', use_bias=True)(x)
                # x = BatchNorm(axis=3, name=bn_name_base + 'b')(x)
                x = KL.LeakyReLU(alpha=0.1)(x)
            x = KL.Conv2D(num_filters, (1, 1), padding="SAME", name=conv_name_base + "_out", use_bias=True)(x)
            # x = BatchNorm(axis=3, name=bn_name_base + 'b')(x)
            x = KL.LeakyReLU(alpha=0.1)(x)

            conv_name_base = "detection_head_" + name
            y = KL.Conv2D(num_filters*2, (3, 3), padding="SAME", name=conv_name_base + 'a', use_bias=True)(x)
            # x = BatchNorm(axis=3, name=bn_name_base + 'b')(x)
            y = KL.LeakyReLU(alpha=0.1)(y)
            y = KL.Conv2D(out_filters, (1, 1), padding="SAME", name=conv_name_base + 'b', use_bias=True)(y)
            # x = BatchNorm(axis=3, name=bn_name_base + 'b')(x)
            y = KL.LeakyReLU(alpha=0.1)(y)

            return x, y

        # FPN: top to bottom
        # stage 1
        x, y1 = output_detection_layers(S5, num_filters=512, out_filters=len(config.ANCHOR_SCALES[0]) * 3 * (4 + 1), name="fpn1")

        # stage 2
        # x = KL.Conv2D(256, (1, 1), padding="SAME", name="last", use_bias=True)(x)
        # # x = BatchNorm(axis=3, name=bn_name_base + 'b')(x)
        # x = KL.LeakyReLU(alpha=0.1)(x)
        x = KL.UpSampling2D(2)(x)
        x = KL.Concatenate()([x, S4])
        x, y2 = output_detection_layers(x, num_filters=256, out_filters=len(config.ANCHOR_SCALES[1]) * 3 * (4 + 1), name="fpn2")

        # stage 3
        # x = KL.Conv2D(128, (1, 1), padding="SAME", name="last", use_bias=True)(x)
        # # x = BatchNorm(axis=3, name=bn_name_base + 'b')(x)
        # x = KL.LeakyReLU(alpha=0.1)(x)
        x = KL.UpSampling2D(2)(x)
        x = KL.Concatenate()([x, S3])
        x, y3 = output_detection_layers(x, num_filters=128, out_filters=len(config.ANCHOR_SCALES[2]) * 3 * (4 + 1), name="fpn3")

        # detection: 3 * (4 + 1) for each anchor
        y1 = KL.Reshape((-1, 5))(y1)
        y2 = KL.Reshape((-1, 5))(y2)
        y3 = KL.Reshape((-1, 5))(y3)
        detection = KL.Concatenate(name="final_detection", axis=1)([y1, y2, y3])

        # image feature
        image_feature = x

        if mode == 'training':
            # Define the input layers for ground truth
            num_anchors = K.int_shape(detection)[0]
            gt_bbox = KL.Input(shape=[num_anchors, 4], name="ground_truth_bbox")
            gt_bbox_object = KL.Input(shape=[num_anchors], name="ground_truth_bbox_object")
            gt_object = KL.Input(shape=[num_anchors], name="ground_truth_object")

            # Define the loss
            loss = KL.Lambda(lambda x: yolo_loss(*x), name='yolo_loss')([detection, gt_object, gt_bbox, gt_bbox_object])

            # print(type(loss))
            return KM.Model([input_image, input_attribute, gt_object, gt_bbox, gt_bbox_object], [image_feature, detection, loss])

        if mode == 'inference':
            pass

    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=momentum,
                                         clipnorm=5.0)
        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = ["yolo_loss"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            self.keras_model.add_loss(
                tf.reduce_mean(layer.output, keep_dims=True))

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
                      for w in self.keras_model.trainable_weights
                      if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(optimizer=optimizer, loss=[None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            self.keras_model.metrics_tensors.append(tf.reduce_mean(
                layer.output, keep_dims=True))

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/yolo\_\w+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6)) + 1

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "yolo_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

    def train(self, train_dataset, val_dataset, learning_rate, epochs):
        """

        :param train_dataset:
        :param val_dataset:
        :param learning_rate:
        :param epochs:
        :param layers:
        :return:
        """
        assert self.mode == 'training'

        # Data generators
        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                         batch_size=self.config.BATCH_SIZE)
        val_generator = data_generator(val_dataset, self.config, shuffle=True,
                                       batch_size=self.config.BATCH_SIZE,
                                       augment=False)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=5, write_graph=True, write_images=True, write_grads=True),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Multi-thread programming will fail on Windows system.
        if os.name is 'nt':
            workers = 0
        else:
            workers = max(self.config.BATCH_SIZE // 2, 2)

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=next(val_generator),
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
        )
        self.epoch = max(self.epoch, epochs)
















