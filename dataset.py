"""
challenger ai zero-shot learning competition.
"""

import sys
import os
import math
import random
import numpy as np
import tensorflow as tf
import scipy.misc
import skimage.color
import skimage.io
import urllib.request
import shutil

import utils

############################
# Custom Data Set
############################

class CAIData(object):
    def __init__(self, root_dir, mode):
        assert mode in ["train", "validation"]
        self.mode = mode
        self.root_dir = root_dir

        self.image_dir = ""
        self.image_annotated_info = dict()
        self.image_nonannotated_info = dict()
        self.image_type_info = dict()

        self.num_class = 0
        self.class_list = list()
        self.class_dict = dict()

        self.num_attr = 0
        self.attr_info = dict()
        self.attr_class_dict = dict()
        self.attr_samples = dict()

        self.prepare()

        if self.mode == "train":
            self.image_ids = list(self.image_annotated_info.keys()) + sorted(list(self.image_nonannotated_info.keys()))[:-1000]
        else:
            self.image_ids = sorted(list(self.image_nonannotated_info.keys()))[-1000:]

    def prepare(self):
        image_dir = [x for x in os.listdir(self.root_dir) if x.startswith('zsl') and not x.endswith('.txt')]
        assert len(image_dir) == 1
        self.image_dir = image_dir[0]

        """
        Class part
        """
        class_file = [x for x in os.listdir(self.root_dir)
                     if x.endswith('train_annotations_label_list_20180321.txt')]
        assert len(class_file) == 1
        class_file = class_file[0]

        with open(os.path.join(self.root_dir, class_file), 'r') as f:
            for line in f:
                class_label, class_name, _ = line.split(',')
                self.class_list.append(class_label)
                self.class_dict[class_label] = class_name
        self.num_class = len(self.class_list)

        """
        Attribute part
        """
        attr_list = [x for x in os.listdir(self.root_dir)
                     if x.endswith('train_annotations_attribute_list_20180321.txt')]
        assert len(attr_list) == 1
        attr_list = attr_list[0]

        attr_annotations = [x for x in os.listdir(self.root_dir)
                            if x.endswith('train_annotations_attributes_20180321.txt')]
        assert len(attr_annotations) == 1
        attr_annotations = attr_annotations[0]

        attr_class = [x for x in os.listdir(self.root_dir)
                      if x.endswith('train_annotations_attributes_per_class_20180321.txt')]
        assert len(attr_class) == 1
        attr_class = attr_class[0]

        # load the detailed information about attributes
        self.num_attr = 0
        with open(os.path.join(self.root_dir, attr_list), 'r') as f:
            for line in f:
                attr_name, attr_info = line.split(",")[:2]
                self.attr_info[attr_name] = attr_info
                self.num_attr += 1
        # load the attribute annotation for those labeled training samples
        with open(os.path.join(self.root_dir, attr_annotations), 'r') as f:
            for line in f:
                image_class_label, image_path, attr = line.split(',')
                attr = [float(x) for x in attr.split(',')[-1].split('[')[-1].split(']')[0].split(' ')[1:-1]]
                assert len(attr) == self.num_attr, print(attr)
                self.attr_samples[image_path] = np.array(attr).astype('float32')

        # load the average attribute for each class
        with open(os.path.join(self.root_dir, attr_class), 'r') as f:
            for line in f:
                image_class_label, attr = line.split(',')
                attr = [float(x) for x in attr.split(',')[-1].split('[')[-1].split(']')[0].split(' ')[1:-1]]
                assert len(attr) == self.num_attr
                self.attr_class_dict[image_class_label] = np.array(attr).astype('float32')

        """
        Image part
        """
        image_annotations = [x for x in os.listdir(self.root_dir) if
                      x.endswith('train_annotations_labels_20180321.txt')]
        assert len(image_annotations) == 1
        image_annotations = image_annotations[0]

        # load each image info
        with open(os.path.join(self.root_dir, image_annotations), 'r') as f:
            for line in f:
                data = line.split(', ')
                image_id = int(data[0])
                image_class_label = data[1]
                x1, y1, x2, y2 = float(data[2].replace('[', '')), float(data[3]),\
                                 float(data[4]), float(data[5].replace(']', ''))
                bbox = [x1, y1, x2, y2]
                image_path = data[-1].replace("\n", "")
                if image_path in self.attr_samples:
                    attr_type = "self"
                    self.image_annotated_info[image_id] = [image_path, image_class_label, bbox]
                else:
                    attr_type = "class"
                    self.image_nonannotated_info[image_id] = [image_path, image_class_label, bbox]
                self.image_type_info[image_id] = attr_type

    def load_image(self, image_id):
        """

        :param image_id:
        :return: image, image shape (different raw image has different size.)
        """
        attr_type = self.image_type_info[image_id]
        if attr_type == "self":
            path, _, _ = self.image_annotated_info[image_id] # path, image_class_label, bbox
        else:
            path, _, _ = self.image_nonannotated_info[image_id]  # path, image_class_label, bbox
        image = skimage.io.imread(os.path.join(self.root_dir, self.image_dir, path))
        image_shape = image.shape
        return image, image_shape

    def load_bbox_class_attr(self, image_id):
        """

        :param image_id:
        :return: bbox, class_index, attribute, attr_type["self", "class"]
        """
        attr_type = self.image_type_info[image_id]
        if attr_type == "self":
            path, image_class_label, bbox = self.image_annotated_info[image_id]  # path, image_class_label, bbox
        else:
            path, image_class_label, bbox = self.image_nonannotated_info[image_id]  # path, image_class_label, bbox

        if attr_type == "self":
            attr = self.attr_samples[path].astype(np.float32)
        else:
            attr = self.attr_class_dict[image_class_label]

        class_attr = self.attr_class_dict[image_class_label]

        return bbox, self.class_list.index(image_class_label), attr, class_attr

    def print_image_info(self, image_id):
        """

        :param image_id:
        :return:
        """
        attr_type = self.image_type_info[image_id]
        if attr_type == "self":
            path, image_class_label, bbox = self.image_annotated_info[image_id]  # path, image_class_label, bbox
        else:
            path, image_class_label, bbox = self.image_nonannotated_info[image_id]  # path, image_class_label, bbox

        image_class = self.class_dict[image_class_label]

        print("ImageID: {}, Image {} belongs to class {} and bbox is {}".format(image_id, path, image_class, bbox))

        if path in self.attr_class_dict:
            print("This sample has its own attribute annotation.")
        else:
            print("This sample doesn't have its own attribute annotation.")
