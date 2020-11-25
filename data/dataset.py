# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : dataset.py
# Description :preprocess data
# --------------------------------------

import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
import os
import math
import numpy as np
import tensorflow as tf
from xml.etree import ElementTree as ET
from cfg.config import path_params, model_params, data_params, classes_map
from utils.process_utils import *

class Dataset(object):
    def __init__(self):
        self.data_path = path_params['data_path']
        self.input_height = model_params['input_height']
        self.input_width = model_params['input_width']
        self.iou_threshold = model_params['iou_threshold']
        self.strides = model_params['strides']
        self.anchors = model_params['anchors']
        self.anchor_per_scale = model_params['anchor_per_scale']
        self.class_num = len(model_params['classes'])
        self.max_bbox_per_scale = model_params['max_bbox_per_scale']
        self.feature_map_sizes = [np.array([self.input_height, self.input_width]) // stride for stride in self.strides]

        self.x_min = data_params['x_min']
        self.x_max = data_params['x_max']
        self.y_min = data_params['y_min']
        self.y_max = data_params['y_max']
        self.z_min = data_params['z_min']
        self.z_max = data_params['z_max']
        self.voxel_size = data_params['voxel_size']

    def load_bev_image(self, data_num):
        pcd_path = os.path.join(self.data_path, "object/training/livox", data_num+'.pcd')
        if not os.path.exists(pcd_path):
            raise KeyError("%s does not exist ... " %pcd_path)

        pts = self.load_pcd(pcd_path)
        roi_pts = self.filter_roi(pts)
        bev_image = self.transform_bev_image(roi_pts)

        return bev_image

    def load_bev_label(self, data_num):
        txt_path = os.path.join(self.data_path, "object/training/label", data_num + '.txt')
        if not os.path.exists(txt_path):
            raise KeyError("%s does not exist ... " %txt_path)

        label = self.load_label(txt_path)
        bev_label = self.transform_bev_label(label)
        encoded_label = self.encode(bev_label)

        return encoded_label

    def load_pcd(self, pcd_path):
        pts = []
        f = open(pcd_path, 'r')
        data = f.readlines()
        f.close()

        line = data[9].strip('\n')
        pts_num = eval(line.split(' ')[-1])

        for line in data[11:]:
            line = line.strip('\n')
            xyzi = line.split(' ')
            x, y, z, i = [eval(i) for i in xyzi[:4]]
            pts.append([x, y, z, i])

        assert len(pts) == pts_num
        res = np.zeros((pts_num, len(pts[0])), dtype=np.float)
        for i in range(pts_num):
            res[i] = pts[i]

        return res

    def filter_roi(self, pts):
        mask = np.where((pts[:, 0] >= self.x_min) & (pts[:, 0] <= self.x_max) &
                        (pts[:, 1] >= self.y_min) & (pts[:, 1] <= self.y_max) &
                        (pts[:, 2] >= self.z_min) & (pts[:, 2] <= self.z_max))
        pts = pts[mask]

        return pts

    def transform_bev_image(self, pts):
        bev_height = (self.x_max - self.x_min) / self.voxel_size
        bev_width = (self.y_max - self.y_min) / self.voxel_size

        range_x = data_params['x_max'] - data_params['x_min']
        range_y = data_params['y_max'] - data_params['y_min']

        # Discretize Feature Map
        point_cloud = np.copy(pts)
        point_cloud[:, 0] = np.int_(np.floor(point_cloud[:, 0] / range_x * (bev_height - 1)))
        point_cloud[:, 1] = np.int_(np.floor(point_cloud[:, 1] / range_y * (bev_width - 1)) + bev_width / 2)

        # sort-3times
        indices = np.lexsort((-point_cloud[:, 2], point_cloud[:, 1], point_cloud[:, 0]))
        point_cloud = point_cloud[indices]

        # Height Map
        height_map = np.zeros((bev_height, bev_width))

        _, indices = np.unique(point_cloud[:, 0:2], axis=0, return_index=True)
        point_cloud_frac = point_cloud[indices]

        # some important problem is image coordinate is (y,x), not (x,y)
        max_height = float(np.abs(data_params['z_max'] - data_params['z_min']))
        height_map[np.int_(point_cloud_frac[:, 0]), np.int_(point_cloud_frac[:, 1])] = point_cloud_frac[:,
                                                                                       2] / max_height

        # Intensity Map & DensityMap
        intensity_map = np.zeros((bev_height, bev_width))
        density_map = np.zeros((bev_height, bev_width))

        _, indices, counts = np.unique(point_cloud[:, 0:2],
                                       axis=0,
                                       return_index=True,
                                       return_counts=True)

        point_cloud_top = point_cloud[indices]
        normalized_counts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
        intensity_map[np.int_(point_cloud_top[:, 0]), np.int_(point_cloud_top[:, 1])] = point_cloud_top[:, 3]
        density_map[np.int_(point_cloud_top[:, 0]), np.int_(point_cloud_top[:, 1])] = normalized_counts

        rgb_map = np.zeros((bev_height, bev_width, 3))
        rgb_map[:, :, 0] = density_map  # r_map
        rgb_map[:, :, 1] = height_map  # g_map
        rgb_map[:, :, 2] = intensity_map  # b_map

        return rgb_map

    def load_label(self, label_path):
        lines = [line.rstrip() for line in open(label_path)]
        num_obj = len(lines)

        index = 0
        label = np.zeros([num_obj, (6 + 1 + 1)], dtype=np.float32)
        for line in lines:
            data = line.split(' ')
            data[4:] = [float(t) for t in data[4:]]
            label[index, 0] = self.cls_type_to_id(data)
            label[index, 1], label[index, 2], label[index, 3] = self.calc_xyz(data)
            label[index, 4], label[index, 5], label[index, 6] = self.calc_hwl(data)
            label[index, 7] = self.calc_yaw(data)
            index += 1

        return label

    def cls_type_to_id(self, data):
        type = data[1]
        if type not in classes_map.keys():
            return -1
        return classes_map[type]

    def calc_xyz(self, data):
        center_x = (data[16] + data[19] + data[22] + data[25]) / 4.0
        center_y = (data[17] + data[20] + data[23] + data[26]) / 4.0
        center_z = (data[18] + data[21] + data[24] + data[27]) / 4.0
        return center_x, center_y, center_z

    def calc_hwl(self, data):
        height = (data[15] - data[27])
        width = math.sqrt(math.pow((data[17] - data[26]), 2) + math.pow((data[16] - data[25]), 2))
        length = math.sqrt(math.pow((data[17] - data[20]), 2) + math.pow((data[16] - data[19]), 2))
        return height, width, length

    def calc_yaw(self, data):
        angle = math.atan2(data[17] - data[26], data[16] - data[25])

        if (angle < -1.57):
            return angle + 3.14 * 1.5
        else:
            return angle - 1.57

    def preprocess_true_boxes(self, labels, input_height, input_width, anchors, num_classes):
        """
        preprocess true boxes to train input format
        :param labels: numpy.ndarray of shape [num, 5]
                       shape[0]: the number of labels in each image.
                       shape[1]: x_min, y_min, x_max, y_max, class_index
        :param input_height: the shape of input image height
        :param input_width: the shape of input image width
        :param anchors: array, shape=[9, 2]
                        shape[0]: the number of anchors
                        shape[1]: width, height
        :param num_classes: the number of class
        :return: y_true shape is [feature_height, feature_width, per_anchor_num, 5 + num_classes]
        """
        # class id must be less than num_classes
        assert (labels[..., 4] < num_classes).all()

        input_shape = np.array([input_height, input_width], dtype=np.int32)
        num_layers = len(anchors) // 3
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        feature_map_sizes = [input_shape // 32, input_shape // 16, input_shape // 8]

        y_true_13 = np.zeros(shape=[feature_map_sizes[0][0], feature_map_sizes[0][1], 3, 5 + num_classes], dtype=np.float32)
        y_true_26 = np.zeros(shape=[feature_map_sizes[1][0], feature_map_sizes[1][1], 3, 5 + num_classes], dtype=np.float32)
        y_true_52 = np.zeros(shape=[feature_map_sizes[2][0], feature_map_sizes[2][1], 3, 5 + num_classes], dtype=np.float32)
        y_true = [y_true_13, y_true_26, y_true_52]

        # convert boxes from (min_x, min_y, max_x, max_y) to (x, y, w, h)
        boxes_xy = (labels[:, 0:2] + labels[:, 2:4]) / 2    # 中心点坐标
        boxes_wh = labels[:, 2:4] - labels[:, 0:2]          # 宽，高
        true_boxes = np.concatenate([boxes_xy, boxes_wh], axis=-1)

        anchors_max = anchors / 2.
        anchors_min = - anchors / 2.
        valid_mask = boxes_wh[:, 0] > 0
        wh = boxes_wh[valid_mask]

        # [N, 1, 2]
        wh = np.expand_dims(wh, -2)
        boxes_max = wh / 2.
        boxes_min = - wh / 2.

        # [N, 1, 2] & [9, 2] ==> [N, 9, 2]
        intersect_mins = np.maximum(boxes_min, anchors_min)
        intersect_maxs = np.minimum(boxes_max, anchors_max)
        # [N, 9, 2]
        intersect_wh = np.maximum(intersect_maxs - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[:, 0] * anchors[:, 1]
        # [N, 9]
        iou = intersect_area / (box_area + anchor_area - intersect_area + tf.keras.backend.epsilon())

        # Find best anchor for each true box [N]
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n not in anchor_mask[l]: continue
                i = np.floor(true_boxes[t, 0] / input_shape[0] * feature_map_sizes[l][0]).astype('int32')
                j = np.floor(true_boxes[t, 1] / input_shape[1] * feature_map_sizes[l][1]).astype('int32')
                k = anchor_mask[l].index(n)
                c = labels[t][4].astype('int32')
                y_true[l][j, i, k, 0:4] = true_boxes[t, 0:4]
                y_true[l][j, i, k, 4] = 1
                y_true[l][j, i, k, 5 + c] = 1

        return y_true_13, y_true_26, y_true_52