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
from PIL import Image
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
        bev_image = self.transform_bev_image(roi_pts, data_num)

        return bev_image

    def load_bev_label(self, data_num):
        txt_path = os.path.join(self.data_path, "object/training/label", data_num + '.txt')
        if not os.path.exists(txt_path):
            raise KeyError("%s does not exist ... " %txt_path)

        label = self.load_label(txt_path)
        bev_label = self.transform_bev_label(label, data_num)

        return bev_label

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

    def scale_to_255(self, a, min, max, dtype=np.uint8):
        return (((a - min) / float(max - min)) * 255).astype(dtype)

    def transform_bev_image(self, pts, data_num):
        x_points = pts[:, 0]
        y_points = pts[:, 1]
        z_points = pts[:, 2]
        i_points = pts[:, 3]

        # convert to pixel position values
        x_img = (-y_points / self.voxel_size).astype(np.int32)  # x axis is -y in LIDAR
        y_img = (-x_points / self.voxel_size).astype(np.int32)  # y axis is -x in LIDAR

        # shift pixels to (0, 0)
        x_img -= int(np.floor(self.y_min / self.voxel_size))
        y_img += int(np.ceil(self.x_max / self.voxel_size))

        # clip height value
        pixel_values = np.clip(a=z_points, a_min=self.z_min, a_max=self.z_max)

        # rescale the height values
        pixel_values = self.scale_to_255(pixel_values, min=self.z_min, max=self.z_max)

        # initalize empty array
        x_max = 1 + math.ceil((self.y_max - self.y_min) / self.voxel_size)
        y_max = 1 + math.ceil((self.x_max - self.x_min) / self.voxel_size)

        # Height Map
        height_map = np.zeros((y_max, x_max))
        height_map[y_img, x_img] = pixel_values

        # save bev image
        image = Image.fromarray(height_map)
        image = image.convert('L')
        image.save('/home/chenwei/HDD/livox_dl/LIVOX/bev_image/' + data_num + ".bmp")

        # Intensity Map
        intensity_map = np.zeros((y_max, x_max))
        intensity_map[y_img, x_img] = i_points

        rgb_map = np.zeros((y_max, x_max, 2))
        rgb_map[:, :, 0] = height_map  # g_map
        rgb_map[:, :, 1] = intensity_map  # b_map

        return rgb_map

    def load_label(self, label_path):
        lines = [line.rstrip() for line in open(label_path)]
        label_list = []
        for line in lines:
            data = line.split(' ')
            data[4:] = [float(t) for t in data[4:]]
            type = data[1]
            if type not in classes_map.keys():
                continue
            label = np.zeros([8], dtype=np.float32)
            label[0] = self.cls_type_to_id(data)
            label[1], label[2], label[3] = self.calc_xyz(data)
            label[4], label[5], label[6] = self.calc_hwl(data)
            label[7] = self.calc_yaw(data)
            label_list.append(label)
        return np.array(label_list)

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

    def transform_bev_label(self, label, data_num):
        image_width = (self.y_max - self.y_min) / self.voxel_size
        image_height = (self.x_max - self.x_min) / self.voxel_size

        boxes_list = []
        boxes_num = label.shape[0]

        txt_path = "/home/chenwei/HDD/livox_dl/LIVOX/bev_label/" + data_num + '.txt'
        f = open(txt_path, mode='w')

        for i in range(boxes_num):
            center_x = (-label[i][2] / self.voxel_size).astype(np.int32) - int(np.floor(self.y_min / self.voxel_size))
            center_y = (-label[i][1] / self.voxel_size).astype(np.int32) + int(np.ceil(self.x_max / self.voxel_size))
            width = label[i][5] / self.voxel_size
            height = label[i][6] / self.voxel_size

            left = center_x - width / 2
            right = center_x + width / 2
            top = center_y - height / 2
            bottom = center_y + height / 2
            if((left > image_width) or right < 0 or (top > image_height) or bottom < 0):
                continue
            if(left < 0):
                center_x = (0 + right) / 2
                width = 0 + right
            if(right > image_width):
                center_x = (image_width + left) / 2
                width = image_width - left
            if(top < 0):
                center_y = (0 + bottom) / 2
                height = 0 + bottom
            if(bottom > image_height):
                center_y = (top + image_height) / 2
                height = image_height - top

            box = np.zeros([6], dtype=np.float32)
            box[0] = center_x
            box[1] = center_y
            box[2] = width
            box[3] = height
            box[4] = label[i][0]
            box[5] = label[i][7]
            boxes_list.append(box)

            for k in range(6):
                f.write(str(box[k]) + " ")
            f.write("\n")
        f.close()

        return np.array(boxes_list)

    def preprocess_true_boxes(self, labels):
        """
        preprocess true boxes to train input format
        :param labels: numpy.ndarray of shape [num, 6]
                       shape[0]: the number of labels in each image.
                       shape[1]: x_min, y_min, x_max, y_max, class_index, yaw
        :return: y_true shape is [feature_height, feature_width, per_anchor_num, 7 + num_classes]
        """
        # class id must be less than num_classes
        #assert (labels[..., 4] < self.class_num).all()

        anchor_array = np.array(self.anchors)

        input_shape = np.array([self.input_height, self.input_width], dtype=np.int32)
        num_layers = len(self.anchors) // 2
        anchor_mask = [[4, 5], [2, 3], [0, 1]]
        feature_map_sizes = [input_shape // 32, input_shape // 16, input_shape // 8]

        y_true_13 = np.zeros(shape=[feature_map_sizes[0][0], feature_map_sizes[0][1], 3, 7 + self.class_num], dtype=np.float32)
        y_true_26 = np.zeros(shape=[feature_map_sizes[1][0], feature_map_sizes[1][1], 3, 7 + self.class_num], dtype=np.float32)
        y_true_52 = np.zeros(shape=[feature_map_sizes[2][0], feature_map_sizes[2][1], 3, 7 + self.class_num], dtype=np.float32)
        y_true = [y_true_13, y_true_26, y_true_52]

        boxes_xy = labels[:, 0:2]    # 中心点坐标
        boxes_wh = labels[:, 2:4]    # 宽，高
        boxes_yaw = labels[:, 5:6]   # yaw
        boxes_re = np.cos(boxes_yaw)
        boxes_im = np.sin(boxes_yaw)
        true_boxes = np.concatenate([boxes_xy, boxes_wh, boxes_re, boxes_im], axis=-1)

        anchors_max = anchor_array / 2.
        anchors_min = - anchor_array / 2.
        valid_mask = boxes_wh[:, 0] > 0
        wh = boxes_wh[valid_mask]

        # [N, 1, 2]
        wh = np.expand_dims(wh, -2)
        boxes_max = wh / 2.
        boxes_min = - wh / 2.

        # [N, 1, 2] & [6, 2] ==> [N, 6, 2]
        intersect_mins = np.maximum(boxes_min, anchors_min)
        intersect_maxs = np.minimum(boxes_max, anchors_max)
        # [N, 6, 2]
        intersect_wh = np.maximum(intersect_maxs - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchor_array[:, 0] * anchor_array[:, 1]
        # [N, 6]
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
                y_true[l][j, i, k, 4:6] = true_boxes[t, 4:6]
                y_true[l][j, i, k, 6] = 1
                y_true[l][j, i, k, 7 + c] = 1

        return y_true_13, y_true_26, y_true_52

if __name__ == '__main__':
    image_path = '/home/chenwei/HDD/livox_dl/LIVOX/bev_image/000501.bmp'
    label_path = '/home/chenwei/HDD/livox_dl/LIVOX/bev_label/000501.txt'

    lines = [line.rstrip() for line in open(label_path)]
    label_list = []
    for line in lines:
        data = line.split(' ')
        data[0:] = [float(t) for t in data[0:]]
        x_min = int(data[0] - data[2] / 2)
        y_min = int(data[1] - data[3] / 2)
        x_max = int(data[0] + data[2] / 2)
        y_max = int(data[1] + data[3] / 2)

        label_list.append([x_min, y_min, x_max, y_max])

    image = cv2.imread(image_path, 0)
    for label in label_list:
        cv2.rectangle(image, (label[0], label[1]), (label[2], label[3]), (255, 0, 0), 2)
    cv2.imshow("image", image)
    cv2.waitKey(0)
