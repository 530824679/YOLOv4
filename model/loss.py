# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : loss.py
# Description :Yolo_v2 Loss损失函数.
# --------------------------------------

import math
import numpy as np
import tensorflow as tf
from cfg.config import model_params, solver_params

class Loss(object):
    def __init__(self):
        self.batch_size = solver_params['batch_size']
        self.anchor_per_scale = model_params['anchor_per_scale']
        self.class_num = len(model_params['classes'])
        self.iou_threshold = model_params['iou_threshold']
        self.label_smoothing = model_params['label_smoothing']

    def calc_loss(self, pred_conv, pred_bbox, label_bbox, scope='loss'):
        """
        :param pred_conv: [pred_sconv, pred_mconv, pred_lconv]. pred_conv_shape=[batch_size, conv_height, conv_width, anchor_per_scale, 7 + num_classes]
        :param pred_bbox: [pred_sbbox, pred_mbbox, pred_lbbox]. pred_bbox_shape=[batch_size, conv_height, conv_width, anchor_per_scale, 7 + num_classes]
        :param label_bbox: [label_sbbox, label_mbbox, label_lbbox].
        :return:
        """
        with tf.name_scope('smaller_box_loss'):
            loss_sbbox = self.loss_layer(pred_conv[0], pred_bbox[0], label_bbox[0])

        with tf.name_scope('medium_box_loss'):
            loss_mbbox = self.loss_layer(pred_conv[1], pred_bbox[1], label_bbox[1])

        with tf.name_scope('bigger_box_loss'):
            loss_lbbox = self.loss_layer(pred_conv[2], pred_bbox[2], label_bbox[2])

        with tf.name_scope('ciou_loss'):
            ciou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]

        with tf.name_scope('reim_loss'):
            reim_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]

        with tf.name_scope('conf_loss'):
            conf_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]

        with tf.name_scope('prob_loss'):
            prob_loss = loss_sbbox[3] + loss_mbbox[3] + loss_lbbox[3]

        with tf.name_scope('rec_50'):
            rec_50 = loss_sbbox[4] + loss_mbbox[4] + loss_lbbox[4]

        with tf.name_scope('rec_75'):
            rec_75 = loss_sbbox[5] + loss_mbbox[5] + loss_lbbox[5]

        with tf.name_scope('avg_iou'):
            avg_iou = loss_sbbox[6] + loss_mbbox[6] + loss_lbbox[6]

        total_loss = 0.0
        total_loss = ciou_loss + reim_loss + conf_loss + prob_loss

        return total_loss, ciou_loss, reim_loss, conf_loss, prob_loss, rec_50, rec_75, avg_iou

    def loss_layer(self, pred_feat, pred_bbox, y_true):
        feature_shape = tf.shape(pred_feat)[1:3]
        predicts = tf.reshape(pred_feat, [-1, feature_shape[0], feature_shape[1], self.anchor_per_scale, (7 + self.class_num)])
        conv_conf = predicts[:, :, :, :, 6:7]
        conv_prob = predicts[:, :, :, :, 7:]

        pred_xywh = pred_bbox[:, :, :, :, 0:4]
        pred_reim = pred_bbox[:, :, :, :, 4:6]
        pred_conf = pred_bbox[:, :, :, :, 6:7]
        pred_class = tf.argmax(pred_bbox[:, :, :, :, 7:], axis=-1)

        label_xywh = y_true[:, :, :, :, 0:4]
        label_reim = y_true[:, :, :, :, 4:6]
        object_mask = y_true[:, :, :, :, 6:7]
        label_prob = self.smooth_labels(y_true[:, :, :, :, 7:], self.label_smoothing)
        label_class = tf.argmax(y_true[:, :, :, :, 7:], axis=-1)

        """
        compare online statistics
        """
        true_mins = label_xywh[..., 0:2] - label_xywh[..., 2:4] / 2.
        true_maxs = label_xywh[..., 0:2] + label_xywh[..., 2:4] / 2.
        pred_mins = pred_xywh[..., 0:2] - pred_xywh[..., 2:4] / 2.
        pred_maxs = pred_xywh[..., 0:2] + pred_xywh[..., 2:4] / 2.

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxs = tf.minimum(pred_maxs, true_maxs)

        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_area = label_xywh[..., 2] * label_xywh[..., 3]
        pred_area = pred_xywh[..., 2] * pred_xywh[..., 3]

        union_area = pred_area + true_area - intersect_area
        iou_scores = tf.truediv(intersect_area, union_area)

        iou_scores = object_mask * tf.expand_dims(iou_scores, 4)

        count = tf.reduce_sum(object_mask)
        detect_mask = tf.to_float((pred_conf * object_mask) >= 0.5)
        class_mask = tf.expand_dims(tf.to_float(tf.equal(pred_class, label_class)), 4)
        recall50 = tf.reduce_mean(tf.to_float(iou_scores >= 0.5) * detect_mask * class_mask) / (count + 1e-3)
        recall75 = tf.reduce_mean(tf.to_float(iou_scores >= 0.75) * detect_mask * class_mask) / (count + 1e-3)
        avg_iou = tf.reduce_mean(iou_scores) / (count + 1e-3)


        # coord loss label_wh normalzation 0-1
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / tf.cast(feature_shape[0], dtype=tf.float32) / tf.cast(feature_shape[1], dtype=tf.float32)
        ciou = tf.expand_dims(self.box_ciou(pred_xywh, label_xywh), axis=-1)
        ciou_loss = object_mask * bbox_loss_scale * (1 - ciou)

        # angle loss
        loss_reim = object_mask * bbox_loss_scale * 0.5 * tf.square(tf.square(label_reim - pred_reim))

        # confidence loss
        valid_boxes = tf.boolean_mask(label_xywh, tf.cast(object_mask, 'bool'))
        bboxes = tf.concat([valid_boxes[:, 0:2], valid_boxes[:, 2:4]], axis=-1)
        # shape: [V, 2] ——> [1, V, 2]
        bboxes = tf.expand_dims(bboxes, 0)
        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        best_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
        noobject_mask = (1.0 - object_mask) * tf.cast( best_iou < self.iou_threshold, tf.float32)

        # Focal loss分配了权重，注释掉原本的object_scale和noobject_scale
        conf_focal = self.focal(object_mask, pred_conf)
        object_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=conv_conf)
        noobject_loss = noobject_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=conv_conf)
        conf_loss = conf_focal * (object_loss + noobject_loss)

        # prob loss
        prob_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_prob)

        ciou_loss = tf.reduce_mean(tf.reduce_sum(ciou_loss, axis=[1, 2, 3, 4]))
        reim_loss = tf.reduce_mean(tf.reduce_sum(loss_reim, axis=[1, 2, 3, 4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

        return ciou_loss, reim_loss, conf_loss, prob_loss, recall50, recall75, avg_iou

    def bbox_iou(self, boxes_1, boxes_2):
        """
        calculate regression loss using iou
        :param boxes_1: boxes_1 shape is [x, y, w, h]
        :param boxes_2: boxes_2 shape is [x, y, w, h]
        :return:
        """
        # transform [x, y, w, h] to [x_min, y_min, x_max, y_max]
        boxes_1 = tf.concat([boxes_1[..., :2] - boxes_1[..., 2:] * 0.5,
                             boxes_1[..., :2] + boxes_1[..., 2:] * 0.5], axis=-1)
        boxes_2 = tf.concat([boxes_2[..., :2] - boxes_2[..., 2:] * 0.5,
                             boxes_2[..., :2] + boxes_2[..., 2:] * 0.5], axis=-1)
        boxes_1 = tf.concat([tf.minimum(boxes_1[..., :2], boxes_1[..., 2:]),
                             tf.maximum(boxes_1[..., :2], boxes_1[..., 2:])], axis=-1)
        boxes_2 = tf.concat([tf.minimum(boxes_2[..., :2], boxes_2[..., 2:]),
                             tf.maximum(boxes_2[..., :2], boxes_2[..., 2:])], axis=-1)

        # calculate area of boxes_1 boxes_2
        boxes_1_area = (boxes_1[..., 2] - boxes_1[..., 0]) * (boxes_1[..., 3] - boxes_1[..., 1])
        boxes_2_area = (boxes_2[..., 2] - boxes_2[..., 0]) * (boxes_2[..., 3] - boxes_2[..., 1])

        # calculate the two corners of the intersection
        left_up = tf.maximum(boxes_1[..., :2], boxes_2[..., :2])
        right_down = tf.minimum(boxes_1[..., 2:], boxes_2[..., 2:])

        # calculate area of intersection
        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]

        # calculate union area
        union_area = boxes_1_area + boxes_2_area - inter_area

        # calculate iou add epsilon in denominator to avoid dividing by 0
        iou = inter_area / (union_area + tf.keras.backend.epsilon())

        return iou

    def bbox_giou(self, boxes_1, boxes_2):
        """
        calculate regression loss using giou
        :param boxes_1: boxes_1 shape is [x, y, w, h]
        :param boxes_2: boxes_2 shape is [x, y, w, h]
        :return:
        """
        # transform [x, y, w, h] to [x_min, y_min, x_max, y_max]
        boxes_1 = tf.concat([boxes_1[..., :2] - boxes_1[..., 2:] * 0.5,
                             boxes_1[..., :2] + boxes_1[..., 2:] * 0.5], axis=-1)
        boxes_2 = tf.concat([boxes_2[..., :2] - boxes_2[..., 2:] * 0.5,
                             boxes_2[..., :2] + boxes_2[..., 2:] * 0.5], axis=-1)
        boxes_1 = tf.concat([tf.minimum(boxes_1[..., :2], boxes_1[..., 2:]),
                             tf.maximum(boxes_1[..., :2], boxes_1[..., 2:])], axis=-1)
        boxes_2 = tf.concat([tf.minimum(boxes_2[..., :2], boxes_2[..., 2:]),
                             tf.maximum(boxes_2[..., :2], boxes_2[..., 2:])], axis=-1)

        # calculate area of boxes_1 boxes_2
        boxes_1_area = (boxes_1[..., 2] - boxes_1[..., 0]) * (boxes_1[..., 3] - boxes_1[..., 1])
        boxes_2_area = (boxes_2[..., 2] - boxes_2[..., 0]) * (boxes_2[..., 3] - boxes_2[..., 1])

        # calculate the two corners of the intersection
        left_up = tf.maximum(boxes_1[..., :2], boxes_2[..., :2])
        right_down = tf.minimum(boxes_1[..., 2:], boxes_2[..., 2:])

        # calculate area of intersection
        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]

        # calculate union area
        union_area = boxes_1_area + boxes_2_area - inter_area

        # calculate iou add epsilon in denominator to avoid dividing by 0
        iou = inter_area / (union_area + tf.keras.backend.epsilon())

        # calculate the upper left and lower right corners of the minimum closed convex surface
        enclose_left_up = tf.minimum(boxes_1[..., :2], boxes_2[..., :2])
        enclose_right_down = tf.maximum(boxes_1[..., 2:], boxes_2[..., 2:])

        # calculate width and height of the minimun closed convex surface
        enclose_wh = tf.maximum(enclose_right_down - enclose_left_up, 0.0)

        # calculate area of the minimun closed convex surface
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]

        # calculate the giou add epsilon in denominator to avoid dividing by 0
        giou = iou - 1.0 * (enclose_area - union_area) / (enclose_area + tf.keras.backend.epsilon())

        return giou

    def bbox_diou(self, boxes_1, boxes_2):
        """
        calculate regression loss using diou
        :param boxes_1: boxes_1 shape is [x, y, w, h]
        :param boxes_2: boxes_2 shape is [x, y, w, h]
        :return:
        """
        # calculate center distance
        center_distance = tf.reduce_sum(tf.square(boxes_1[..., :2] - boxes_2[..., :2]), axis=-1)

        # transform [x, y, w, h] to [x_min, y_min, x_max, y_max]
        boxes_1 = tf.concat([boxes_1[..., :2] - boxes_1[..., 2:] * 0.5,
                             boxes_1[..., :2] + boxes_1[..., 2:] * 0.5], axis=-1)
        boxes_2 = tf.concat([boxes_2[..., :2] - boxes_2[..., 2:] * 0.5,
                             boxes_2[..., :2] + boxes_2[..., 2:] * 0.5], axis=-1)
        boxes_1 = tf.concat([tf.minimum(boxes_1[..., :2], boxes_1[..., 2:]),
                             tf.maximum(boxes_1[..., :2], boxes_1[..., 2:])], axis=-1)
        boxes_2 = tf.concat([tf.minimum(boxes_2[..., :2], boxes_2[..., 2:]),
                             tf.maximum(boxes_2[..., :2], boxes_2[..., 2:])], axis=-1)

        # calculate area of boxes_1 boxes_2
        boxes_1_area = (boxes_1[..., 2] - boxes_1[..., 0]) * (boxes_1[..., 3] - boxes_1[..., 1])
        boxes_2_area = (boxes_2[..., 2] - boxes_2[..., 0]) * (boxes_2[..., 3] - boxes_2[..., 1])

        # calculate the two corners of the intersection
        left_up = tf.maximum(boxes_1[..., :2], boxes_2[..., :2])
        right_down = tf.minimum(boxes_1[..., 2:], boxes_2[..., 2:])

        # calculate area of intersection
        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]

        # calculate union area
        union_area = boxes_1_area + boxes_2_area - inter_area

        # calculate IoU, add epsilon in denominator to avoid dividing by 0
        iou = inter_area / (union_area + tf.keras.backend.epsilon())

        # calculate the upper left and lower right corners of the minimum closed convex surface
        enclose_left_up = tf.minimum(boxes_1[..., :2], boxes_2[..., :2])
        enclose_right_down = tf.maximum(boxes_1[..., 2:], boxes_2[..., 2:])

        # calculate width and height of the minimun closed convex surface
        enclose_wh = tf.maximum(enclose_right_down - enclose_left_up, 0.0)

        # calculate enclosed diagonal distance
        enclose_diagonal = tf.reduce_sum(tf.square(enclose_wh), axis=-1)

        # calculate diou add epsilon in denominator to avoid dividing by 0
        diou = iou - 1.0 * center_distance / (enclose_diagonal + tf.keras.backend.epsilon())

        return diou

    def box_ciou(self, boxes_1, boxes_2):
        """
        calculate regression loss using ciou
        :param boxes_1: boxes_1 shape is [x, y, w, h]
        :param boxes_2: boxes_2 shape is [x, y, w, h]
        :return:
        """
        # calculate center distance
        center_distance = tf.reduce_sum(tf.square(boxes_1[..., :2] - boxes_2[..., :2]), axis=-1)

        v = 4 * tf.square(tf.math.atan2(boxes_1[..., 2], boxes_1[..., 3]) - tf.math.atan2(boxes_2[..., 2], boxes_2[..., 3])) / (math.pi * math.pi)

        # transform [x, y, w, h] to [x_min, y_min, x_max, y_max]
        boxes_1 = tf.concat([boxes_1[..., :2] - boxes_1[..., 2:] * 0.5,
                             boxes_1[..., :2] + boxes_1[..., 2:] * 0.5], axis=-1)
        boxes_2 = tf.concat([boxes_2[..., :2] - boxes_2[..., 2:] * 0.5,
                             boxes_2[..., :2] + boxes_2[..., 2:] * 0.5], axis=-1)
        boxes_1 = tf.concat([tf.minimum(boxes_1[..., :2], boxes_1[..., 2:]),
                             tf.maximum(boxes_1[..., :2], boxes_1[..., 2:])], axis=-1)
        boxes_2 = tf.concat([tf.minimum(boxes_2[..., :2], boxes_2[..., 2:]),
                             tf.maximum(boxes_2[..., :2], boxes_2[..., 2:])], axis=-1)

        # calculate area of boxes_1 boxes_2
        boxes_1_area = (boxes_1[..., 2] - boxes_1[..., 0]) * (boxes_1[..., 3] - boxes_1[..., 1])
        boxes_2_area = (boxes_2[..., 2] - boxes_2[..., 0]) * (boxes_2[..., 3] - boxes_2[..., 1])

        # calculate the two corners of the intersection
        left_up = tf.maximum(boxes_1[..., :2], boxes_2[..., :2])
        right_down = tf.minimum(boxes_1[..., 2:], boxes_2[..., 2:])

        # calculate area of intersection
        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]

        # calculate union area
        union_area = boxes_1_area + boxes_2_area - inter_area

        # calculate IoU, add epsilon in denominator to avoid dividing by 0
        iou = inter_area / (union_area + tf.keras.backend.epsilon())

        # calculate the upper left and lower right corners of the minimum closed convex surface
        enclose_left_up = tf.minimum(boxes_1[..., :2], boxes_2[..., :2])
        enclose_right_down = tf.maximum(boxes_1[..., 2:], boxes_2[..., 2:])

        # calculate width and height of the minimun closed convex surface
        enclose_wh = tf.maximum(enclose_right_down - enclose_left_up, 0.0)

        # calculate enclosed diagonal distance
        enclose_diagonal = tf.reduce_sum(tf.square(enclose_wh), axis=-1)

        # calculate diou
        diou = iou - 1.0 * center_distance / (enclose_diagonal + tf.keras.backend.epsilon())

        # calculate param v and alpha to CIoU
        alpha = v / (1.0 - iou + v)

        # calculate ciou
        ciou = diou - alpha * v

        return ciou

    def focal(self, target, actual, alpha=0.25, gamma=2):
        focal_loss = tf.abs(alpha + target - 1) * tf.pow(tf.abs(target - actual), gamma)
        return focal_loss

    def smooth_labels(self, y_true, label_smoothing=0.01):
        # smooth labels
        label_smoothing = tf.constant(label_smoothing, dtype=tf.float32)
        uniform_distribution = np.full(self.class_num, 1.0 / self.class_num)
        smooth_onehot = y_true * (1 - label_smoothing) + label_smoothing * uniform_distribution
        return smooth_onehot
