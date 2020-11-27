# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : network.py
# Description :YOLO v3 network architecture
# --------------------------------------

import numpy as np
import tensorflow as tf
from cfg.config import model_params
from model.ops import *

class Network(object):
    def __init__(self, is_train):
        self.is_train = is_train
        self.strides = model_params['strides']
        self.class_num = len(model_params['classes'])
        self.anchors = model_params['anchors']
        self.iou_loss_thresh = model_params['iou_threshold']
        self.upsample_method = model_params['upsample_method']

    def forward(self, inputs):
        try:
            conv_lbbox, conv_mbbox, conv_sbbox = self.build_network(inputs)
        except:
            raise NotImplementedError("Can not build up yolov4 network!")

        with tf.variable_scope('pred_sbbox'):
            pred_sbbox = self.reorg_layer(conv_sbbox, self.anchors[0])

        with tf.variable_scope('pred_mbbox'):
            pred_mbbox = self.reorg_layer(conv_mbbox, self.anchors[1])

        with tf.variable_scope('pred_lbbox'):
            pred_lbbox = self.reorg_layer(conv_lbbox, self.anchors[2])

        logits = [conv_sbbox, conv_mbbox, conv_lbbox]
        preds = [pred_sbbox, pred_mbbox, pred_lbbox]
        return logits, preds

    def CSPDarknet53(self, inputs, scope='CSPDarknet53'):
        """
        定义网络特征提取层
        :param inputs:待输入的样本图片
        :param scope: 命名空间
        :return: 三个不同尺度的基础特征图输出
        """
        with tf.variable_scope(scope):
            input_data = conv2d(inputs, filters_shape=(3, 3, 2, 32), trainable=self.is_train, scope='conv0')
            input_data = csp_block(input_data, 64, 64, 64, 64, 64, 1, trainable=self.is_train, scope='csp_1')
            input_data = csp_block(input_data, 128, 64, 64, 64, 128, 2, trainable=self.is_train, scope='csp_2')
            input_data = csp_block(input_data, 256, 128, 128, 128, 256, 8, trainable=self.is_train, scope='csp_3')
            route_1 = input_data
            input_data = csp_block(input_data, 512, 256, 256, 256, 512, 8, trainable=self.is_train, scope='csp_4')
            route_2 = input_data
            input_data = csp_block(input_data, 1024, 512, 512, 512, 1024, 4, trainable=self.is_train, scope='csp_5')

            return route_1, route_2, input_data

    def build_network(self, inputs, scope='yolo_v4'):
        """
        定义前向传播过程
        :param inputs:待输入的样本图片
        :param scope: 命名空间
        :return: 三个不同尺度的检测层输出
        """
        route_1, route_2, route_3 = self.CSPDarknet53(inputs)
        route_3 = spp_block(route_3, trainable=self.is_train, scope='spp_block')
        route_2 = upsample_block(route_2, route_3, 256, 512, trainable=self.is_train, scope='upsample_block_1')
        route_1 = upsample_block(route_1, route_2, 128, 256, trainable=self.is_train, scope='upsample_block_2')
        conv_lobj_branch = conv2d(route_1, (3, 3, 128, 256), activate='leaky', trainable=self.is_train, scope='conv_lobj_branch')
        conv_lbbox = conv2d(conv_lobj_branch, (1, 1, 256, 2 * (self.class_num + 7)), activate=None, bn=False, trainable=self.is_train, scope='conv_lbbox')

        route_2 = downsample_block(route_1, route_2, 256, 512, trainable=self.is_train, scope='downsample_block_1')
        conv_mobj_branch = conv2d(route_2, (3, 3, 256, 512), activate='leaky', trainable=self.is_train, scope='conv_mobj_branch')
        conv_mbbox = conv2d(conv_mobj_branch, (1, 1, 512, 2 * (self.class_num + 7)), activate=None, bn=False, trainable=self.is_train, scope='conv_mbbox')

        route_3 = downsample_block(route_2, route_3, 512, 1024, trainable=self.is_train, scope='downsample_block_2')
        conv_sobj_branch = conv2d(route_3, (3, 3, 512, 1024), activate='leaky',  trainable=self.is_train, scope='conv_sobj_branch')
        conv_sbbox = conv2d(conv_sobj_branch, (1, 1, 1024, 2 * (self.class_num + 7)), activate=None, bn=False, trainable=self.is_train, scope='conv_sbbox')

        return conv_lbbox, conv_mbbox, conv_sbbox

    def reorg_layer(self, feature_maps, anchors):
        """
        解码网络输出的特征图
        :param feature_maps:网络输出的特征图
        :param anchors:当前层使用的anchor尺度
        :param stride:特征图相比原图的缩放比例
        :return: 预测层最终的输出 shape=[batch_size, feature_size, feature_size, anchor_per_scale, 7 + num_classes]
        """

        feature_shape = tf.shape(feature_maps)[1:3]
        batch_size = tf.shape(feature_maps)[0]
        anchor_per_scale = len(anchors)

        anchors = tf.constant(anchors, dtype=tf.float32)

        # 网络输出转化——偏移量、置信度、类别概率
        predict = tf.reshape(feature_maps, [batch_size, feature_shape[0], feature_shape[1], anchor_per_scale, self.class_num + 7])
        # 中心坐标相对于该cell左上角的偏移量，sigmoid函数归一化到0-1
        xy_offset = tf.nn.sigmoid(predict[:, :, :, :, 0:2])
        # 相对于anchor的wh比例，通过e指数解码, 避免出现NAN值
        wh_offset = tf.clip_by_value(tf.exp(predict[:, :, :, :, 2:4]), 1e-9, 50)
        # 复数角度re im
        pred_re = 2 * tf.sigmoid(predict[:, :, :, :, 4:5]) - 1
        pred_im = 2 * tf.sigmoid(predict[:, :, :, :, 5:6]) - 1
        # 置信度，sigmoid函数归一化到0-1
        pred_obj = tf.nn.sigmoid(predict[:, :, :, :, 6:7])
        # 网络回归的是得分,用softmax转变成类别概率
        pred_class = tf.nn.softmax(predict[:, :, :, :, 7:])

        # 构建特征图每个cell的左上角的xy坐标
        height_index = tf.range(feature_shape[0], dtype=tf.int32)
        width_index = tf.range(feature_shape[1], dtype=tf.int32)
        x_cell, y_cell = tf.meshgrid(height_index, width_index)
        x_cell = tf.reshape(x_cell, [-1, 1])
        y_cell = tf.reshape(y_cell, [-1, 1])
        xy_cell = tf.concat([x_cell, y_cell], axis=-1)
        xy_cell = tf.cast(tf.reshape(xy_cell, [feature_shape[0], feature_shape[1], 1, 2]), tf.float32)

        # decode to raw image norm 0-1
        bboxes_xy = (xy_cell + xy_offset) / tf.cast(feature_shape[::-1], tf.float32)
        bboxes_wh = (anchors * wh_offset) / tf.cast(feature_shape[::-1], tf.float32)
        pred_xywh = tf.concat([bboxes_xy, bboxes_wh], axis=-1)
        pred_remi = tf.concat([pred_re, pred_im], axis=-1)
        return tf.concat([pred_xywh, pred_remi, pred_obj, pred_class], axis=-1)