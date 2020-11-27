# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : configs.py
# Description :config parameters
# --------------------------------------
import os

path_params = {
    'data_path': '/home/chenwei/HDD/livox_dl/LIVOX',
    'checkpoints_dir': './checkpoints',
    'logs_dir': './logs',
    'tfrecord_dir': '/home/chenwei/HDD/livox_dl/LIVOX/tfrecord',
    'checkpoints_name': 'model.ckpt',
    'train_tfrecord_name': 'train.tfrecord',
    'val_tfrecord_name': 'val.tfrecord',
    'test_output_dir': './test'
}

data_params = {
    'x_min': 0.0,
    'x_max': 60.8,
    'y_min': -30.4,
    'y_max': 30.4,
    'z_min': -3.0,
    'z_max': 3.0,
    'voxel_size': 0.1,
}

model_params = {
    'input_height': 608,                                # 图片高度
    'input_width': 608,                                 # 图片宽度
    'channels': 2,                                      # 输入图片通道数
    'anchors': [[5, 5], [7, 15],
                [17, 41], [18, 44],
                [19, 49], [28, 104]],
    'classes': ['car', 'bus', 'truck', 'pedestrains'],  # 类别
    'anchor_per_scale': 2,                              # 每个尺度的anchor个数
    'strides': [8, 16, 32],                             # 不同尺度的步长
    'upsample_method': "resize",                        # 上采样的方式
    'iou_threshold': 0.5,
    'max_bbox_per_scale': 50,
    'label_smoothing': 0.01
}

solver_params = {
    'gpu': '0',                     # 使用的gpu索引
    'learning_rate': 0.001,        # 初始学习率
    'decay_steps': 30000,           #衰变步数
    'decay_rate': 0.1,              #衰变率
    'warm_up_lr': 5e-5,
    'warm_up_epoch': 1,
    'batch_size': 8,                # 每批次输入的数据个数
    'total_epoches': 100000,        # 训练的最大迭代次数
    'val_step': 10,                 # 评估间隔
    'save_step': 1000,              # 权重保存间隔
    'log_step': 1000,               # 日志保存间隔
    'display_step': 100,            # 显示打印间隔
    'weight_decay': 0.0001,         # 正则化系数
}

test_params = {
    'prob_threshold': 0.3,         # 类别置信度分数阈值
    'iou_threshold': 0.45,           # nms阈值，小于0.45被过滤掉
    'max_output_size': 10           # nms选择的边界框最大数量
}

classes_map = {'bus': 0, 'car': 1, 'truck': 2, 'pedestrains': 3}
