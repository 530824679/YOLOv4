# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : freeze_graph.py
# Description : 冻结权重ckpt——>pb
# --------------------------------------

import os
import numpy as np
import tensorflow as tf
from model.network import Network

pb_file = './checkpoints/model.pb'
ckpt_file = './checkpoints/model.ckpt'
output_node_names = ["input/input_data", "pred_sbbox/concat_2", "pred_mbbox/concat_2", "pred_lbbox/concat_2"]

with tf.name_scope('input'):
    input_data = tf.placeholder(dtype=tf.float32, name='input_data')

    model = Network()
    print(model.conv_sbbox, model.conv_mbbox, model.conv_lbbox)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_file)

    converted_graph_def = tf.graph_util.convert_variables_to_constants(sess, input_graph_def=sess.graph.as_graph_def(), output_node_names=output_node_names)

    with tf.gfile.GFile(pb_file, "wb") as f:
        f.write(converted_graph_def.SerializeToString())

    print("=> {} ops written to {}.".format(len(converted_graph_def.node), pb_file))
