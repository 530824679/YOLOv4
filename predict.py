# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : predict.py
# Description : predict demo for single image
# --------------------------------------

import cv2
import numpy as np
import tensorflow as tf
from utils.process_utils import *
from PIL import Image

return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = "./checkpoints/model.pb"
image_path      = "./test/1.jpeg"
num_classes     = 2
input_size      = 416
graph           = tf.Graph()

original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]
image_data = preporcess(np.copy(original_image), [input_size, input_size])
image_data = image_data[np.newaxis, ...]

return_tensors = read_pb_return_tensors(graph, pb_file, return_elements)

with tf.Session(graph=graph) as sess:
    pred_sbbox, pred_mbbox, pred_lbbox = sess.run([return_tensors[1], return_tensors[2], return_tensors[3]], feed_dict={ return_tensors[0]: image_data})

pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                            np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                            np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

bboxes = postprocess(pred_bbox, original_image_size, input_size, 0.3)
bboxes = bboxes_nms(bboxes, 0.45, method='nms')
image = visualization(original_image, bboxes)
image = Image.fromarray(image)
image.show()