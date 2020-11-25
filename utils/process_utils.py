# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : process_utils.py
# Description :function
# --------------------------------------
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
import random
import colorsys
import numpy as np

def calc_iou_wh(box1_wh, box2_wh):
    """
    param box1_wh (list, tuple): Width and height of a box
    param box2_wh (list, tuple): Width and height of a box
    return (float): iou
    """
    min_w = min(box1_wh[0], box2_wh[0])
    min_h = min(box1_wh[1], box2_wh[1])
    area_r1 = box1_wh[0] * box1_wh[1]
    area_r2 = box2_wh[0] * box2_wh[1]
    intersect = min_w * min_h
    union = area_r1 + area_r2 - intersect
    return intersect / union

def calculate_iou(box_1, box_2):
    """
    calculate iou
    :param box_1: (x0, y0, x1, y1)
    :param box_2: (x0, y0, x1, y1)
    :return: value of iou
    """
    bboxes1 = np.transpose(box_1)
    bboxes2 = np.transpose(box_2)

    # 计算两个box的交集：交集左上角的点取两个box的max，交集右下角的点取两个box的min
    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])

    # 计算两个box交集的wh：如果两个box没有交集，那么wh为0(按照计算方式wh为负数，跟0比较取最大值)
    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)

    # 交集面积
    intersection = int_h * int_w
    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])  # bboxes1面积
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])  # bboxes2面积

    # iou=交集/并集
    iou = intersection / (vol1 + vol2 - intersection)

    return iou

def bboxes_cut(bbox_min_max, bboxes):
    bboxes = np.concatenate([np.maximum(bboxes[:, :2], [bbox_min_max[0], bbox_min_max[1]]), np.minimum(bboxes[:, 2:], [bbox_min_max[2] - 1, bbox_min_max[3] - 1])], axis=-1)
    invalid_mask = np.logical_or((bboxes[:, 0] > bboxes[:, 2]), (bboxes[:, 1] > bboxes[:, 3]))
    return invalid_mask

def bboxes_sort(coords, scores, classes, top_k=400):
    index = np.argsort(-scores)
    classes = classes[index][:top_k]
    scores = scores[index][:top_k]
    coords = coords[index][:top_k]
    return coords, scores, classes

def non_maximum_suppression(bboxes, scores, classes, iou_threshold=0.5):
    """
    calculate the non-maximum suppression to eliminate the overlapped box
    :param classes: shape is [num, 1] classes
    :param scores: shape is [num, 1] scores
    :param bboxes: shape is [num, 4] (xmin, ymin, xmax, ymax)
    :param nms_threshold: iou threshold
    :return:
    """
    results = np.concatenate([bboxes, scores, classes], axis=-1)
    classes_in_img = list(set(results[:, 5]))
    best_results = []

    for cls in classes_in_img:
        cls_mask = (results[:, 5] == cls)
        cls_bboxes = results[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_result = cls_bboxes[max_ind]
            best_results.append(best_result)
            cls_bboxes = np.concatenate([cls_bboxes[:max_ind], cls_bboxes[max_ind + 1:]])
            overlap = calculate_iou(best_result[np.newaxis, :4], cls_bboxes[:, :4])

            weight = np.ones((len(overlap),), dtype=np.float32)
            iou_mask = overlap > iou_threshold
            weight[iou_mask] = 0.0

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_results


def soft_non_maximum_suppression(classes, scores, bboxes, sigma=0.3):
    """
    calculate the soft non-maximum suppression to eliminate the overlapped box
    :param classes: shape is [num, 1] classes
    :param scores: shape is [num, 1] scores
    :param bboxes: shape is [num, 4] (xmin, ymin, xmax, ymax)
    :param sigma: soft weight
    :return:
    """
    results = np.concatenate([bboxes, scores, classes], axis=-1)
    classes_in_img = list(set(results[:, 5]))
    best_results = []

    for cls in classes_in_img:
        cls_mask = (results[:, 5] == cls)
        cls_bboxes = results[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_result = cls_bboxes[max_ind]
            best_results.append(best_result)
            cls_bboxes = np.concatenate([cls_bboxes[:max_ind], cls_bboxes[max_ind + 1:]])
            overlap = calculate_iou(best_result[np.newaxis, :4], cls_bboxes[:, :4])

            weight = np.ones((len(overlap),), dtype=np.float32)
            weight = np.exp(-(1.0 * overlap ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_results

def postpreocess(bboxes, input_size, origin_size, score_threshold=0.5):
    """
    The result of network prediction is processed and filtered bounding boxes
    :param bboxes: predict result shape is [num, 6] x, y, w, h, score, class
    :param input_size: network input size (h, w)
    :param origin_size: image origin size (h, w)
    :param score_threshold: filter threshold
    :return:
    """
    valid_scale = [0, np.inf]
    pred_xywh = bboxes[:, 0:4]
    pred_conf = bboxes[:, 4]
    pred_prob = bboxes[:, 5:]

    # transform (x, y, w, h) to (xmin, ymin, xmax, ymax)
    pred_coord = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5, pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)

    # transform (xmin, ymin, xmax, ymax) to (xmin_origin, ymin_origin, xmax_origin, ymax_origin)
    input_height, input_width = input_size
    origin_height, origin_width = origin_size
    resize_ratio = min(input_width / origin_width, input_height / origin_height)
    dw = (input_width - resize_ratio * origin_width) / 2
    dh = (input_height - resize_ratio * origin_height) / 2
    pred_coord[:, 0::2] = 1.0 * (pred_coord[:, 0::2] - dw) / resize_ratio
    pred_coord[:, 1::2] = 1.0 * (pred_coord[:, 1::2] - dh) / resize_ratio

    # cut out the part of the boundary box that goes beyond the entire image
    bbox_min_max = [0, 0, origin_width - 1, origin_height - 1]
    invalid_mask = bboxes_cut(bbox_min_max, pred_coord)
    pred_coord[invalid_mask] = 0

    # discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coord[:, 2:4] - pred_coord[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # confidence * category conditional probability = category confidence scores
    classes_max = np.argmax(pred_prob, axis=1)
    pred_prob = pred_prob[np.arange(len(pred_coord)), classes_max]
    conf_scores = pred_conf * pred_prob

    # class confidence scores > threshold
    keep_index = conf_scores > score_threshold
    mask = np.logical_and(scale_mask, keep_index)
    coords, scores, classes = pred_coord[mask], conf_scores[mask], classes_max[mask]

    # sort the first 400
    scores = scores[:, np.newaxis]
    classes = classes[:, np.newaxis]
    coords, scores, classes = bboxes_sort(coords, scores, classes)

    # calculate nms
    results = non_maximum_suppression(coords, scores, classes)

    return results

def preporcess(image, target_size, gt_boxes=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    ih, iw = target_size
    h,  w, _ = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes

def visualization(image, bboxes, labels, thr=0.3):
    """
    # Generate colors for drawing bounding boxes.
    :param image: raw image
    :param bboxes: shape is [num, 6]
    :param labels: shape is [num, ]
    :param thr:
    :return:
    """
    if bboxes is None:
        return image

    hsv_tuples = [(x / float(len(labels)), 1., 1.) for x in range(len(labels))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.

    # draw image
    cvImage = np.copy(image)
    h, w, _ = cvImage.shape
    for i, box in enumerate(bboxes[:, 0:4]):
        if bboxes[i, 4] < thr:
            continue
        cls_indx = bboxes[i][5]

        thick = int((h + w) / 300)
        cv2.rectangle(cvImage, (box[0], box[1]), (box[2], box[3]), colors[cls_indx], thick)
        bbox_text = '%s: %.3f' % (labels[cls_indx], bboxes[i, 4])
        if box[1] < 20:
            text_loc = (box[0] + 2, box[1] + 15)
        else:
            text_loc = (box[0], box[1] - 10)
        cv2.putText(cvImage, bbox_text, text_loc, cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * h, (255, 255, 255), thick // 3)
    cv2.imshow("test", cvImage)
    cv2.waitKey(0)