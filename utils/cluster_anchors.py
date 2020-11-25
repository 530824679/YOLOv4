# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : generator_anchors.py
# Description :generator k anchors
# --------------------------------------
"""
Boxes:
[[  5   5]
 [  7  15]
 [ 16  41]
 [ 18  43]
 [ 19  47]
 [ 28 104]]
Accuracy: 85.75%

Boxes:
[[  6   7]
 [ 18  43]
 [ 20  46]
 [ 22  62]
 [ 28  85]
 [ 28 109]]
Accuracy: 83.69%

Boxes:
[[  6   7]
 [ 17  42]
 [ 19  44]
 [ 19  49]
 [ 23  67]
 [ 28 105]]
Accuracy: 84.48%

Boxes:
[[  5   5]
 [  7  15]
 [ 17  41]
 [ 18  44]
 [ 19  49]
 [ 28 104]]
Accuracy: 85.78%
"""
import os
import math
import xml.etree.ElementTree as ET
import numpy as np
import glob
import matplotlib.pyplot as plt

def calc_iou(box, clusters):
    """
    calculate iou of a ground truth and all anchor
    :param box: width and height of a ground truth
    :param clusters: numpy shape is (k, 2), k is number of the anchor
    :return: iou of the ground truth and each anchor
    """
    w = np.minimum(clusters[:, 0], box[0])
    h = np.minimum(clusters[:, 1], box[1])

    if np.count_nonzero(w == 0) > 0 or np.count_nonzero(h == 0) > 0:
        raise ValueError("Box has no area")

    intersection = w * h
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou = intersection / (box_area + cluster_area - intersection) * 1.0
    return iou

def avg_iou(boxes, clusters):
    """
    calculate iou mean between ground truth and k anchors
    :param boxes: ground truth
    :param clusters: k clustering center
    :return: average iou
    """
    return np.mean([np.max(calc_iou(boxes[i], clusters)) for i in range(boxes.shape[0])])

def kmeans(boxes, k):
    """
    generate k anchors throught clustering
    :param boxes: shape is (n, 2) ground truth, n is the number of ground truth
    :param k: the number of anchors
    :return: shape is (k, 2) anchors, k is the number of anchors
    """
    num = boxes.shape[0]
    distances = np.empty((num, k))
    last_clusters = np.zeros((num,))

    np.random.seed()
    # randomly initialize k clustering center from num ground truth
    clusters = boxes[np.random.choice(num, k, replace=False)]

    while True:
        # calculate the distance of each ground truth and k anchors
        for i in range(num):
            distances[i] = 1 - calc_iou(boxes[i], clusters)
        # select nearest anchor index for each ground truth
        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        # update clustering center using medium of every group cluster
        for cluster in range(k):
            clusters[cluster] = np.median((boxes[nearest_clusters == cluster]), axis=0)
        last_clusters = nearest_clusters

    return clusters, nearest_clusters

def load_dataset(path, target_size=None):
    """
    Gets the width and height of the annotation from dataset
    :param path: path of the dataset
    :return: list width and height
    """
    list_wh = []
    file_list = os.listdir(path)
    for txt in file_list:
        print(txt)
        label_path = os.path.join(path, txt)
        lines = [line.rstrip() for line in open(label_path)]
        for line in lines:
            data = line.split(' ')
            data[4:] = [float(t) for t in data[4:]]
            width = math.sqrt(math.pow((data[17] - data[26]), 2) + math.pow((data[16] - data[25]), 2))
            length = math.sqrt(math.pow((data[17] - data[20]), 2) + math.pow((data[16] - data[19]), 2))

            width = int(width / 0.1)
            length = int(length / 0.1)
            if width <= 0 or length <= 0 or width > 1000 or length > 1000:
                continue

            list_wh.append([width, length])

    return np.array(list_wh)

def plot_data(data, out, k, index):
    for i in range(k):
        color = ['orange', 'green', 'blue', 'gray', 'yellow', 'purple', 'pink', 'black', 'brown']
        mark = ['o', 's', '^', '*', 'd', '+', 'x', 'p', '|']
        lab = 'cluster' + str(i + 1)
        plt.scatter(data[index == i, 0], data[index == i, 1], s=10, c=color[i], marker=mark[i], label=lab)
        # draw the centers
    out = np.array(out)
    plt.scatter(out[:, 0], out[:, 1], s=250, marker="*", c="red", label="cluster center")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    path = '/home/chenwei/HDD/livox_dl/LIVOX/object/training/label'
    cluster_num = 6

    data = load_dataset(path)
    anchors, cluster_index = kmeans(data, cluster_num)
    anchors = anchors.astype('int').tolist()
    anchors = sorted(anchors, key=lambda x: x[0] * x[1])

    print('Boxes:')
    print(np.array(anchors))
    print("Accuracy: {:.2f}%".format(avg_iou(data, np.array(anchors)) * 100))

    plot_data(data, anchors, cluster_num, cluster_index)