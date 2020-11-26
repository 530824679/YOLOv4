import numpy as np
import tensorflow as tf
from collections import Counter
from utils.process_utils import calculate_iou, non_maximum_suppression

def evaluate(y_pred, y_true, num_classes, score_thresh=0.5, iou_thresh=0.5):

    num_images = y_true[0].shape[0]
    true_labels_dict   = {i:0 for i in range(num_classes)} # {class: count}
    pred_labels_dict   = {i:0 for i in range(num_classes)}
    true_positive_dict = {i:0 for i in range(num_classes)}

    for i in range(num_images):
        true_labels_list, true_boxes_list = [], []
        for j in range(3): # three feature maps
            true_probs_temp = y_true[j][i][...,5: ]
            true_boxes_temp = y_true[j][i][...,0:4]

            object_mask = true_probs_temp.sum(axis=-1) > 0

            true_probs_temp = true_probs_temp[object_mask]
            true_boxes_temp = true_boxes_temp[object_mask]

            true_labels_list += np.argmax(true_probs_temp, axis=-1).tolist()
            true_boxes_list  += true_boxes_temp.tolist()

        if len(true_labels_list) != 0:
            for cls, count in Counter(true_labels_list).items(): true_labels_dict[cls] += count

        pred_boxes = y_pred[0][i:i+1]
        pred_confs = y_pred[1][i:i+1]
        pred_probs = y_pred[2][i:i+1]

        pred_boxes, pred_confs, pred_labels = non_maximum_suppression(pred_boxes, pred_confs, pred_probs)

        true_boxes = np.array(true_boxes_list)
        box_centers, box_sizes = true_boxes[:,0:2], true_boxes[:,2:4]

        true_boxes[:,0:2] = box_centers - box_sizes / 2.
        true_boxes[:,2:4] = true_boxes[:,0:2] + box_sizes

        pred_labels_list = [] if pred_labels is None else pred_labels.tolist()
        if pred_labels_list == []: continue

        detected = []
        for k in range(len(true_labels_list)):
            # compute iou between predicted box and ground_truth boxes
            iou = calculate_iou(true_boxes[k:k+1], pred_boxes)
            m = np.argmax(iou) # Extract index of largest overlap
            if iou[m] >= iou_thresh and true_labels_list[k] == pred_labels_list[m] and m not in detected:
                pred_labels_dict[true_labels_list[k]] += 1
                detected.append(m)
        pred_labels_list = [pred_labels_list[m] for m in detected]

        for c in range(num_classes):
            t = true_labels_list.count(c)
            p = pred_labels_list.count(c)
            true_positive_dict[c] += p if t >= p else t

    recall    = sum(true_positive_dict.values()) / (sum(true_labels_dict.values()) + 1e-6)
    precision = sum(true_positive_dict.values()) / (sum(pred_labels_dict.values()) + 1e-6)
    avg_prec  = [true_positive_dict[i] / (true_labels_dict[i] + 1e-6) for i in range(num_classes)]
    mAP       = sum(avg_prec) / (sum([avg_prec[i] != 0 for i in range(num_classes)]) + 1e-6)

    return recall, precision, mAP

