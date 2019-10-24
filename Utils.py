
import numpy as np

from Define import *

def log_print(string, log_path = './log.txt'):
    print(string)
    
    f = open(log_path, 'a+')
    f.write(string + '\n')
    f.close()

def one_hot(label, classes):
    v = np.zeros((classes), dtype = np.float32)
    v[label] = 1.
    return v

def smooth_one_hot(label, classes, delta = 0.01):
    v = one_hot(label, classes)
    uniform_distribution = np.full(classes, 1. / classes)
    
    v = (1 - delta) * v + delta * uniform_distribution
    return v

def ccwh_to_xyxy(bbox):
    cx, cy, w, h = bbox

    xmin = max(cx - w / 2, 0)
    ymin = max(cy - h / 2, 0)
    xmax = cx + w / 2
    ymax = cy + h / 2 

    return np.asarray([xmin, ymin, xmax, ymax])

def xyxy_to_ccwh(bbox):
    xmin, ymin, xmax, ymax = bbox

    cx = (xmax + xmin) / 2
    cy = (ymax + ymin) / 2
    width = xmax - xmin
    height = ymax - ymin

    return np.asarray([cx, cy, width, height])

def convert_bboxes(bboxes, image_wh, ori_wh):
    return bboxes / (ori_wh * 2) * (image_wh * 2)

def compute_bboxes_IoU(bboxes_1, bboxes_2):
    area_1 = (bboxes_1[:, 2] - bboxes_1[:, 0] + 1) * (bboxes_1[:, 3] - bboxes_1[:, 1] + 1)
    area_2 = (bboxes_2[:, 2] - bboxes_2[:, 0] + 1) * (bboxes_2[:, 3] - bboxes_2[:, 1] + 1)
    
    iw = np.minimum(bboxes_1[:, 2][:, np.newaxis], bboxes_2[:, 2]) - np.maximum(bboxes_1[:, 0][:, np.newaxis], bboxes_2[:, 0]) + 1
    ih = np.minimum(bboxes_1[:, 3][:, np.newaxis], bboxes_2[:, 3]) - np.maximum(bboxes_1[:, 1][:, np.newaxis], bboxes_2[:, 1]) + 1

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)
    
    intersection = iw * ih
    union = (area_1[:, np.newaxis] + area_2) - iw * ih

    return intersection / np.maximum(union, 1e-10)

def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return dets[keep]

def class_nms(pred_bboxes, pred_classes, threshold = NMS_THRESHOLD):
    data_dic = {}
    nms_bboxes = []
    nms_classes = []

    for bbox, class_index in zip(pred_bboxes, pred_classes):
        try:
            data_dic[class_index].append(bbox)
        except KeyError:
            data_dic[class_index] = []
            data_dic[class_index].append(bbox)
    
    for key in data_dic.keys():
        pred_bboxes = np.asarray(data_dic[key], dtype = np.float32)
        pred_bboxes = nms(pred_bboxes, threshold)

        for pred_bbox in pred_bboxes:
            nms_bboxes.append(pred_bbox)
            nms_classes.append(key)
    
    return np.asarray(nms_bboxes, dtype = np.float32), np.asarray(nms_classes, dtype = np.int32)
