# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import tensorflow as tf

from Define import *

'''
pt = {
    p    , if y = 1
    1 − p, otherwise
}
FL(pt) = −(1 − pt)γ * log(pt)
'''
def Focal_Loss(pred_classes, gt_classes, alpha = 0.25, gamma = 2):
    with tf.variable_scope('Focal'):
        # focal_loss = [BATCH_SIZE, 22890, CLASSES]
        pt = gt_classes * pred_classes + (1 - gt_classes) * (1 - pred_classes) 
        focal_loss = -alpha * tf.pow(1. - pt, gamma) * tf.log(pt + 1e-10)

        # focal_loss = [BATCH_SIZE]
        focal_loss = tf.reduce_sum(tf.abs(focal_loss))

    return focal_loss

'''
GIoU = IoU - (C - (A U B))/C
Loss = 1 - GIoU
'''
def GIoU(bboxes_1, bboxes_2):
    with tf.variable_scope('GIoU'):
        # 1. calulate intersection over union
        area_1 = (bboxes_1[..., 2] - bboxes_1[..., 0]) * (bboxes_1[..., 3] - bboxes_1[..., 1])
        area_2 = (bboxes_2[..., 2] - bboxes_2[..., 0]) * (bboxes_2[..., 3] - bboxes_2[..., 1])
        
        intersection_wh = tf.minimum(bboxes_1[:, :, 2:], bboxes_2[:, :, 2:]) - tf.maximum(bboxes_1[:, :, :2], bboxes_2[:, :, :2])
        intersection_wh = tf.maximum(intersection_wh, 0)
        
        intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
        union = (area_1 + area_2) - intersection
        
        ious = intersection / tf.maximum(union, 1e-10)

        # 2. (C - (A U B))/C
        C_wh = tf.maximum(bboxes_1[..., 2:], bboxes_2[..., 2:]) - tf.minimum(bboxes_1[..., :2], bboxes_2[..., :2])
        C_wh = tf.maximum(C_wh, 0.0)
        C = C_wh[..., 0] * C_wh[..., 1]
        
        giou = ious - (C - union) / tf.maximum(C, 1e-10)
    return giou

def YOLOv3_Loss(pred_tensors, gt_tensors, input_size, alpha = 1.0):
    # calculate focal_loss & GIoU_loss
    positive_count = tf.reduce_sum(gt_tensors[..., 4])

    # confidence, classes (without IoU)
    conf_loss_op = Focal_Loss(pred_tensors[..., 4], gt_tensors[..., 4]) / positive_count
    class_loss_op = tf.nn.sigmoid_cross_entropy_with_logits(logits = pred_tensors[..., 5:], labels = gt_tensors[..., 5:])
    class_loss_op = tf.reduce_sum(gt_tensors[..., 4] * class_loss_op) / positive_count

    # small object scale up, large object scale down
    bbox_scale = 2. - gt_tensors[..., 2] * gt_tensors[..., 3] / tf.pow(input_size, 2)

    giou_loss_op = 1. - GIoU(pred_tensors[..., :4], gt_tensors[..., :4])
    giou_loss_op = tf.reduce_sum(gt_tensors[..., 4] * bbox_scale * giou_loss_op) / positive_count
    
    loss_op = giou_loss_op + conf_loss_op + class_loss_op
    return loss_op, giou_loss_op, conf_loss_op, class_loss_op

if __name__ == '__main__':
    ## check loss shape
    pred_tensors = tf.placeholder(tf.float32, [None, None, 5 + CLASSES])
    gt_tensors = tf.placeholder(tf.float32, [None, None, 5 + CLASSES])
    input_size = tf.placeholder(tf.float32)

    loss_op, giou_loss_op, conf_loss_op, class_loss_op = YOLOv3_Loss(pred_tensors, gt_tensors, input_size)
    print(loss_op, giou_loss_op + conf_loss_op, class_loss_op)
