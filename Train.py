# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import os
import cv2
import sys
import glob
import time
import random

import numpy as np
import tensorflow as tf

from Define import *
from Utils import *
from Teacher import *

from YOLOv3 import *
from YOLOv3_Loss import *
from YOLOv3_Utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INFO

# 1. dataset
train_data_list = np.load('./dataset/train_detection.npy', allow_pickle = True)
valid_data_list = np.load('./dataset/validation_detection.npy', allow_pickle = True)
valid_count = len(valid_data_list)

open('log.txt', 'w')
log_print('[i] Train : {}'.format(len(train_data_list)))
log_print('[i] Valid : {}'.format(len(valid_data_list)))

# 2. build
input_var = tf.placeholder(tf.float32)
input_size_var = tf.placeholder(tf.int32)
is_training = tf.placeholder(tf.bool)

gt_tensors_var = tf.placeholder(tf.float32, [None, None, 5 + CLASSES])

utils = YOLOv3_Utils()
pred_tensors_op = YOLOv3(input_var, input_size_var, is_training, utils.anchors)

log_print('[i] pred_bboxes_op : {}'.format(pred_tensors_op))
log_print('[i] gt_bboxes_var : {}'.format(gt_tensors_var))

loss_op, giou_loss_op, conf_loss_op, class_loss_op = YOLOv3_Loss(pred_tensors_op, gt_tensors_var, tf.cast(input_size_var, tf.float32))

vars = tf.trainable_variables()
l2_reg_loss_op = tf.add_n([tf.nn.l2_loss(var) for var in vars]) * WEIGHT_DECAY
loss_op += l2_reg_loss_op

learning_rate_var = tf.placeholder(tf.float32)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_op = tf.train.AdamOptimizer(learning_rate_var).minimize(loss_op)
    # train_op = tf.train.MomentumOptimizer(learning_rate_var, momentum = 0.9).minimize(loss_op, colocate_gradients_with_ops = True)

train_summary_dic = {
    'Loss/Total_Loss' : loss_op,
    'Loss/GIoU_Loss' : giou_loss_op,
    'Loss/Confidence_Loss' : conf_loss_op,
    'Loss/Classification_Loss' : class_loss_op,
    'Loss/L2_Regularization_Loss' : l2_reg_loss_op,
    'Learning_rate' : learning_rate_var,
}

train_summary_list = []
for name in train_summary_dic.keys():
    value = train_summary_dic[name]
    train_summary_list.append(tf.summary.scalar(name, value))
train_summary_op = tf.summary.merge(train_summary_list)

log_image_var = tf.placeholder(tf.float32, [None, TEST_INPUT_SIZE, TEST_INPUT_SIZE, 3])
log_image_op = tf.summary.image('Image/Train', log_image_var[..., ::-1], BATCH_SIZE)

# 3. train
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# '''
pretrained_vars = []
for var in vars:
    if 'resnet_v1_50' in var.name:
        pretrained_vars.append(var)

pretrained_saver = tf.train.Saver(var_list = pretrained_vars)
pretrained_saver.restore(sess, './resnet_v1_model/resnet_v1_50.ckpt')
# '''

saver = tf.train.Saver(max_to_keep = 10)
# saver.restore(sess, './model/YOLOv3_{}.ckpt'.format(30000))

learning_rate = INIT_LEARNING_RATE

log_print('[i] max_iteration : {}'.format(MAX_ITERATION))
log_print('[i] decay_iteration : {}'.format(DECAY_ITERATIONS))

loss_list = []
giou_loss_list = []
conf_loss_list = []
class_loss_list = []
l2_reg_loss_list = []
train_time = time.time()

train_writer = tf.summary.FileWriter('./logs/train')

train_threads = []
for i in range(NUM_THREADS):
    train_thread = Teacher('./dataset/train_detection.npy', debug = False)
    train_thread.start()
    train_threads.append(train_thread)

sample_data_list = train_data_list[:BATCH_SIZE]

for iter in range(1, MAX_ITERATION + 1):
    if iter in DECAY_ITERATIONS:
        learning_rate /= 10
        log_print('[i] learning rate decay : {} -> {}'.format(learning_rate * 10, learning_rate))

    # Thread
    find = False
    while not find:
        for train_thread in train_threads:
            if train_thread.ready:
                find = True
                batch_image_data, batch_label_data, input_size = train_thread.get_batch_data()        
                break
                
    _feed_dict = {input_var : batch_image_data, gt_tensors_var : batch_label_data, input_size_var : input_size, is_training : True, learning_rate_var : learning_rate}
    log = sess.run([train_op, loss_op, giou_loss_op, conf_loss_op, class_loss_op, l2_reg_loss_op, train_summary_op], feed_dict = _feed_dict)
    # print(log[1:-1])
    
    if np.isnan(log[1]):
        print('[!]', log[1:-1])
        input()

    loss_list.append(log[1])
    giou_loss_list.append(log[2])
    conf_loss_list.append(log[3])
    class_loss_list.append(log[4])
    l2_reg_loss_list.append(log[5])
    train_writer.add_summary(log[6], iter)
    
    if iter % LOG_ITERATION == 0:
        loss = np.mean(loss_list)
        giou_loss = np.mean(giou_loss_list)
        conf_loss = np.mean(conf_loss_list)
        class_loss = np.mean(class_loss_list)
        l2_reg_loss = np.mean(l2_reg_loss_list)
        train_time = int(time.time() - train_time)
        
        log_print('[i] iter : {}, loss : {:.4f}, giou_loss : {:.4f}, conf_loss : {:.4f}, class_loss : {:.4f}, l2_reg_loss : {:.4f}, train_time : {}sec'.format(iter, loss, giou_loss, conf_loss, class_loss, l2_reg_loss, train_time))

        loss_list = []
        giou_loss_list = []
        conf_loss_list = []
        class_loss_list = []
        l2_reg_loss_list = []
        train_time = time.time()

    # if iter % SAMPLE_ITERATION == 0:
    #     total_gt_bboxes = []
    #     batch_image_data = np.zeros((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL), dtype = np.float32)

    #     for i, data in enumerate(sample_data_list):
    #         image_name, gt_bboxes, gt_classes = data        
    #         image_path = TRAIN_DIR + image_name

    #         gt_bboxes = np.asarray(gt_bboxes, dtype = np.float32)
    #         gt_classes = np.asarray([CLASS_DIC[c] for c in gt_classes], dtype = np.int32)

    #         image = cv2.imread(image_path)
    #         image_h, image_w, image_c = image.shape

    #         tf_image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)

    #         gt_bboxes /= [image_w, image_h, image_w, image_h]
    #         gt_bboxes *= [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT]

    #         batch_image_data[i] = tf_image.copy()
    #         total_gt_bboxes.append(gt_bboxes)

    #     total_pred_bboxes, total_pred_classes = sess.run([pred_bboxes_op, pred_classes_op], feed_dict = {input_var : batch_image_data, is_training : False})
        
    #     sample_images = []
    #     for i in range(BATCH_SIZE):
    #         image = batch_image_data[i]
    #         pred_bboxes, pred_classes = retina_utils.Decode(total_pred_bboxes[i], total_pred_classes[i], [IMAGE_WIDTH, IMAGE_HEIGHT], detect_threshold = 0.05)
            
    #         for bbox, class_index in zip(pred_bboxes, pred_classes):
    #             xmin, ymin, xmax, ymax = bbox[:4].astype(np.int32)
    #             conf = bbox[4]
    #             class_name = CLASS_NAMES[class_index]
                
    #             string = "{} : {:.2f}%".format(class_name, conf * 100)
    #             cv2.putText(image, string, (xmin, ymin - 10), 1, 1, (0, 255, 0))
    #             cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    #         for gt_bbox in total_gt_bboxes[i]:
    #             xmin, ymin, xmax, ymax = gt_bbox.astype(np.int32)
    #             cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

    #         sample_images.append(image.copy())
            
    #     image_summary = sess.run(log_image_op, feed_dict = {log_image_var : sample_images})
    #     train_writer.add_summary(image_summary, iter)

    if iter % SAVE_ITERATION == 0:
        saver.save(sess, './model/RetinaNet_{}.ckpt'.format(iter))

saver.save(sess, './model/RetinaNet.ckpt')
