
import cv2
import random
import numpy as np

from Define import *
from Utils import *

from DataAugmentation import *

class YOLOv3_Utils:
    def __init__(self):
        self.input_sizes = np.linspace(MIN_INPUT_SIZE, MAX_INPUT_SIZE, INPUT_SIZE_COUNT)
        self.input_sizes = self.input_sizes.astype(np.int32)

        self.strides = np.asarray(STRIDES)
        self.anchors = np.asarray(ANCHORS, dtype = np.float32).reshape((3, 3, 2))

    def Encode(self, image_path, gt_bboxes, gt_classes, input_size, augment = False):
        # input_size = random.choice(self.input_sizes)
        output_sizes = input_size // self.strides
        
        image = cv2.imread(image_path)
        if augment:
            image, gt_bboxes, gt_classes = DataAugmentation(image, gt_bboxes, gt_classes)

        image_h, image_w, c = image.shape

        image = cv2.resize(image, (input_size, input_size), interpolation = cv2.INTER_CUBIC)
        labels = [np.zeros((output_size, output_size, 3, 5 + CLASSES)) for output_size in output_sizes] 

        # normalize
        gt_bboxes /= [image_w, image_h, image_w, image_h] 
        gt_bboxes *= input_size

        for gt_bbox, gt_class in zip(gt_bboxes, gt_classes):
            ccwh_bbox = xyxy_to_ccwh(gt_bbox)
            onehot = smooth_one_hot(gt_class, CLASSES)

            positive_count = 0

            best_max_iou = 0.0
            best_label_index = 0
            best_anchor_index = 0

            for i in range(3):
                anchors = np.zeros((3, 4), dtype = np.float32)
                anchors[:, :2] = ccwh_bbox[:2]
                anchors[:, 2:] = self.anchors[i]
                
                # anchors (cx, cy, w, h -> left, top, right, bottom)
                anchors[:, :2], anchors[:, 2:] = anchors[:, :2] - anchors[:, 2:] / 2, \
                                                 anchors[:, :2] + anchors[:, 2:] / 2

                ious = compute_bboxes_IoU(anchors, gt_bbox[np.newaxis, :])[:, 0]
                mask = ious >= IOU_THRESHOLD

                if positive_count == 0:
                    max_iou = np.max(ious)
                    max_index = np.argmax(ious)

                    if max_iou > best_max_iou:
                        best_max_iou = max_iou
                        best_label_index = i
                        best_anchor_index = max_index

                x_index, y_index = (ccwh_bbox[:2] / STRIDES[i]).astype(np.int32)
                
                labels[i][y_index, x_index, mask, :4] = gt_bbox
                labels[i][y_index, x_index, mask, 4] = 1
                labels[i][y_index, x_index, mask, 5:] = onehot

                positive_count += np.sum(mask)

            if positive_count == 0:
                x_index, y_index = (ccwh_bbox[:2] / STRIDES[best_label_index]).astype(np.int32)
                labels[best_label_index][y_index, x_index, best_anchor_index, :4] = gt_bbox
                labels[best_label_index][y_index, x_index, best_anchor_index, 4] = 1
                labels[best_label_index][y_index, x_index, best_anchor_index, 5:] = onehot

        slabel_data = labels[0].reshape((-1, 5 + CLASSES))
        mlabel_data = labels[1].reshape((-1, 5 + CLASSES))
        llabel_data = labels[2].reshape((-1, 5 + CLASSES))
        label_data = np.concatenate([slabel_data, mlabel_data, llabel_data], axis = 0)

        return image, label_data
    
    def Decode(self, pred_data, input_size, image_wh, detect_threshold = 0.5, use_nms = True):
        pred_data = pred_data
        mask = pred_data[:, 4] >= detect_threshold

        pred_confs = pred_data[mask, 4][:, np.newaxis]
        pred_bboxes = np.concatenate([pred_data[mask, :4], pred_confs], axis = -1)

        pred_classes = np.argmax(pred_data[mask, 5:], axis = -1)
        pred_bboxes[:, :4] = convert_bboxes(pred_bboxes[:, :4], image_wh = image_wh, ori_wh = [input_size, input_size])

        if use_nms:
            pred_bboxes, pred_classes = class_nms(pred_bboxes, pred_classes)

        return pred_bboxes, pred_classes
        
if __name__ == '__main__':

    utils = YOLOv3_Utils()

    for data in np.load('./dataset/validation_detection.npy', allow_pickle = True):
        image_name, gt_bboxes, gt_classes = data

        image_path = VALID_DIR + image_name
        gt_bboxes = np.asarray(gt_bboxes, dtype = np.float32)
        gt_classes = np.asarray([CLASS_DIC[c] for c in gt_classes], dtype = np.int32)

        image = cv2.imread(image_path)
        h, w, c = image.shape
        
        # print(w, h)
        # print(gt_bboxes[:, [0, 2]].min(), gt_bboxes[:, [0, 2]].max())
        # print(gt_bboxes[:, [1, 3]].min(), gt_bboxes[:, [1, 3]].max())
        # print(gt_bboxes.shape)

        input_size = 608
        image, label_data = utils.Encode(image_path, gt_bboxes, gt_classes, input_size)
        pred_bboxes, pred_classes = utils.Decode(label_data, input_size, [input_size, input_size])

        print(input_size)

        for pred_bbox, pred_class in zip(pred_bboxes, pred_classes):
            xmin, ymin, xmax, ymax = pred_bbox[:4].astype(np.int32)
            cv2.putText(image, CLASS_NAMES[pred_class], (xmin, ymin - 10), 1, 1, (0, 255, 0), 2)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        cv2.imshow('show', image)
        cv2.waitKey(0)

        # print(image.shape, label_data.keys())
        # print(image.shape, label_data.shape)
        # input()
        