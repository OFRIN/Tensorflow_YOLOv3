# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

# dataset parameters
ROOT_DIR = 'D:/_ImageDataset/COCO/'
TRAIN_DIR = ROOT_DIR + 'train2017/image/'
VALID_DIR = ROOT_DIR + 'valid2017/image/'

CLASS_NAMES = [class_name.strip() for class_name in open('./coco/label_names.txt').readlines()]
CLASS_DIC = {class_name : index for index, class_name in enumerate(CLASS_NAMES)}
CLASSES = len(CLASS_NAMES)

# network parameters
MIN_INPUT_SIZE = 320
MAX_INPUT_SIZE = 608
INPUT_SIZE_COUNT = 10

TEST_INPUT_SIZE = 416

STRIDES = [8, 16, 32]
ANCHORS = [10, 13, 16, 30, 33, 23,      # Small
           30, 61, 62, 45, 59, 119,     # Medium
           116, 90, 156, 198, 373, 326] # Large

IOU_THRESHOLD = 0.3
NMS_THRESHOLD = 0.5

R_MEAN = 123.68
G_MEAN = 116.78
B_MEAN = 103.94
MEAN = [R_MEAN, G_MEAN, B_MEAN]

# loss parameters
WEIGHT_DECAY = 0.0001

# train
# use thread (Dataset)
NUM_THREADS = 8

# single gpu training
GPU_INFO = "0"
NUM_GPU = len(GPU_INFO.split(','))

BATCH_SIZE = 4 * NUM_GPU
INIT_LEARNING_RATE = 1e-4

# iteration & learning rate schedule
MAX_ITERATION = 200000
DECAY_ITERATIONS = [100000, 160000]

SAMPLES = BATCH_SIZE

LOG_ITERATION = 50
SAMPLE_ITERATION = 5000
SAVE_ITERATION = 5000

# color_list (OpenCV - BGR)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)

COLOR_PBLUE = (204, 72, 63)
COLOR_ORANGE = (0, 128, 255)
