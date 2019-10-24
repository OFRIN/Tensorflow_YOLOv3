
import numpy as np
import tensorflow as tf

import resnet_v1.resnet_v1 as resnet_v1

from Define import *

kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 0.01, seed = None)
bias_initializer = tf.constant_initializer(value = 0.0)

def group_normalization(x, is_training, G = 32, ESP = 1e-5, scope = 'group_norm'):
    with tf.variable_scope(scope):
        # 1. [N, H, W, C] -> [N, C, H, W]
        x = tf.transpose(x, [0, 3, 1, 2])
        N, C, H, W = x.shape.as_list()

        # 2. reshape (group normalization)
        G = min(G, C)
        x = tf.reshape(x, [-1, G, C // G, H, W])
        
        # 3. get mean, variance
        mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
        # 4. normalize
        x = (x - mean) / tf.sqrt(var + ESP)

        # 5. create gamma, bete
        gamma = tf.Variable(tf.constant(1.0, shape = [C]), dtype = tf.float32, name = 'gamma')
        beta = tf.Variable(tf.constant(0.0, shape = [C]), dtype = tf.float32, name = 'beta')
        
        gamma = tf.reshape(gamma, [1, C, 1, 1])
        beta = tf.reshape(beta, [1, C, 1, 1])

        # 6. gamma * x + beta
        x = tf.reshape(x, [-1, C, H, W]) * gamma + beta

        # 7. [N, C, H, W] -> [N, H, W, C]
        x = tf.transpose(x, [0, 2, 3, 1])
    return x

def conv_bn_relu(x, filters, kernel_size, strides, padding, is_training, scope, bn = True, activation = True, use_bias = True, upscaling = False):
    with tf.variable_scope(scope):
        if not upscaling:
            x = tf.layers.conv2d(inputs = x, filters = filters, kernel_size = kernel_size, strides = strides, padding = padding, kernel_initializer = kernel_initializer, use_bias = use_bias, name = 'conv2d')
        else:
            x = tf.layers.conv2d_transpose(inputs = x, filters = filters, kernel_size = kernel_size, strides = strides, padding = padding, kernel_initializer = kernel_initializer, use_bias = use_bias, name = 'upconv2d')
        
        if bn:
            # x = group_normalization(x, is_training = is_training, scope = 'gn')
            x = tf.layers.batch_normalization(x, training = is_training, name = 'bn')
        
        if activation:
            x = tf.nn.leaky_relu(x, name = 'leaky_relu')

    return x

def Decode_Layer(pred_tensors, anchor, stride, name):
    with tf.variable_scope(name):
        shape = tf.shape(pred_tensors)
        output_size = shape[1]

        x = tf.range(output_size, dtype = tf.int32)
        y = tf.range(output_size, dtype = tf.int32)
        x, y = tf.meshgrid(x, y)

        offset_xy = tf.transpose(tf.stack([x, y]), (2, 1, 0))
        offset_xy = tf.reshape(offset_xy, (1, output_size, output_size, 1, 2))
        offset_xy = tf.cast(offset_xy, tf.float32)
        
        pred_tensors = tf.reshape(pred_tensors, (-1, output_size, output_size, 3, 5 + CLASSES))
        dxdy = pred_tensors[:, :, :, :, :2]
        dwdh = pred_tensors[:, :, :, :, 2:4]
        conf = pred_tensors[:, :, :, :, 4]
        class_prob = pred_tensors[:, :, :, :, 5:]

        pred_cxcy = (tf.sigmoid(dxdy) + offset_xy) * stride
        pred_wh = tf.clip_by_value(tf.exp(dwdh) * anchor, 1., tf.cast(output_size, dtype = tf.float32))

        pred_lt = pred_cxcy - pred_wh / 2
        pred_rb = pred_cxcy + pred_wh / 2
        pred_conf = tf.expand_dims(tf.nn.sigmoid(conf), axis = -1)
        pred_class = tf.nn.sigmoid(class_prob)

        pred_tensors = tf.concat([pred_lt, pred_rb, pred_conf, pred_class], axis = -1)
        pred_tensors = tf.reshape(pred_tensors, (-1, output_size * output_size * 3, 5 + CLASSES), name = 'outputs')

    return pred_tensors

def YOLOv3(input_var, input_size, is_training, anchors, reuse = False):
    # convert BGR -> RGB
    x = tf.reshape(input_var, (-1, input_size, input_size, 3))
    x = x[..., ::-1] - MEAN
    
    with tf.contrib.slim.arg_scope(resnet_v1.resnet_arg_scope()):
        logits, end_points = resnet_v1.resnet_v1_50(x, is_training = is_training, reuse = reuse)
    
    # for key in end_points.keys():
    #     print(key, end_points[key])
    # input()

    pyramid_dic = {}
    feature_maps = [end_points['resnet_v1_50/block{}'.format(i)] for i in [4, 2, 1]]
    
    pyramid_dic['route_1'] = feature_maps[2] # stride = 8, 'route_1': <tf.Tensor 'resnet_v1_50/block1/unit_3/bottleneck_v1/Relu:0' shape=(?, ?, ?, 256) dtype=float32>
    pyramid_dic['route_2'] = feature_maps[1] # stride = 16, 'route_2': <tf.Tensor 'resnet_v1_50/block2/unit_4/bottleneck_v1/Relu:0' shape=(?, ?, ?, 512) dtype=float32>
    pyramid_dic['route_3'] = feature_maps[0] # stride = 32, 'route_3': <tf.Tensor 'resnet_v1_50/block4/unit_3/bottleneck_v1/Relu:0' shape=(?, ?, ?, 2048) dtype=float32>

    # print(pyramid_dic['route_1'])
    # print(pyramid_dic['route_2'])
    # print(pyramid_dic['route_3'])

    with tf.variable_scope('YOLOv3', reuse = reuse):

        with tf.variable_scope('Large'):
            x = pyramid_dic['route_3']

            x = conv_bn_relu(x, 512, [1, 1], 1, 'valid', is_training, 'conv1')
            x = conv_bn_relu(x, 1024, [3, 3], 1, 'same', is_training, 'conv2')
            x = conv_bn_relu(x, 512, [1, 1], 1, 'valid', is_training, 'conv3')
            x = conv_bn_relu(x, 1024, [3, 3], 1, 'same', is_training, 'conv4')
            x = conv_bn_relu(x, 512, [1, 1], 1, 'valid', is_training, 'conv5')

            x_large = conv_bn_relu(x, 1024, [3, 3], 1, 'same', is_training, 'conv6')
            x_large = conv_bn_relu(x_large, 3 * (5 + CLASSES), [1, 1], 1, 'valid', is_training, 'outputs', bn = False, activation = False)

            x = conv_bn_relu(x, 256, [3, 3], 2, 'same', is_training, 'upconv', upscaling = True)

        with tf.variable_scope('Medium'):
            x = tf.concat([x, pyramid_dic['route_2']], axis = -1)

            x = conv_bn_relu(x, 256, [1, 1], 1, 'valid', is_training, 'conv1')
            x = conv_bn_relu(x, 512, [3, 3], 1, 'same', is_training, 'conv2')
            x = conv_bn_relu(x, 256, [1, 1], 1, 'valid', is_training, 'conv3')
            x = conv_bn_relu(x, 512, [3, 3], 1, 'same', is_training, 'conv4')
            x = conv_bn_relu(x, 256, [1, 1], 1, 'valid', is_training, 'conv5')

            x_medium = conv_bn_relu(x, 512, [3, 3], 1, 'same', is_training, 'conv6')
            x_medium = conv_bn_relu(x_medium, 3 * (5 + CLASSES), [1, 1], 1, 'valid', is_training, 'outputs', bn = False, activation = False)

            x = conv_bn_relu(x, 128, [3, 3], 2, 'same', is_training, 'upconv', upscaling = True)

        with tf.variable_scope('Small'):
            x = tf.concat([x, pyramid_dic['route_1']], axis = -1)

            x = conv_bn_relu(x, 128, [1, 1], 1, 'valid', is_training, 'conv1')
            x = conv_bn_relu(x, 256, [3, 3], 1, 'same', is_training, 'conv2')
            x = conv_bn_relu(x, 128, [1, 1], 1, 'valid', is_training, 'conv3')
            x = conv_bn_relu(x, 256, [3, 3], 1, 'same', is_training, 'conv4')
            x = conv_bn_relu(x, 128, [1, 1], 1, 'valid', is_training, 'conv5')

            x_small = conv_bn_relu(x, 256, [3, 3], 1, 'same', is_training, 'conv6')
            x_small = conv_bn_relu(x_small, 3 * (5 + CLASSES), [1, 1], 1, 'valid', is_training, 'outputs', bn = False, activation = False)
        
        '''
        # 320x320
        Tensor("YOLOv3/Small/outputs/conv2d/BiasAdd:0", shape=(?, 40, 40, 255), dtype=float32)
        Tensor("YOLOv3/Medium/outputs/conv2d/BiasAdd:0", shape=(?, 20, 20, 255), dtype=float32)
        Tensor("YOLOv3/Large/outputs/conv2d/BiasAdd:0", shape=(?, 10, 10, 255), dtype=float32)
        '''
        # print(x_small)
        # print(x_medium)
        # print(x_large)

        x_small = Decode_Layer(x_small, anchors[0], STRIDES[0], 'Decode_Small')
        x_medium = Decode_Layer(x_medium, anchors[1], STRIDES[1], 'Decode_Medium')
        x_large = Decode_Layer(x_large, anchors[2], STRIDES[2], 'Decode_Large')

        x = tf.concat([x_small, x_medium, x_large], axis = 1)

    return x

if __name__ == '__main__':
    from YOLOv3_Utils import *

    utils = YOLOv3_Utils()

    input_var = tf.placeholder(tf.float32)
    input_size = tf.placeholder(tf.int32)

    output = YOLOv3(input_var, input_size, False, utils.anchors, reuse = False)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    '''
    (1, 6300, 85)
    (1, 7623, 85)
    (1, 9072, 85)
    (1, 10647, 85)
    (1, 12348, 85)
    (1, 14175, 85)
    (1, 16128, 85)
    (1, 18207, 85)
    (1, 20412, 85)
    (1, 22743, 85)
    '''
    test_sizes = np.linspace(MIN_INPUT_SIZE, MAX_INPUT_SIZE, INPUT_SIZE_COUNT).astype(np.int32)
    for test_size in test_sizes:
        image = np.random.randint(0, 255, test_size * test_size * 3).reshape((1, test_size, test_size, 3))

        t = sess.run(output, feed_dict = {input_var : image, input_size : test_size})
        print(t.shape)
    
