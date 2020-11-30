# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : tfrecord.py
# Description :create and parse tfrecord
# --------------------------------------

import os
import numpy as np
import tensorflow as tf
from data.dataset import Dataset
from cfg.config import path_params, model_params, solver_params, classes_map

class TFRecord(object):
    def __init__(self):
        self.data_path = path_params['data_path']
        self.tfrecord_dir = path_params['tfrecord_dir']
        self.train_tfrecord_name = path_params['train_tfrecord_name']
        self.val_tfrecord_name = path_params['val_tfrecord_name']
        self.input_width = model_params['input_width']
        self.input_height = model_params['input_height']
        self.channels = model_params['channels']
        self.class_num = len(model_params['classes'])
        self.batch_size = solver_params['batch_size']
        self.dataset = Dataset()

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def create_tfrecord(self):
        # 获取作为训练验证集的图片序列
        trainval_path = os.path.join(self.data_path, 'ImageSets', 'Main', 'trainval.txt')
        split_ratio = 0.8
        train_file = os.path.join(self.tfrecord_dir, self.train_tfrecord_name)
        val_file = os.path.join(self.tfrecord_dir, self.val_tfrecord_name)
        if os.path.exists(train_file):
            os.remove(train_file)
        if os.path.exists(val_file):
            os.remove(val_file)

        # 循环写入每一帧点云转换的bev和标签到tfrecord文件
        train_writer = tf.python_io.TFRecordWriter(train_file)
        val_writer = tf.python_io.TFRecordWriter(val_file)
        with open(trainval_path, 'r') as read:
            lines = read.readlines()
            train_sample_num = len(lines) * split_ratio
            for count, line in enumerate(lines):
                index = line[0:-1]
                image = self.dataset.load_bev_image(index)
                bbox = self.dataset.load_bev_label(index)

                if len(bbox) == 0:
                    continue

                image_string = image.tobytes()
                bbox_string = bbox.tobytes()
                bbox_shape = bbox.shape

                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
                        'bbox': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bbox_string])),
                        'bbox_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=bbox_shape))
                    }))
                if count < train_sample_num:
                    train_writer.write(example.SerializeToString())
                else:
                    val_writer.write(example.SerializeToString())
        train_writer.close()
        val_writer.close()
        print('Finish trainval.tfrecord Done')

    def parse_single_example(self, serialized_example):
        """
        :param serialized_example:待解析的tfrecord文件的名称
        :return: 从文件中解析出的单个样本的相关特征，image, label
        """
        # 解析单个样本文件
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'bbox': tf.FixedLenFeature([], tf.string),
                'bbox_shape': tf.FixedLenFeature(shape=(2,), dtype=tf.int64)
            })

        image = features['image']
        bbox = features['bbox']
        bbox_shape = features['bbox_shape']

        # 进行解码
        tf_image = tf.decode_raw(image, tf.float32)
        tf_bbox = tf.decode_raw(bbox, tf.float32)

        # 转换为网络输入所要求的形状
        tf_image = tf.reshape(tf_image, [self.input_height, self.input_width, self.channels])
        tf_label = tf.reshape(tf_bbox, bbox_shape)

        # preprocess
        tf_image = tf_image / 255
        y_true_19, y_true_38, y_true_76 = tf.py_func(self.dataset.preprocess_true_boxes, inp=[tf_label], Tout = [tf.float32, tf.float32, tf.float32])

        return tf_image, y_true_19, y_true_38, y_true_76

    def create_dataset(self, filenames, batch_size=1, is_shuffle=False, n_repeats=0):
        """
        :param filenames: record file names
        :param batch_size: batch size
        :param is_shuffle: whether shuffle
        :param n_repeats: number of repeats
        :return:
        """
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.repeat(n_repeats)
        dataset = dataset.map(self.parse_single_example, num_parallel_calls = 1)
        if is_shuffle:
            dataset = dataset.shuffle(10)
        dataset = dataset.batch(batch_size)
        return dataset

if __name__ == '__main__':
    tfrecord = TFRecord()
    #tfrecord.create_tfrecord()

    import cv2
    import utils.visualize as v
    record_file = '/home/chenwei/HDD/livox_dl/LIVOX1/tfrecord/train.tfrecord'
    data_train = tfrecord.create_dataset(record_file, batch_size=2, is_shuffle=False, n_repeats=20)
    # data_train = tf.data.TFRecordDataset(record_file)
    # data_train = data_train.map(tfrecord.parse_single_example)
    iterator = data_train.make_one_shot_iterator()
    batch_image, y_true_19, y_true_38, y_true_76 = iterator.get_next()

    with tf.Session() as sess:
        for i in range(20):
            try:
                image, true_19, true_38, true_76 = sess.run([batch_image, y_true_19, y_true_38, y_true_76])

                # for boxes in label:
                #     v.draw_rotated_box(image, int(boxes[0]), int(boxes[1]), int(boxes[2]), int(boxes[3]), boxes[5],
                #                        (255, 0, 0))
                # cv2.imshow("image", image)
                # cv2.waitKey(0)
                print(np.shape(image))
            except tf.errors.OutOfRangeError:
                print("Done!!!")
                break

    # import cv2
    # import utils.visualize as v
    # record_file = '/home/chenwei/HDD/livox_dl/LIVOX1/tfrecord/train.tfrecord'
    # reader = tf.TFRecordReader()
    # filename_queue = tf.train.string_input_producer([record_file])
    #
    # _, serialized_example = reader.read(filename_queue)
    # batch_image, y_true_19, y_true_38, y_true_76 = tfrecord.parse_single_example(serialized_example)
    # with tf.Session() as sess:
    #
    #     init_op = tf.global_variables_initializer()
    #     sess.run(init_op)
    #
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #     for i in range(20):
    #         image, true_19, true_38, true_76 = sess.run([batch_image, y_true_19, y_true_38, y_true_76])
    #
    #         # for boxes in label:
    #         #     v.draw_rotated_box(image, int(boxes[0]), int(boxes[1]), int(boxes[2]), int(boxes[3]), boxes[5], (255, 0, 0))
    #         # cv2.imshow("image", image)
    #         # cv2.waitKey(0)
    #         print(np.shape(image))
    #     coord.request_stop()
    #     coord.join(threads)