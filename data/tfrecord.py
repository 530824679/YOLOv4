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

    # 数值形式的数据,首先转换为string,再转换为int形式进行保存
    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    # 数组形式的数据,首先转换为string,再转换为二进制形式进行保存
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
                label = self.dataset.load_bev_label(index)

                if len(label) == 0:
                    continue

                y_true_13, y_true_26, y_true_52 = self.dataset.preprocess_true_boxes(label)
                image_string = image.tostring()
                y_true_13_string = y_true_13.tostring()
                y_true_26_string = y_true_26.tostring()
                y_true_52_string = y_true_52.tostring()

                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
                        'y_true_13': tf.train.Feature(bytes_list=tf.train.BytesList(value=[y_true_13_string])),
                        'y_true_26': tf.train.Feature(bytes_list=tf.train.BytesList(value=[y_true_26_string])),
                        'y_true_52': tf.train.Feature(bytes_list=tf.train.BytesList(value=[y_true_52_string]))
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
                'label': tf.FixedLenFeature([], tf.string)
            })

        image = features['image']
        label = features['label']

        # 进行解码
        tf_image = tf.decode_raw(image, tf.float32)
        tf_label = tf.decode_raw(label, tf.float32)

        # 转换为网络输入所要求的形状
        tf_image = tf.reshape(tf_image, [self.input_height, self.input_width, self.channels])
        tf_label = tf.reshape(tf_label, [-1, 7])

        # preprocess
        tf_image = tf_image / 255
        y_true_13, y_true_26, y_true_52 = tf.py_func(self.dataset.preprocess_true_boxes, inp=[tf_label], Tout = [tf.float32, tf.float32, tf.float32])

        return tf_image, [y_true_13, y_true_26, y_true_52]

    def create_dataset(self, filenames, batch_size=8, is_shuffle=False, n_repeats=0):
        """
        :param filenames: record file names
        :param batch_size: batch size
        :param is_shuffle: whether shuffle
        :param n_repeats: number of repeats
        :return:
        """
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.repeat(n_repeats)
        dataset = dataset.map(lambda x: self.parse_single_example(x), num_parallel_calls = 4)
        if is_shuffle:
            dataset = dataset.shuffle(100)
        dataset = dataset.batch(batch_size)
        return dataset

if __name__ == '__main__':
    tfrecord = TFRecord()
    tfrecord.create_tfrecord()

    # file = './tfrecord/train.tfrecord'
    # tfrecord = TFRecord()
    # batch_example, batch_label = tfrecord.parse_batch_examples(file)
    # with tf.Session() as sess:
    #
    #     init_op = tf.global_variables_initializer()
    #     sess.run(init_op)
    #
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #     for i in range(1):
    #         example, label = sess.run([batch_example, batch_label])
    #         print(label)
    #         print(label.astype(np.float32))
    #         box = label[0, ]
    #         # cv2.imshow('w', example[0, :, :, :])
    #         # cv2.waitKey(0)
    #         print(np.shape(example), np.shape(label))
    #     # cv2.imshow('img', example)
    #     # cv2.waitKey(0)
    #     # print(type(example))
    #     coord.request_stop()
    #     # coord.clear_stop()
    #     coord.join(threads)