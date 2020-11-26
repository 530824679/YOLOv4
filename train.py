# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : train.py
# Description : train code
# --------------------------------------

import os
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim

from cfg.config import path_params, model_params, solver_params
from model.loss import Loss
from model.network import Network
from data import dataset, tfrecord


def train():
    start_step = 0
    log_step = solver_params['log_step']
    display_step = solver_params['display_step']
    restore = solver_params['restore']
    batch_size = solver_params['batch_size']
    checkpoint_dir = path_params['checkpoints_dir']
    checkpoints_name = path_params['checkpoints_name']
    tfrecord_dir = path_params['tfrecord_dir']
    train_tfrecord_name = path_params['train_tfrecord_name']
    val_tfrecord_name = path_params['val_tfrecord_name']
    log_dir = path_params['logs_dir']

    # 配置GPU
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)

    # 解析得到训练样本以及标注
    data = tfrecord.TFRecord()
    train_tfrecord = os.path.join(tfrecord_dir, train_tfrecord_name)
    val_tfrecord = os.path.join(tfrecord_dir, val_tfrecord_name)
    train_dataset = data.create_dataset(train_tfrecord, batch_size=batch_size, is_shuffle=True, n_repeats=0)
    val_dataset = data.create_dataset(val_tfrecord, batch_size=batch_size, is_shuffle=False, n_repeats=-1)
    train_iterator = train_dataset.make_initializable_iterator()
    val_iterator = val_dataset.make_one_shot_iterator()
    train_images, train_labels = train_iterator.get_next()
    val_images, val_labels = val_iterator.get_next()

    # 定义输入的占位符
    #inputs = tf.placeholder(dtype=tf.float32, shape=[None, model_params['image_height'], model_params['image_width'], model_params['channels']], name='inputs')
    #labels = tf.placeholder(dtype=tf.float32, shape=[None, model_params['grid_height'], model_params['grid_width'], model_params['anchor_num'], 8], name='labels')

    # 构建网络
    network = Network(is_train=True)
    logits, preds = network.forward(train_images)

    # 计算损失函数
    losses = Loss()
    loss_list = losses.calc_loss(logits, preds, train_labels, 'loss')
    loss_op = tf.losses.get_total_loss()

    vars = tf.trainable_variables()
    l2_reg_loss_op = tf.add_n([tf.nn.l2_loss(var) for var in vars]) * solver_params['weight_decay']
    total_loss = loss_op + l2_reg_loss_op

    # 配置tensorboard
    tf.summary.scalar("giou_loss", loss_list[0])
    tf.summary.scalar("confs_loss", loss_list[1])
    tf.summary.scalar("class_loss", loss_list[2])
    tf.summary.scalar('total_loss', total_loss)

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph(), flush_secs=60)

    # 创建全局的步骤
    global_step = tf.train.create_global_step()
    # 设定变化的学习率
    learning_rate = tf.train.exponential_decay(
        solver_params['learning_rate'],
        global_step,
        solver_params['decay_steps'],
        solver_params['decay_rate'],
        solver_params['staircase'],
        name='learning_rate')

    # 设置优化器
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        train_op = slim.learning.create_train_op(total_loss, optimizer, global_step)

    # 模型保存
    save_variable = tf.global_variables()
    saver = tf.train.Saver(save_variable, max_to_keep=1000)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(train_iterator.initializer)

        summary_writer.add_graph(sess.graph)

        while True:
            try:
                start_time = time.time()

                batch_images, batch_labels = sess.run([train_images, train_labels])
                feed_dict = {inputs: image, outputs: label}
                _, loss, current_global_step = sess.run([train_op, total_loss, global_step], feed_dict=feed_dict)

                end_time = time.time()

                if epoch % solver_params['save_step'] == 0:
                    save_path = saver.save(sess, os.path.join(checkpoint_dir, checkpoints_name), global_step=epoch)
                    print('Save modle into {}....'.format(save_path))

                if epoch % log_step == 0:
                    summary = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary, global_step=epoch)

                if epoch % display_step == 0:
                    per_iter_time = end_time - start_time
                    print("step:{:.0f}  total_loss:  {:.5f} {:.2f} s/iter".format(epoch, loss, per_iter_time))
            except tf.errors.OutOfRangeError:
                break

        sess.close()

if __name__ == '__main__':
    train()