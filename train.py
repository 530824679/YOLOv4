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
    input_height = model_params['input_height']
    input_width = model_params['input_width']
    total_epoches = solver_params['total_epoches']
    warm_up_epoch = solver_params['warm_up_epoch']
    warm_up_lr = solver_params['warm_up_lr']
    val_step = solver_params['val_step']
    log_step = solver_params['log_step']
    display_step = solver_params['display_step']
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

    # 定义输入的占位符
    handle_flag = tf.placeholder(tf.string, [], name='iterator_handle_flag')

    # 解析得到训练样本以及标注
    data = tfrecord.TFRecord()
    train_tfrecord = os.path.join(tfrecord_dir, train_tfrecord_name)
    val_tfrecord = os.path.join(tfrecord_dir, val_tfrecord_name)
    train_dataset = data.create_dataset(train_tfrecord, batch_size=batch_size, is_shuffle=True, n_repeats=0)
    val_dataset = data.create_dataset(val_tfrecord, batch_size=batch_size, is_shuffle=False, n_repeats=-1)

    # 设置预取
    train_dataset = train_dataset.prefetch(1)
    val_dataset = val_dataset.prefetch(1)

    # 创建训练和验证数据迭代器
    train_iterator = train_dataset.make_initializable_iterator()
    val_iterator = val_dataset.make_initializable_iterator()

    # 创建训练和验证数据句柄
    train_handle = train_iterator.string_handle()
    val_handle = val_iterator.string_handle()

    dataset_iterator = tf.data.Iterator.from_string_handle(handle_flag, train_dataset.output_types, train_dataset.output_shapes)
    train_images, y_true_19, y_true_38, y_true_76 = dataset_iterator.get_next()
    y_true = [y_true_19, y_true_38, y_true_76]

    # tf.data pipeline will lose the data shape, so we need to set it manually
    train_images.set_shape([None, input_height, input_width, 2])
    for y in y_true:
        y.set_shape([None, None, None, None, None])

    # 构建网络
    network = Network(is_train=True)
    logits, preds = network.forward(train_images)

    # 计算损失函数
    losses = Loss()
    loss_op = losses.calc_loss(logits, preds, y_true, 'loss')

    vars = tf.trainable_variables()
    l2_reg_loss_op = tf.add_n([tf.nn.l2_loss(var) for var in vars]) * solver_params['weight_decay']
    total_loss = loss_op[0] + l2_reg_loss_op

    # 配置tensorboard
    tf.summary.scalar("ciou_loss", loss_op[1])
    tf.summary.scalar("reim_loss", loss_op[2])
    tf.summary.scalar("confs_loss", loss_op[3])
    tf.summary.scalar("class_loss", loss_op[4])
    tf.summary.scalar("recall_50", loss_op[5])
    tf.summary.scalar("recall_70", loss_op[6])
    tf.summary.scalar("avg_iou", loss_op[7])
    tf.summary.scalar('total_loss', total_loss)

    # 创建全局的步骤
    global_step = tf.Variable(0, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    # 设定变化的学习率
    learning_rate_exp = tf.train.exponential_decay(
        solver_params['learning_rate'],
        global_step,
        solver_params['decay_steps'],
        solver_params['decay_rate'],
        solver_params['staircase'],
        name='learning_rate')

    learning_rate = tf.cond(tf.less(global_step, batch_size * warm_up_epoch), lambda: warm_up_lr, lambda: learning_rate_exp)
    tf.summary.scalar('learning_rate', learning_rate)

    # 设置优化器
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        train_op = slim.learning.create_train_op(total_loss, optimizer, global_step)

    # 模型保存
    save_variable = tf.global_variables()
    saver = tf.train.Saver(save_variable, max_to_keep=1000)

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph(), flush_secs=60)

    with tf.Session(config=config) as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), train_iterator.initializer])
        train_handle_value, val_handle_value = sess.run([train_handle, val_handle])
        summary_writer.add_graph(sess.graph)

        print('\n----------- start to train -----------\n')
        for epoch in range(total_epoches):
            try:
                _, summary, loss_, global_step_, lr = sess.run([train_op, summary_op, loss_op, global_step, learning_rate], feed_dict={handle_flag: train_handle_value})
                summary_writer.add_summary(summary, global_step=global_step_)

                if epoch % solver_params['save_step'] == 0 and epoch > 0:
                    save_path = saver.save(sess, os.path.join(checkpoint_dir, checkpoints_name), global_step=epoch)
                    print('Save modle into {}....'.format(save_path))

                if epoch % log_step == 0 and epoch > 0:
                    summary_writer.add_summary(summary, global_step=epoch)

                if epoch % display_step == 0 and epoch > 0:
                    print("Epoch: {}, global_step: {}, lr: {:.8f}, total_loss: {:.3f}, loss_ciou: {:.3f}, loss_reim: {:.3f}, loss_conf: {:.3f}, loss_class: {:.3f}, recall50: {:.3f}, recall75: {:.3f}, avg_iou: {:.3f}".format(
                    epoch, global_step_, lr, loss_[0], loss_[1], loss_[2], loss_[3], loss_[4], loss_[5], loss_[6], loss_[7]))

                # 评估模型
                # if epoch % val_step == 0 and epoch > 0:
                #     sess.run(val_iterator.initializer)
                #     true_positive_dict, true_labels_dict, pred_labels_dict = {}, {}, {}
                #     val_loss = [0., 0., 0., 0., 0.]
                #     for j in range(args.val_batch_num):
                #         y_pred_, y_true_, loss_ = sess.run([y_pred, y_true, loss], feed_dict={is_train: False, handle_flag: val_handle_value})
                #         true_positive_dict_tmp, true_labels_dict_tmp, pred_labels_dict_tmp = evaluate_on_gpu(sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag, y_pred_, y_true_, args.class_num, calc_now=False)
                #         true_positive_dict = update_dict(true_positive_dict, true_positive_dict_tmp)
                #         true_labels_dict = update_dict(true_labels_dict, true_labels_dict_tmp)
                #         pred_labels_dict = update_dict(pred_labels_dict, pred_labels_dict_tmp)
                #
                #         val_loss = list_add(val_loss, loss_)
                #
                #     # make sure there is at least one ground truth object in each image
                #     # avoid divided by 0
                #     recall = float(sum(true_positive_dict.values())) / (sum(true_labels_dict.values()) + 1e-6)
                #     precision = float(sum(true_positive_dict.values())) / (sum(pred_labels_dict.values()) + 1e-6)
                #
                #     info = "===> Epoch: {}, global_step: {}, recall: {:.3f}, precision: {:.3f}, total_loss: {:.3f}, loss_xy: {:.3f}, loss_wh: {:.3f}, loss_conf: {:.3f}, loss_class: {:.3f}".format(
                #         epoch, global_step_, recall, precision, val_loss[0] / args.val_batch_num,
                #                                                 val_loss[1] / args.val_batch_num,
                #                                                 val_loss[2] / args.val_batch_num,
                #                                                 val_loss[3] / args.val_batch_num,
                #                                                 val_loss[4] / args.val_batch_num)

                sess.run(train_iterator.initializer)
            except tf.errors.OutOfRangeError:
                break

        sess.close()

if __name__ == '__main__':
    train()