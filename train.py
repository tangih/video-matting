import tensorflow as tf
import numpy as np
import cv2
import random
import os
import time

import loader
import unet
import params


def composite(fg, bg, alpha):
    """ create composite image from tensors """
    tri_alpha = tf.concat(values=[alpha, alpha, alpha], axis=3)
    cmp = tf.add(tf.multiply(tri_alpha, fg), tf.multiply(1. - tri_alpha, bg))
    return cmp


def regular_l1(output, gt, name):
    """ regularized L1 loss """
    sq_loss = tf.square(tf.subtract(output, gt))
    eps2 = tf.square(tf.constant(1e-6))
    loss = tf.sqrt(sq_loss + eps2, name=name)
    return loss


def training_procedure(sess, x, gt, raw_fg, train_file_list, test_file_list, pred,
                       train_writer, test_writer, ex_writer,
                       saver, t_str, starting_point=0):
    in_cmp, in_bg = tf.split(value=x, num_or_size_splits=[3, 3], axis=3)
    with tf.variable_scope('loss'):
        alpha_loss = regular_l1(pred, gt, name='alpha_loss')
        pred_cmp = composite(raw_fg, in_bg, pred)
        cmp_loss = regular_l1(pred_cmp, in_cmp, name='compositional_loss')
        s_loss = tf.add(0.5 * alpha_loss, 0.5 * cmp_loss)
        loss = tf.reduce_mean(s_loss, name='loss')
    with tf.variable_scope('training'):
        lr = 1e-4
        print('Training with learning rate of {}'.format(lr))
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-08)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    with tf.variable_scope('summary'):
        summary_loss = tf.summary.scalar('loss', loss)
        summary_alpha_loss = tf.summary.scalar('alpha_loss', tf.reduce_mean(alpha_loss))
        summary_cmp_loss = tf.summary.scalar('compositional_loss', tf.reduce_mean(cmp_loss))
        summary_cmp = tf.summary.image('composite', in_cmp)
        summary_gt = tf.summary.image('ground_truth', gt)
        summary_pred = tf.summary.image('prediction', pred)
    train_summaries = [summary_loss, summary_alpha_loss, summary_cmp_loss]
    test_summaries = [summary_loss, summary_alpha_loss, summary_cmp_loss]
    ex_summaries = [summary_cmp, summary_gt, summary_pred]
    train_merged = tf.summary.merge(train_summaries)
    test_merged = tf.summary.merge(test_summaries)
    ex_merged = tf.summary.merge(ex_summaries)
    prev_val_loss = -1.
    iteration = starting_point
    sess.run(tf.global_variables_initializer())

    for epoch in range(params.N_EPOCHS):
        training_list = train_file_list.copy()
        test_list = test_file_list.copy()
        random.shuffle(training_list)
        random.shuffle(test_list)
        # training
        while not loader.epoch_is_over(training_list, params.BATCH_SIZE):
            print('Training model, epoch {}/{}, iteration {}.'.format(epoch + 1, params.N_EPOCHS, iteration + 1))
            batch_list = loader.get_batch_list(training_list, params.BATCH_SIZE)
            inp, lab, rfg = loader.get_batch(batch_list, params.INPUT_SIZE, rd_scale=False, rd_mirror=True)
            feed_dict = {x: inp, gt: lab, raw_fg: rfg}
            summary, _ = sess.run([train_merged, train_op], feed_dict=feed_dict)
            train_writer.add_summary(summary, iteration)
            iteration += 1
        # validation
        print('Training completed. Computing validation loss...')
        val_loss = 0.
        n_batch = 0
        while not loader.epoch_is_over(test_list, params.BATCH_SIZE):
            batch_list = loader.get_batch_list(test_list, params.BATCH_SIZE)
            inp, lab, rfg = loader.get_batch(batch_list, params.INPUT_SIZE, rd_scale=False, rd_mirror=True)
            feed_dict = {x: inp, gt: lab, raw_fg: rfg}
            ls = sess.run([loss], feed_dict=feed_dict)
            # test_writer.add_summary(summary, iteration)
            val_loss += np.mean(ls)
            n_batch += 1
        val_loss /= n_batch
        improvement = 'NaN'
        if prev_val_loss != -1.:
            improvement = '{:2f}%'.format((prev_val_loss - val_loss) / prev_val_loss)
        print('Validation loss: {:.3f}. Improvement: {}'.format(val_loss, improvement))
        print('Saving examples')
        # loads and visualize example prediction of current model
        n_ex = 5
        ex_list = [test_file_list[np.random.randint(0, len(test_file_list))] for _ in range(n_ex)]
        ex_inp, ex_lab, _ = loader.get_batch(ex_list, params.INPUT_SIZE, rd_scale=False, rd_mirror=True)
        feed_dict = {x: ex_inp, gt: ex_lab}
        summary = sess.run([ex_merged], feed_dict)[0]
        ex_writer.add_summary(summary, iteration)
        print('Saving chekpoint...')
        saver.save(sess, os.path.join(params.LOG_DIR, 'weights_{}'.format(t_str), 'model'), global_step=iteration)


def train():
    """ train UNet network from scratch """
    model = unet.UNetImage()
    with tf.variable_scope('input'):
        x = tf.placeholder('float', [None, params.INPUT_SIZE[0], params.INPUT_SIZE[1], 6], name='input')
        raw_fg = tf.placeholder('float', [None, params.INPUT_SIZE[0], params.INPUT_SIZE[1], 3], name='raw_fg')
        gt = tf.placeholder('float', [None, params.INPUT_SIZE[0], params.INPUT_SIZE[1], 1], name='gt')

    with tf.variable_scope('model'):
        model.build(x)
        pred = model.output

    train_file_list = loader.get_file_list(params.SYNTHETIC_DATASET, params.TRAINING_LIST)
    test_file_list = loader.get_file_list(params.SYNTHETIC_DATASET, params.TRAINING_LIST)

    with tf.Session() as sess:
        t_str = time.asctime().replace(' ', '_')
        train_writer = tf.summary.FileWriter(os.path.join(params.LOG_DIR, 'train_{}'.format(t_str)), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(params.LOG_DIR, 'test_{}'.format(t_str)))
        ex_writer = tf.summary.FileWriter(os.path.join(params.LOG_DIR, 'examples_{}'.format(t_str)))
        saver = tf.train.Saver()
        training_procedure(sess, x, gt, raw_fg, train_file_list, test_file_list, pred,
                           train_writer, test_writer, ex_writer,
                           saver, t_str, starting_point=0)


def resume_training(meta_path, weight_folder):
    """ resume training from checkpoint """
    train_file_list = loader.get_file_list(params.SYNTHETIC_DATASET, params.TRAINING_LIST)
    test_file_list = loader.get_file_list(params.SYNTHETIC_DATASET, params.TRAINING_LIST)
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(meta_path)
        new_saver.restore(sess, tf.train.latest_checkpoint(weight_folder))
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('input/input:0')
        gt = graph.get_tensor_by_name('input/gt:0')
        raw_fg = graph.get_tensor_by_name('input/raw_fg:0')
        pred = graph.get_tensor_by_name('model/output:0')

        t_str = time.asctime().replace(' ', '_')
        train_writer = tf.summary.FileWriter(os.path.join(params.LOG_DIR, 'train_{}'.format(t_str)), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(params.LOG_DIR, 'test_{}'.format(t_str)))
        ex_writer = tf.summary.FileWriter(os.path.join(params.LOG_DIR, 'examples_{}'.format(t_str)))
        saver = tf.train.Saver()
        training_procedure(sess, x, gt, raw_fg, train_file_list, test_file_list, pred,
                           train_writer, test_writer, ex_writer,
                           saver, t_str, starting_point=0)


if __name__ == '__main__':
    train()
