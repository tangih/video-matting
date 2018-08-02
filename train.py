import tensorflow as tf
import numpy as np
import cv2
import random
import os
import time

import loader
import unet
import params
import unet_simple

def composite(fg, bg, alpha):
    """ create composite image from tensors """
    tri_alpha = tf.concat(values=[alpha, alpha, alpha], axis=3)
    cmp = tf.add(tf.multiply(tri_alpha, fg), tf.multiply(1. - tri_alpha, bg))
    return cmp


def regular_l1(output, gt, name):
    """ regularized L1 loss """
    print(output)
    print(gt)
    sq_loss = tf.square(tf.subtract(output, gt))
    eps2 = tf.square(tf.constant(1e-6))
    loss = tf.sqrt(sq_loss + eps2, name=name)
    return loss


def bgr2rgb(bgr):
    blue, green, red = tf.split(value=bgr, num_or_size_splits=3, axis=-1)
    image = tf.concat([red, green, blue], axis=-1)
    return image


def training_procedure(sess, x, gt, raw_fg, train_file_list, test_file_list, pred,
                       train_writer, test_writer, ex_writer,
                       saver, t_str, starting_point=0):
    improvement = 'NaN'
    in_cmp, in_bg = tf.split(value=x, num_or_size_splits=[3, 3], axis=3)
    with tf.variable_scope('loss'):
        alpha_loss = regular_l1(pred, gt, name='alpha_loss')
        pred_cmp = composite(raw_fg, in_bg, pred)
        cmp_loss = regular_l1(pred_cmp, in_cmp, name='compositional_loss')
        s_loss = tf.add(0.5 * alpha_loss, 0.5 * cmp_loss)
        loss = tf.reduce_mean(s_loss, name='loss')
    with tf.variable_scope('resume_training'):
        lr = 1e-5
        print('Training with learning rate of {}'.format(lr))
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-08)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    with tf.variable_scope('summary'):
        summary_loss = tf.summary.scalar('loss', loss)
        summary_alpha_loss = tf.summary.scalar('alpha_loss', tf.reduce_mean(alpha_loss))
        summary_cmp_loss = tf.summary.scalar('compositional_loss', tf.reduce_mean(cmp_loss))
        summary_cmp = tf.summary.image('composite', bgr2rgb(in_cmp))
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


def simple_procedure(sess, in_cmp, in_bg, gt, raw_fg, phase, pred, train_writer, ex_writer, saver,
                     train_file_list, test_file_list):
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'model/simple_unet')
    print('Training variables:')
    print(train_vars)
    improvement = 'NaN'
    t_str = time.asctime().replace(' ', '_')
    month = t_str.split('_')[1]
    date = int(t_str.split('_')[3])
    with tf.variable_scope('loss'):
        alpha_loss = regular_l1(pred, gt, name='alpha_loss')
        pred_cmp = composite(raw_fg, in_bg, pred)
        cmp_loss = regular_l1(pred_cmp, in_cmp, name='compositional_loss')
        s_loss = tf.add(0.5 * alpha_loss, 0.5 * cmp_loss)
        loss = tf.reduce_mean(s_loss, name='loss')
    with tf.variable_scope('resume_training_{}_{}'.format(month, date)):
        lr = 1e-4
        print('Training with learning rate of {}'.format(lr))
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-08)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step(), var_list=train_vars)
    with tf.variable_scope('summary'):
        summary_loss = tf.summary.scalar('loss', loss)
        summary_alpha_loss = tf.summary.scalar('alpha_loss', tf.reduce_mean(alpha_loss))
        summary_cmp_loss = tf.summary.scalar('compositional_loss', tf.reduce_mean(cmp_loss))
        summary_cmp = tf.summary.image('composite', bgr2rgb(in_cmp))
        summary_gt = tf.summary.image('ground_truth', gt)
        summary_pred = tf.summary.image('prediction', pred)
    train_summaries = [summary_loss, summary_alpha_loss, summary_cmp_loss]
    ex_summaries = [summary_cmp, summary_gt, summary_pred]
    train_merged = tf.summary.merge(train_summaries)
    ex_merged = tf.summary.merge(ex_summaries)
    prev_val_loss = -1.
    iteration = 0
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
            cmp, bg, lab, rfg = loader.simple_batch(batch_list, params.INPUT_SIZE)
            feed_dict = {in_cmp: cmp, in_bg: bg, gt: lab, raw_fg: rfg, phase: True}
            summary, _ = sess.run([train_merged, train_op], feed_dict=feed_dict)
            train_writer.add_summary(summary, iteration)
            iteration += 1
        # validation
        print('Training completed. Computing validation loss...')
        # val_loss = 0.
        # n_batch = 0
        # while not loader.epoch_is_over(test_list, params.BATCH_SIZE):
        #     batch_list = loader.get_batch_list(test_list, params.BATCH_SIZE)
        #     cmp, bg, lab, rfg = loader.simple_batch(batch_list, params.INPUT_SIZE)
        #     inp, lab, rfg = loader.get_batch(batch_list, params.INPUT_SIZE, rd_scale=False, rd_mirror=True)
        #     feed_dict = {in_cmp: cmp, in_bg: bg, gt: lab, raw_fg: rfg, phase: False}
        #     ls = sess.run([loss], feed_dict=feed_dict)
            # test_writer.add_summary(summary, iteration)
            # val_loss += np.mean(ls)
            # n_batch += 1
        # val_loss /= n_batch
        # if prev_val_loss != -1.:
        #     improvement = '{:2f}%'.format((prev_val_loss - val_loss) / prev_val_loss)
        # print('Validation loss: {:.3f}. Improvement: {}'.format(val_loss, improvement))
        # prev_val_loss = val_loss

        print('Saving examples')
        # loads and visualize example prediction of current model
        n_ex = 5
        ex_list = [test_file_list[np.random.randint(0, len(test_file_list))] for _ in range(n_ex)]
        ex_cmp, ex_bg, ex_lab, _ = loader.simple_batch(ex_list, params.INPUT_SIZE)
        feed_dict = {in_cmp: ex_cmp, in_bg: ex_bg, gt: ex_lab, phase: False}
        summary = sess.run([ex_merged], feed_dict)[0]
        ex_writer.add_summary(summary, iteration)
        print('Saving chekpoint...')
        saver.save(sess, os.path.join(params.LOG_DIR, 'weights_{}'.format(t_str), 'model'), global_step=iteration)


def simple_train():
    with tf.variable_scope('input'):
        in_cmp = tf.placeholder('float', [None, params.INPUT_SIZE[0], params.INPUT_SIZE[1], 3], name='composite')
        in_bg = tf.placeholder('float', [None, params.INPUT_SIZE[0], params.INPUT_SIZE[1], 3], name='background')
        diff = tf.subtract(in_cmp, in_bg, name='difference')
        raw_fg = tf.placeholder('float', [None, params.INPUT_SIZE[0], params.INPUT_SIZE[1], 3], name='raw_fg')
        gt = tf.placeholder('float', [None, params.INPUT_SIZE[0], params.INPUT_SIZE[1], 1], name='gt')
        phase = tf.placeholder(tf.bool, name='phase')

    with tf.variable_scope('model'):
        unet = unet_simple.create_model(in_cmp, in_bg, diff, phase)
        pred = unet.output

    train_file_list = loader.get_file_list(params.SYNTHETIC_DATASET, params.TRAINING_LIST)
    test_file_list = loader.get_file_list(params.SYNTHETIC_DATASET, params.TRAINING_LIST)
    with tf.Session() as sess:
        t_str = time.asctime().replace(' ', '_')
        train_writer = tf.summary.FileWriter(os.path.join(params.LOG_DIR, 'train_{}'.format(t_str)), sess.graph)
        ex_writer = tf.summary.FileWriter(os.path.join(params.LOG_DIR, 'examples_{}'.format(t_str)))
        saver = tf.train.Saver()
        simple_procedure(sess, in_cmp, in_bg, gt, raw_fg, phase, pred, train_writer, ex_writer, saver,
                         train_file_list, test_file_list)


def resume_simple(meta_path, weight_folder):
    """ resume training from checkpoint """
    train_file_list = loader.get_file_list(params.SYNTHETIC_DATASET, params.TRAINING_LIST)
    test_file_list = loader.get_file_list(params.SYNTHETIC_DATASET, params.TRAINING_LIST)
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(meta_path)
        new_saver.restore(sess, tf.train.latest_checkpoint(weight_folder))
        graph = tf.get_default_graph()
        in_cmp = graph.get_tensor_by_name('input/composite:0')
        in_bg = graph.get_tensor_by_name('input/background:0')
        phase = graph.get_tensor_by_name('input/phase:0')
        gt = graph.get_tensor_by_name('input/gt:0')
        raw_fg = graph.get_tensor_by_name('input/raw_fg:0')
        pred = graph.get_tensor_by_name('model/simple_unet/probs:0')

        t_str = time.asctime().replace(' ', '_')
        train_writer = tf.summary.FileWriter(os.path.join(params.LOG_DIR, 'train_{}'.format(t_str)), sess.graph)
        ex_writer = tf.summary.FileWriter(os.path.join(params.LOG_DIR, 'examples_{}'.format(t_str)))
        saver = tf.train.Saver()
        simple_procedure(sess, in_cmp, in_bg, gt, raw_fg, phase, pred, train_writer, ex_writer, saver,
                         train_file_list, test_file_list)


def video_procedure(sess, in_cmp, in_bg, in_warped, gt, raw_fg, phase, pred, train_writer, ex_writer, saver,
                    train_file_list, test_file_list):
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'model/simple_unet')
    # print('Training variables:')
    # print(train_vars)
    t_str = time.asctime().replace(' ', '_')
    with tf.variable_scope('loss'):
        alpha_loss = regular_l1(pred, gt, name='alpha_loss')
        pred_cmp = composite(raw_fg, in_bg, pred)
        cmp_loss = regular_l1(pred_cmp, in_cmp, name='compositional_loss')
        s_loss = tf.add(0.5 * alpha_loss, 0.5 * cmp_loss)
        loss = tf.reduce_mean(s_loss, name='loss')
    with tf.variable_scope('resume_training_2'):
        lr = 1e-3
        print('Training with learning rate of {}'.format(lr))
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-08)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step(), var_list=train_vars)
    with tf.variable_scope('summary'):
        summary_loss = tf.summary.scalar('loss', loss)
        summary_alpha_loss = tf.summary.scalar('alpha_loss', tf.reduce_mean(alpha_loss))
        summary_cmp_loss = tf.summary.scalar('compositional_loss', tf.reduce_mean(cmp_loss))
        summary_cmp = tf.summary.image('composite', bgr2rgb(in_cmp))
        summary_gt = tf.summary.image('ground_truth', gt)
        summary_pred = tf.summary.image('prediction', pred)
    train_summaries = [summary_loss, summary_alpha_loss, summary_cmp_loss]
    ex_summaries = [summary_cmp, summary_gt, summary_pred]
    train_merged = tf.summary.merge(train_summaries)
    ex_merged = tf.summary.merge(ex_summaries)
    iteration = 0
    sess.run(tf.global_variables_initializer())
    for epoch in range(params.N_EPOCHS):
        print('Running epoch {} of {} on {} examples'.format(epoch+1, params.N_EPOCHS, len(train_file_list)))
        training_list = train_file_list.copy()
        test_list = test_file_list.copy()
        random.shuffle(training_list)
        random.shuffle(test_list)
        # training
        while not loader.epoch_is_over(training_list, params.BATCH_SIZE):
            print('Training model, epoch {}/{}, iteration {}.'.format(epoch + 1, params.N_EPOCHS, iteration + 1))
            batch_list = loader.get_batch_list(training_list, params.BATCH_SIZE)
            cmp, bg, lab, prev_lab, rfg = loader.video_batch(batch_list, params.INPUT_SIZE)
            feed_dict = {in_cmp: cmp, in_bg: bg, gt: lab, raw_fg: rfg, in_warped: prev_lab, phase: True}
            summary, _ = sess.run([train_merged, train_op], feed_dict=feed_dict)
            train_writer.add_summary(summary, iteration)
            iteration += 1

        print('Saving examples')
        # loads and visualize example prediction of current model
        n_ex = 5
        ex_list = [test_file_list[np.random.randint(0, len(test_file_list))] for _ in range(n_ex)]
        ex_cmp, ex_bg, ex_lab, ex_prevlab, _ = loader.video_batch(ex_list, params.INPUT_SIZE)
        feed_dict = {in_cmp: ex_cmp, in_bg: ex_bg, gt: ex_lab, in_warped: ex_prevlab,  phase: False}
        summary = sess.run([ex_merged], feed_dict)[0]
        ex_writer.add_summary(summary, iteration)
        print('Saving chekpoint...')
        saver.save(sess, os.path.join(params.LOG_DIR, 'weights_{}'.format(t_str), 'model'), global_step=iteration)


def video_train():
    with tf.variable_scope('input'):
        in_cmp = tf.placeholder('float', [None, params.INPUT_SIZE[0], params.INPUT_SIZE[1], 3], name='composite')
        in_bg = tf.placeholder('float', [None, params.INPUT_SIZE[0], params.INPUT_SIZE[1], 3], name='background')
        in_warped = tf.placeholder('float', [None, params.INPUT_SIZE[0], params.INPUT_SIZE[1], 3], name='background')
        raw_fg = tf.placeholder('float', [None, params.INPUT_SIZE[0], params.INPUT_SIZE[1], 3], name='raw_fg')
        gt = tf.placeholder('float', [None, params.INPUT_SIZE[0], params.INPUT_SIZE[1], 1], name='gt')
        phase = tf.placeholder(tf.bool, name='phase')

    with tf.variable_scope('model'):
        unet = unet_simple.create_model(in_cmp, in_bg, in_warped, phase)
        pred = unet.output
    train_list, test_list = loader.video_file_list()
    with tf.Session() as sess:
        t_str = time.asctime().replace(' ', '_')
        train_writer = tf.summary.FileWriter(os.path.join(params.LOG_DIR, 'train_{}'.format(t_str)), sess.graph)
        ex_writer = tf.summary.FileWriter(os.path.join(params.LOG_DIR, 'examples_{}'.format(t_str)))
        saver = tf.train.Saver()
        video_procedure(sess, in_cmp, in_bg, in_warped, gt, raw_fg, phase, pred, train_writer, ex_writer, saver,
                        train_list, test_list)


def test_out_val_simple(meta_path, weight_folder):
    train_file_list = loader.get_file_list(params.SYNTHETIC_DATASET, params.TRAINING_LIST)
    test_file_list = loader.get_file_list(params.SYNTHETIC_DATASET, params.TRAINING_LIST)
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(meta_path)
        new_saver.restore(sess, tf.train.latest_checkpoint(weight_folder))
        graph = tf.get_default_graph()
        for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            print(v)
        # x = graph.get_tensor_by_name('input/input:0')
        # gt = graph.get_tensor_by_name('input/gt:0')
        # raw_fg = graph.get_tensor_by_name('input/raw_fg:0')
        # pred = graph.get_tensor_by_name('model/output:0')
        test_batch = loader.get_batch_list(train_file_list, 1)
        in_cmp = graph.get_tensor_by_name('input/composite:0')
        in_bg = graph.get_tensor_by_name('input/background:0')
        cmp, bg, _, _ = loader.simple_batch(test_batch, params.INPUT_SIZE)
        conv1 = graph.get_tensor_by_name('model/simple_unet/conv1/biases:0')
        val = sess.run(conv1)
        print(val)
        print('MEAN: {} /// STANDARD DEV: {}'.format(np.mean(val), np.std(val)))
        print('Processing output for file {}'.format(test_batch[0]))
        # output = graph.get_tensor_by_name('model/simple_unet/output/BiasAdd:0')
        # softed = graph.get_tensor_by_name('model/simple_unet/output_1:0')
        # vals, probs = sess.run([output, softed], feed_dict={in_cmp: cmp, in_bg: bg})
        # probs = sess.run(softed, feed_dict={in_cmp: cmp, in_bg: bg})
        # print(vals.shape)
        # print(probs.shape)
        # for i in range(320):
        #     for j in range(320):
        #         print('{} -> {}'.format(vals[0, i, j, 0], probs[0, i, j, 0]))


if __name__ == '__main__':
    # train()
    # simple_train()
    # log_fold = 'log/weights_Wed_Jul_18_17:09:13_2018'
    # log_fold = 'log/weights_Fri_Jul_27_12:03:44_2018'
    # resume_simple(os.path.join(log_fold, 'model-70370.meta'), log_fold)
    video_train()
