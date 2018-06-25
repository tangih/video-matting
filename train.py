import tensorflow as tf
import numpy as np
import cv2
import random
import os

import loader
import unet
import params


def composite(fg, bg, alpha):
    tri_alpha = tf.concat(values=[alpha, alpha, alpha], axis=3)
    cmp = tf.add(tf.multiply(tri_alpha, fg), tf.multiply(1. - tri_alpha, bg))
    return cmp


def regular_l1(output, gt, name):
    """ regularized L1 loss """
    sq_loss = tf.square(tf.subtract(output, gt))
    eps2 = tf.square(tf.constant(1e-6))
    loss = tf.sqrt(sq_loss + eps2, name=name)
    return loss


def train():
    model = unet.UNet()
    with tf.variable_scope('input'):
        x = tf.placeholder('float', [None, params.INPUT_SIZE[0], params.INPUT_SIZE[1], 7])
        raw_fg = tf.placeholder('float', [None, params.INPUT_SIZE[0], params.INPUT_SIZE[1], 3])
        gt = tf.placeholder('float', [None, params.INPUT_SIZE[0], params.INPUT_SIZE[1], 1])
        in_cmp, in_bg, in_trimap = tf.split(value=x, num_or_size_splits=[3, 3, 1], axis=3)
        # ex_in = tf.placeholder('float', [5, params.INPUT_SIZE[0], params.INPUT_SIZE[1], 7])
        # ex_lab = tf.placeholder('float', [5, params.INPUT_SIZE[0], params.INPUT_SIZE[1]])
        # ex_cmp, ex_bg, ex_trimap = tf.split(value=x, num_or_size_splits=[3, 3, 1], axis=3)

    with tf.variable_scope('model'):
        model.build(x)
        pred = model.output

    # loss functions
    with tf.variable_scope('loss'):
        alpha_loss = regular_l1(pred, gt, name='alpha_loss')
        pred_cmp = composite(raw_fg, in_bg, pred)
        cmp_loss = regular_l1(pred_cmp, in_cmp, name='compositional_loss')
        s_loss = tf.add(0.5 * alpha_loss, 0.5 * cmp_loss)
        loss = tf.reduce_mean(s_loss, name='loss')

    with tf.variable_scope('training'):
        # lr = tf.placeholder('float', 1, name='learning_rate')
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    with tf.variable_scope('summary'):
        summary_loss = tf.summary.scalar('loss', loss)
        summary_alpha_loss = tf.summary.scalar('alpha_loss', alpha_loss)
        summary_cmp_loss = tf.summary.scalar('compositional_loss', cmp_loss)
        summary_cmp = tf.summary.image('composite', in_cmp)
        summary_gt = tf.summary.image('ground_truth', gt)
        summary_pred = tf.summary.image('prediction', pred)

    train_summaries = [summary_loss, summary_alpha_loss, summary_cmp_loss]
    test_summaries = [summary_loss, summary_alpha_loss, summary_cmp_loss]
    ex_summaries = [summary_cmp, summary_gt, summary_pred]
    train_merged = tf.summary.merge(train_summaries)
    test_merged = tf.summary.merge(test_summaries)
    ex_merged = tf.summary.merge(ex_summaries)

    train_file_list = loader.get_file_list(params.SYNTHETIC_DATASET, params.TRAINING_LIST)
    test_file_list = loader.get_file_list(params.SYNTHETIC_DATASET, params.TRAINING_LIST)
    prev_val_loss = -1.

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(os.path.join(params.LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(params.LOG_DIR, 'test'))
        ex_writer = tf.summary.FileWriter(os.path.join(params.LOG_DIR, 'examples'))
        saver = tf.train.Saver()
        iteration = 0
        for epoch in range(params.N_EPOCHS):
            training_list = train_file_list
            test_list = test_file_list
            random.shuffle(training_list)
            random.shuffle(test_list)
            # training
            while not loader.epoch_is_over(training_list, params.BATCH_SIZE):
                print('Training model, epoch {}/{}, iteration {}.'.format(epoch+1, params.N_EPOCHS, iteration+1))
                batch_list = loader.get_batch_list(training_list, params.BATCH_SIZE)
                inp, lab, rfg = loader.get_batch(batch_list, params.INPUT_SIZE, rd_scale=False, rd_mirror=True)
                feed_dict = {x: inp, gt: lab, raw_fg: rfg}
                summary, _ = sess.run([train_merged, train_op], feed_dict=feed_dict)
                train_writer.add_summary(train_merged, iteration)
                iteration += 1
            # validation
            print('Training completed. Computing validation loss...')
            val_loss = 0.
            n_batch = 0
            while not loader.epoch_is_over(test_list, params.BATCH_SIZE):
                batch_list = loader.get_batch_list(training_list, params.BATCH_SIZE)
                inp, lab, rfg = loader.get_batch(batch_list, params.INPUT_SIZE, rd_scale=False, rd_mirror=True)
                feed_dict = {x: inp, gt: lab, raw_fg: rfg}
                summary, loss = sess.run([test_merged, loss], feed_dict=feed_dict)
                val_loss += loss
                n_batch += 1
            test_writer.add_summary(test_merged, iteration)
            val_loss /= n_batch
            improvement = 'NaN'
            if prev_val_loss != -1.:
                improvement = '{:2f}%'.format((prev_val_loss-val_loss)/prev_val_loss)
            print('Validation loss: {.3f}. Improvement: {}'.format(val_loss, improvement))
            print('Saving examples')
            # loads and visualize example prediction of current model
            n_ex = 5
            ex_list = [test_file_list[np.random.randint(0, len(test_file_list))] for _ in range(n_ex)]
            ex_inp, ex_lab, _ = loader.get_batch(ex_list, params.INPUT_SIZE, rd_scale=False, rd_mirror=True)
            feed_dict = {x: ex_inp, gt: ex_lab}
            summary = sess.run([ex_merged], feed_dict)
            ex_writer.add_summary(summary, iteration)
            print('Saving chekpoint...')
            saver.save(sess, os.path.join(params.LOG_DIR, 'model'), global_step=iteration)


if __name__ == '__main__':
    train()
