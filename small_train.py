import tensorflow as tf
import numpy as np
import small as unet
import loader
import time
import random
import params
import os


def bgr2rgb(bgr):
    blue, green, red = tf.split(value=bgr, num_or_size_splits=3, axis=-1)
    image = tf.concat([red, green, blue], axis=-1)
    return image


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


def small_training(sess, in_cmp, in_bg, gt, raw_fg, phase, pred, train_writer, ex_writer, saver,
                   train_file_list, test_file_list, lr):
    t_str = time.asctime().replace(' ', '_')
    month = t_str.split('_')[1]
    date = int(t_str.split('_')[3])
    with tf.variable_scope('loss'):
        alpha_loss = regular_l1(pred, gt, name='alpha_loss')
        pred_cmp = composite(raw_fg, in_bg, pred)
        cmp_loss = regular_l1(pred_cmp, in_cmp, name='compositional_loss')
        s_loss = tf.add(0.5 * alpha_loss, 0.5 * cmp_loss)
        loss = tf.reduce_mean(s_loss, name='loss')
    with tf.variable_scope('training_{}_{}'.format(month, date)):
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
    ex_summaries = [summary_cmp, summary_gt, summary_pred]
    train_merged = tf.summary.merge(train_summaries)
    ex_merged = tf.summary.merge(ex_summaries)
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

            if (iteration+1) % 1000 == 0:
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


def train(learning_rate=1e-5):
    with tf.variable_scope('input'):
        in_cmp = tf.placeholder('float', [None, params.INPUT_SIZE[0], params.INPUT_SIZE[1], 3], name='composite')
        in_bg = tf.placeholder('float', [None, params.INPUT_SIZE[0], params.INPUT_SIZE[1], 3], name='background')
        input = tf.concat([in_cmp, in_bg], axis=-1)
        raw_fg = tf.placeholder('float', [None, params.INPUT_SIZE[0], params.INPUT_SIZE[1], 3], name='raw_fg')
        gt = tf.placeholder('float', [None, params.INPUT_SIZE[0], params.INPUT_SIZE[1], 1], name='gt')
        phase = tf.placeholder(tf.bool, name='phase')

    with tf.variable_scope('model'):
        model = unet.UNetSmall(input, phase)
        pred = model.output

    train_file_list = loader.get_file_list(params.SYNTHETIC_DATASET, params.TRAINING_LIST)
    test_file_list = loader.get_file_list(params.SYNTHETIC_DATASET, params.TEST_LIST)
    with tf.Session() as sess:
        t_str = time.asctime().replace(' ', '_')
        train_writer = tf.summary.FileWriter(os.path.join(params.LOG_DIR, 'train_{}'.format(t_str)), sess.graph)
        ex_writer = tf.summary.FileWriter(os.path.join(params.LOG_DIR, 'examples_{}'.format(t_str)))
        saver = tf.train.Saver()
        small_training(sess, in_cmp, in_bg, gt, raw_fg, phase, pred, train_writer, ex_writer, saver,
                       train_file_list, test_file_list, learning_rate)


if __name__ == '__main__':
    train()
