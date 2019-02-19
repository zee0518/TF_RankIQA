#!/usr/bin/env python
# -*- coding:utf-8 -*-

from scipy import stats
import numpy as np
import tensorflow as tf

def evaluate_metric(scores, pre_scores):

    scores = np.reshape(np.asarray(scores), (-1,))
    pre_scores = np.reshape(np.asarray(pre_scores), (-1,))
    print("label_set.shape:{},pre_score_set.shape:{}".format(scores.shape, pre_scores.shape))
    srocc = stats.spearmanr(scores, pre_scores, axis=0)[0]
    krocc = stats.stats.kendalltau(scores, pre_scores)[0]
    plcc = stats.pearsonr(scores, pre_scores)[0]

    rmse = np.sqrt(((scores -  pre_scores)**2).mean())
    mse = ((scores - pre_scores)**2).mean()

    # print("SROCC_v: %.3f\t KROCC: %.3f\t PLCC_v: %.3f\t RMSE_v: %.3f\t mse: %.3f\n"% (srocc, krocc, plcc, rmse, mse))
    return srocc,krocc,plcc,rmse,mse


# https://github.com/master/nima/blob/master/nima.py
def scores_stats(scores):
    """Compute score statistics.
    Args:
      scores: a tensor of shape [batch_size, 10].
    Returns:
      A tuple of 1-D `mean` and `std` `Tensors` with shapes [batch_size].
    """
    values = tf.to_float(tf.range(0, 10))
    values = tf.expand_dims(values, axis=0)
    mean = tf.reduce_sum(values * scores, axis=-1)
    var = tf.reduce_sum(tf.square(values) * scores, axis=-1) - tf.square(mean)
    std = tf.sqrt(var)
    return mean


# https://github.com/tfriedel/neural-image-assessment/blob/master/utils/score_utils.py
def mean_score(scores):
    """ calculate mean score for AVA dataset
    :param scores:
    :return: row wise mean score if scores contains multiple rows, else
             a single mean score
    """
    si = np.arange(0, 10, 1).reshape(1,10)
    mean = np.sum(scores * si, axis=1)
    if mean.shape==(1,):
        mean = mean[0]
    return mean


def std_score(scores):
    """ calculate standard deviation of scores for AVA dataset
    :param scores:
    :return: row wise standard deviations if scores contains multiple rows,
             else a single standard deviation
    """
    si = np.arange(0, 10, 1).reshape(1,10)
    mean = mean_score(scores).reshape(-1,1)
    std = np.sqrt(np.sum(((si - mean) ** 2) * scores, axis=1))
    if std.shape==(1,):
        std = std[0]
    return std

if __name__=="__main__":
    scores = tf.constant([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0],
                          [0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 0.1, 0.2, 0.3, 0]])
    scores = scores / tf.reduce_sum(scores, axis=-1, keepdims=True)
    with tf.Session() as sess:
        print(scores)
        scores,means = sess.run([scores, scores_stats(scores)])

        print(scores,means)

