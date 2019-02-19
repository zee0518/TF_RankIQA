#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf

# code from: https://github.com/master/nima/blob/master/nima.py
def tril_indices(n, k=0):
    """Return the indices for the lower-triangle of an (n, m) array.
    Works similarly to `np.tril_indices`
    Args:
      n: the row dimension of the arrays for which the returned indices will
        be valid.
      k: optional diagonal offset (see `np.tril` for details).
    Returns:
      inds: The indices for the triangle. The returned tuple contains two arrays,
        each with the indices along one dimension of the array.
    """
    m1 = tf.tile(tf.expand_dims(tf.range(n), axis=0), [n, 1])
    m2 = tf.tile(tf.expand_dims(tf.range(n), axis=1), [1, n])
    mask = (m1 - m2) >= -k
    ix1 = tf.boolean_mask(m2, tf.transpose(mask))
    ix2 = tf.boolean_mask(m1, tf.transpose(mask))
    return ix1, ix2


def ecdf(p):
    """Estimate the cumulative distribution function.
    The e.c.d.f. (empirical cumulative distribution function) F_n is a step
    function with jump 1/n at each observation (possibly with multiple jumps
    at one place if there are ties).
    For observations x= (x_1, x_2, ... x_n), F_n is the fraction of
    observations less or equal to t, i.e.,
    F_n(t) = #{x_i <= t} / n = 1/n \sum^{N}_{i=1} Indicator(x_i <= t).
    Args:
      p: a 2-D `Tensor` of observations of shape [batch_size, num_classes].
        Classes are assumed to be ordered.
    Returns:
      A 2-D `Tensor` of estimated ECDFs.
    """
    n = p.get_shape().as_list()[1]
    indices = tril_indices(n)
    indices = tf.transpose(tf.stack([indices[1], indices[0]]))
    ones = tf.ones([n * (n + 1) / 2])
    triang = tf.scatter_nd(indices, ones, [n, n])
    return tf.matmul(p, triang)


def emd_loss(p, p_hat, r=2, scope=None):
    """Compute the Earth Mover's Distance loss.
    Hou, Le, Chen-Ping Yu, and Dimitris Samaras. "Squared Earth Mover's
    Distance-based Loss for Training Deep Neural Networks." arXiv preprint
    arXiv:1611.05916 (2016).
    Args:
      p: a 2-D `Tensor` of the ground truth probability mass functions.
      p_hat: a 2-D `Tensor` of the estimated p.m.f.-s
      r: a constant for the r-norm.
      scope: optional name scope.
    `p` and `p_hat` are assumed to have equal mass as \sum^{N}_{i=1} p_i =
    \sum^{N}_{i=1} p_hat_i
    Returns:
      A 0-D `Tensor` of r-normed EMD loss.
    """
    with tf.name_scope(scope, 'EmdLoss', [p, p_hat]):
        ecdf_p = ecdf(p)
        ecdf_p_hat = ecdf(p_hat)
        emd = tf.reduce_mean(tf.pow(tf.abs(ecdf_p - ecdf_p_hat), r), axis=-1)
        emd = tf.pow(emd, 1 / r)
    return tf.reduce_mean(emd)


# https://github.com/mixuala/nima/blob/master/nima_utils.py
def _cum_CDF(x):
    # e.g. cdf([1,1,1,1]) ==  [ 0.25,  0.5 ,  0.75,  1.  ]
    x = tf.to_float(x)
    cs = tf.cumsum(x, axis=1, reverse=False)
    total = cs[:, -1:]  # last column == cumulative sum
    cdf = tf.divide(cs, total)
    return cdf


def _emd(y, y_hat, reduce_mean=True, r=2):
    """Returns the earth mover distance between to arrays of ratings,
    based on cumulative distribution function

    Args:
      y, y_hat: a mini-batch of ratings, each composed of a count of scores
                shape = (None, n), array of count of scores for score from 1..n
      reduce_mean: apply tf.reduce_mean()
      r: r=2 for rmse loss (default) or r=1 for absolute val
    Returns:
      float
    """
    m, n = tf.convert_to_tensor(y).get_shape().as_list()
    cdf_loss = tf.subtract(_cum_CDF(y), _cum_CDF(y_hat))
    emd_loss = tf.pow(tf.reduce_mean(tf.pow(cdf_loss, r), axis=1), 1 / r)
    if reduce_mean:
        emd_loss = tf.reduce_mean(emd_loss)
        return emd_loss
    else:
        return tf.reshape(emd_loss, [m, 1])


# pytorch: https://github.com/kentsyx/Neural-IMage-Assessment/blob/master/model.py
def single_emd_loss(p, q, r=2):
    """
    Earth Mover's Distance of one sample
    Args:
        p: true distribution of shape num_classes × 1
        q: estimated distribution of shape num_classes × 1
        r: norm parameter
    """
    assert p.shape == q.shape, "Length of the two distribution must be the same"
    length = p.shape[0]
    emd_loss = 0.0
    for i in range(1, length + 1):
        emd_loss += sum(tf.abs(p[:i] - q[:i])) ** r
    return (emd_loss / length) ** (1. / r)


def batch_emd_loss(p, q, r=2):
    """
    Earth Mover's Distance on a batch
    Args:
        p: true distribution of shape mini_batch_size × num_classes × 1
        q: estimated distribution of shape mini_batch_size × num_classes × 1
        r: norm parameters
    """
    assert p.shape == q.shape, "Shape of the two distribution batches must be the same."
    mini_batch_size = p.shape[0]
    loss_vector = []
    for i in range(mini_batch_size):
        loss_vector.append(single_emd_loss(p[i], q[i], r=r))
    return sum(loss_vector) / mini_batch_size

if __name__=="__main__":
    p1 = tf.constant([[0. ,    0.  ,   0.9968 ,0.0032, 0.  ,   0. ,    0.   ,  0.   ,  0.,     0.    ]])
    p2 = tf.constant([[2.4484623e-06 ,1.8617798e-02, 9.2069888e-01 ,6.0438950e-02, 2.4148858e-04,  3.5216067e-07 ,1.5788361e-08 ,4.3034593e-08 ,4.1905771e-08 ,5.9919600e-08]])
    p3 = tf.constant([[0.1, 0.2, 0.3, 0.5], [0.3, 0.3, 0.3, 0.1]])

    p4 = []
    f1 = emd_loss(p1,p2)
    f2 = _emd(p1,p2)

    f3 = ecdf(p1)
    f4 = _cum_CDF(p1)
    with tf.Session() as sess:
        out1=sess.run(f1)
        print(out1)
        out2=sess.run(f2)
        print(out2)

        out3=sess.run(f3)
        print(out3)
        out4=sess.run(f4)
        print(out4)

