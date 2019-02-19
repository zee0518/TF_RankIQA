#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf

def mes(y_,y):
    mse = tf.reduce_sum(tf.square(y_ -  y))
    return mse

def reg_l2(p,p_hat):
    # tf.nn.l2_loss(a-b) = sum((a-b)**2) / 2
    # p_hat=tf.reshape(-1,1)
    loss=tf.nn.l2_loss(p-p_hat)
    return loss


