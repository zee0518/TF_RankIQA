#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created rank_loss.py by rjw at 19-1-15 in WHU.
"""

import numpy as np
import pdb
import tensorflow as tf

class Rank_loss():
    """Layer of Efficient Siamese loss function."""

    def __init__(self):
        self.margin = 6
        print('*********************** SETTING UP****************')
        pass

    def get_rankloss(self, p_hat,batch_size):
        """The forward """
        self.Num = 0
        batch = 1
        level = 6
        dis = 4
        SepSize = batch * level
        self.dis = []
        # for the first
        self.loss = 0
        for k in range(dis):
            for i in range(SepSize * k, SepSize * (k + 1) - batch):
                for j in range(SepSize * k + int((i - SepSize * k) / batch + 1) * batch, SepSize * (k + 1)):
                    self.dis.append(p_hat[i] - p_hat[j])
                    self.Num += 1
        self.dis = tf.cast(tf.reshape(self.dis,[-1]),tf.float32)
        # self.loss = np.maximum(0, self.margin - self.dis)  # Efficient Siamese forward pass of hinge loss
        diff = tf.cast(self.margin,tf.float32) - self.dis
        # temp = tf.greater(tf.cast(0,tf.float32),diff)
        # self.loss = tf.reduce_sum(tf.where(temp,tf.cast(0,tf.float32),diff)) # 维度不对
        self.loss = tf.maximum(0.,diff)

        loss = tf.reduce_mean(self.loss)

        return loss #,self.dis,diff

if __name__=="__main__":

    y_hat = tf.placeholder(tf.float32,[24])
    rank_loss = Rank_loss()
    loss,dis,loss_ = rank_loss.get_rankloss(y_hat, 24)

    with tf.Session() as sess:
        import numpy as np
        y = np.random.random([24])
        _loss, _dis, _loss_ = sess.run([loss,dis,loss_],feed_dict={y_hat:y})
        print(_loss)
        print(_dis)
        print( _loss_)


