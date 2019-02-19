#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created predict.py by rjw at 19-1-11 in WHU.
"""

import os
import tensorflow as tf
from src.net.model import VggNetModel
import matplotlib.pyplot as plt

model_checkpoint_path="../experiments/tid2013/finetuneiqa/model.ckpt-8999"

images = tf.placeholder(tf.float32,[None,224,224,3])
scores = tf.placeholder(tf.float32, [None])
dropout_keep_prob = tf.placeholder(tf.float32)

model = VggNetModel(num_classes=1, dropout_keep_prob=dropout_keep_prob)
y_hat = model.inference(images, False)
y_hat = tf.reshape(y_hat, [-1, ])
saver = tf.train.Saver()

filename = "/home/rjw/desktop/graduation_project/RankIQA/data/tid2013/tid2013/distorted_images/i01_15_2.bmp" #"img/i15_10_5.bmp"
# image_raw_data = tf.gfile.FastGFile(filename, 'r').read()

with tf.Session() as sess:
    saver.restore(sess, model_checkpoint_path)

    image_raw_data = tf.read_file(filename)
    image_raw = tf.image.decode_bmp(image_raw_data, channels=3)
    image = tf.image.resize_images(image_raw, (224, 224))
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    img3 = tf.expand_dims(image, dim=0)

    input = sess.run(img3)
    # input = img3.eval()
    scores_p = sess.run(y_hat,feed_dict={images:input})

    print(scores_p)
    plt.figure(filename)
    plt.imshow(image_raw.eval())
    plt.title("score is :{}".format(scores_p))
    plt.axis("off")
    plt.show()


