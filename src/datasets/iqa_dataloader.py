#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created dataloader.py by rjw at 19-1-15 in WHU.
"""

import tensorflow as tf

IMAGE_SIZE = 224


def parse_data(filename, scores):
    '''
    Loads the image file, and randomly applies crops and flips to each image.
    Args:
        filename: the filename from the record
        scores: the scores from the record
    Returns:
        an image referred to by the filename and its scores
    '''
    image = tf.read_file(filename)
    image = tf.image.decode_bmp(image, channels=3)  # decode_jpeg
    image = tf.image.resize_images(image, (256, 256))
    image = tf.random_crop(image, size=(IMAGE_SIZE, IMAGE_SIZE, 3))
    image = tf.image.random_flip_left_right(image)
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    return image, scores


def parse_data_without_augmentation(filename, scores):
    '''
    Loads the image file without any augmentation. Used for validation set.
    Args:
        filename: the filename from the record
        scores: the scores from the record
    Returns:
        an image referred to by the filename and its scores
    '''
    image = tf.read_file(filename)
    image = tf.image.decode_bmp(image, channels=3)
    image = tf.image.resize_images(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    return image, scores


def train_generator(train_image_paths, train_scores, batchsize=32, shuffle=True):
    '''
    Creates a python generator that loads the AVA dataset images with random data
    augmentation and generates numpy arrays to feed into the Keras model for training.
    Args:
        batchsize: batchsize for training
        shuffle: whether to shuffle the dataset
    Returns:
        a batch of samples (X_images, y_scores)
    '''
    with tf.Session() as sess:
        # create a dataset
        train_dataset = tf.data.Dataset().from_tensor_slices((train_image_paths, train_scores))
        train_dataset = train_dataset.map(parse_data, num_parallel_calls=2)

        train_dataset = train_dataset.batch(batchsize)
        train_dataset = train_dataset.repeat()
        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=4)
        train_iterator = train_dataset.make_initializable_iterator()

        train_batch = train_iterator.get_next()

        sess.run(train_iterator.initializer)

        while True:
            try:
                X_batch, y_batch = sess.run(train_batch)
                yield (X_batch, y_batch)
            except:
                train_iterator = train_dataset.make_initializable_iterator()
                sess.run(train_iterator.initializer)
                train_batch = train_iterator.get_next()

                X_batch, y_batch = sess.run(train_batch)
                yield (X_batch, y_batch)


def val_generator(val_image_paths, val_scores, batchsize):
    '''
    Creates a python generator that loads the AVA dataset images without random data
    augmentation and generates numpy arrays to feed into the Keras model for training.
    Args:
        batchsize: batchsize for validation set
    Returns:
        a batch of samples (X_images, y_scores)
    '''
    with tf.Session() as sess:
        val_dataset = tf.data.Dataset().from_tensor_slices((val_image_paths, val_scores))
        val_dataset = val_dataset.map(parse_data_without_augmentation)

        val_dataset = val_dataset.batch(batchsize)
        val_dataset = val_dataset.repeat()
        val_iterator = val_dataset.make_initializable_iterator()

        val_batch = val_iterator.get_next()

        sess.run(val_iterator.initializer)

        while True:
            try:
                X_batch, y_batch = sess.run(val_batch)
                yield (X_batch, y_batch)
            except:
                val_iterator = val_dataset.make_initializable_iterator()
                sess.run(val_iterator.initializer)
                val_batch = val_iterator.get_next()

                X_batch, y_batch = sess.run(val_batch)
                yield (X_batch, y_batch)


root_dir="/home/rjw/desktop/graduation_project/TF_RankIQA"

import os
if __name__=="__main__":

    live_train_path=os.path.join(root_dir,"data/ft_live_train.txt")
    lvie_test_path=os.path.join(root_dir,"data/ft_live_test.txt")

    train_image_paths = []
    train_scores = []
    f = open(live_train_path,'r')
    for line in f:
        image_path, image_score = line.strip("\n").split()
        train_image_paths.append(image_path)
        train_scores.append(image_score)
    f.close()

    print(type(train_generator(train_image_paths,train_scores))) # <class 'generator'>

    train_gen = train_generator(train_image_paths,train_scores)

    # print(next(train_gen))

    for i in range(3):
        batch_train_images,batch_train_scores = next(train_gen)
        print(batch_train_images.shape,batch_train_scores.shape)


    for iter,(images,targets) in enumerate(train_gen):
        print(iter,images.shape,targets.shape)
        if iter==3:
            break

