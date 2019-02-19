#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created rank_dataloader.py by rjw at 19-1-15 in WHU.
"""

# coding: utf-8
import cv2
import sys
import numpy as np
import multiprocessing as mtp
import pdb
import os.path as osp


class Dataset():

    def __init__(self, param_str):

        # === Read input parameters ===
        self.workers = mtp.Pool(10)
        # params is a python dictionary with layer parameters.
        self.param_str = param_str
        # Check the paramameters for validity.
        check_params(self.param_str)

        # store input as class variables
        self.batch_size = self.param_str['batch_size']
        self.root_dir = self.param_str['root_dir']
        self.data_root = self.param_str['data_root']
        self.im_shape = self.param_str['im_shape']

        # get list of image indexes.
        list_file = self.param_str['split'] + '.txt'
        filename = [line.rstrip('\n') for line in open(osp.join(self.root_dir,self.data_root, list_file))]
        self._roidb = []
        self.scores = []
        for i in filename:
            self._roidb.append(i.split()[0])
            self.scores.append(float(i.split()[1]))

        self._perm = None
        self._cur = 0
        self.num = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        db_inds = []
        dis = 4  # total number of distortions in live dataset
        batch = 1  # number of images for each distortion level
        level = 6  # distortion levels for each   mini_batch = level * dis_mini*batch
        # shuff = np.random.permutation(range(dis))
        Num = len(self.scores) / dis / level
        for k in range(dis):
            for i in range(level):
                temp = self.num
                for j in range(batch):
                    db_inds.append(len(self.scores) / dis * k + i * Num + temp)
                    temp = temp + 1
        self.num = self.num + batch
        if Num - self.num < batch:
            self.num = 0
        db_inds = np.asarray(db_inds)
        return db_inds

    def get_minibatch(self, minibatch_db):
        """Given a roidb, construct a minibatch sampled from it."""
        # Get the input image blob, formatted for caffe

        jobs = self.workers.map(preprocess, minibatch_db)
        # print len(jobs) #48
        index = 0
        images_train = np.zeros([self.batch_size, 224, 224 ,3], np.float32)
        # pdb.set_trace()
        for index_job in range(len(jobs)):
            images_train[index] = jobs[index_job]
            index += 1

        blobs = {'data': images_train}
        return blobs

    def next_batch(self):
        """Get blobs and copy them into this layer's top blob vector."""

        db_inds = self._get_next_minibatch_inds()
        # print(db_inds)
        minibatch_db = []
        for i in range(len(db_inds)):
            minibatch_db.append(self._roidb[int(db_inds[i])])
        # minibatch_db = [self._roidb[i] for i in db_inds]
        # print minibatch_db
        scores = []
        for i in range(len(db_inds)):
            scores.append(self.scores[int(db_inds[i])])
        # print(len(minibatch_db),len(db_inds),len(scores))

        blobs = self.get_minibatch(minibatch_db)
        blobs['label'] = np.asarray(scores)

        return blobs['data'],blobs['label']

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

# 使用opencv读取图像，直接返回numpy.ndarray 对象，通道顺序为BGR ，注意是BGR，通道值默认范围0-255;
# caffe: [B,C,W,H]
# tensorflow模型转caffe模型时，遇到了几个坑其中之一就是caffe的padding方式和tensorflow的padding方式有很大的区别，导致每一层的输出都无法对齐;
# 卷积层的通道顺序：在caffe里是[N,C,H,W]，而tensorflow是[H,W,C,N]
# fc层的通道顺序：在caffe 里是[c_in,c_out]，而tensorflow是[c_out,c_in]
def preprocess(data):
    sp = 224
    im = np.asarray(cv2.imread(data))
    x = im.shape[0]
    y = im.shape[1]
    x_p = np.random.randint(x - sp, size=1)[0]
    y_p = np.random.randint(y - sp, size=1)[0]
    # print x_p,y_p
    images = im[x_p:x_p + sp, y_p:y_p + sp, :] # caffe: .transpose([2, 0, 1])
    # print images.shape
    return images


def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """
    assert 'split' in params.keys(
    ), 'Params must include split (train, val, or test).'

    required = ['batch_size', 'data_root', 'im_shape']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)


root_dir="/home/rjw/desktop/graduation_project/TF_RankIQA"
import os

if __name__=="__main__":
    param_str={'root_dir':root_dir,'data_root':'data','split':'live_train','im_shape':[224,224],'batch_size':24}
    rank_data = Dataset(param_str)

    for i in range(10):
        image_batch, label_batch = rank_data.next_batch()
        # for blob_name, blob in blobs.items():
        #     print(blob_name, blob.shape)
        print(image_batch.shape,label_batch.shape)

    # from PIL import Image
    # image = Image.open(image_path).convert("RGB")
