#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created clive_dataloader.py by rjw at 19-1-20 in WHU.
"""

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
        self.is_training = self.param_str['is_training']
        self.data_dir = '/media/rjw/Ran-software/dataset/iqa_dataset/CLIVE'

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
        self._shuffle_roidb_inds()

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        # TODO(rbg): remove duplicated code
        if self._cur + self.batch_size >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + self.batch_size]
        self._cur += self.batch_size
        return db_inds

    def get_minibatch(self, minibatch_db):
        """Given a roidb, construct a minibatch sampled from it."""
        if self.is_training:
            jobs = self.workers.map(parse_data, minibatch_db)
        else:
            jobs = self.workers.map(parse_data_without_augmentation, minibatch_db)
        # print len(jobs)
        index = 0
        images_train = np.zeros([self.batch_size,224, 224, 3], np.float32)
        # pdb.set_trace()
        for index_job in range(len(jobs)):
            images_train[index, :, :, :] = jobs[index_job]
            index += 1

        blobs = {'data': images_train}
        return blobs

    def next_batch(self):
        """Get blobs and copy them into this layer's top blob vector."""

        db_inds = self._get_next_minibatch_inds()
        minibatch_db = []
        for i in range(len(db_inds)):
            minibatch_db.append(os.path.join(self.data_dir,self._roidb[int(db_inds[i])]))
        # minibatch_db = [self._roidb[i] for i in db_inds]
        # print minibatch_db
        scores = []
        for i in range(len(db_inds)):
            scores.append(self.scores[int(db_inds[i])])

        blobs = self.get_minibatch(minibatch_db)
        blobs['label'] = np.asarray(scores)

        return blobs['data'], blobs['label']

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

# clive datasets include bmp and jpg format images
def preprocess(data):
    im = np.asarray(cv2.imread(data))
    x = im.shape[0]
    y = im.shape[1]
    x_p = np.random.randint(x - 224, size=1)[0]
    y_p = np.random.randint(y - 224, size=1)[0]
    # print x_p,y_p
    images = im[x_p:x_p + 224, y_p:y_p + 224, :]
    # print images.shape
    return images


scale_size = 256
crop_size = 224
def parse_data(data):
    img = np.array(cv2.imread(data))
    h,w,c = img.shape
    assert c==3
    img = cv2.resize(img,(scale_size,scale_size))
    img = img.astype(np.float32)

    shift = (scale_size - crop_size) // 2
    img = img[shift: shift + crop_size, shift: shift + crop_size, :]
    # Flip image at random if flag is selected
    if np.random.random() < 0.5: #self.horizontal_flip and
        img = cv2.flip(img, 1)
    img = (img - np.array(127.5) )/127.5

    return img

def parse_data_without_augmentation(data):

    img = np.array(cv2.imread(data))
    h,w,c = img.shape
    assert c==3
    img = cv2.resize(img,(crop_size,crop_size))
    img = img.astype(np.float32)
    img = (img - np.array(127.5) )/127.5
    return img


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
    param_str={'root_dir':root_dir,'data_root':'data','split':'clive_train','im_shape':[224,224],'batch_size':16,'is_training':True}
    clive_data = Dataset(param_str)

    for i in range(10):
        image_batch, label_batch = clive_data.next_batch()
        # for blob_name, blob in blobs.items():
        #     print(blob_name, blob.shape)
        print(image_batch.shape,label_batch.shape)

    # from PIL import Image
    # image = Image.open(image_path).convert("RGB")