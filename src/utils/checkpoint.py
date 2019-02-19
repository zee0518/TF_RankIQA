#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created checkpoint.py by rjw at 19-1-8 in WHU.
"""

import os
import tensorflow as tf

def save(saver, sess, logdir, step):
    '''Save weights.

    Args:
      saver: TensorFlow Saver object.
      sess: TensorFlow session.
      logdir: path to the snapshots directory.
      step: current training step.
    '''
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step,write_meta_graph=False)
    print('The checkpoint has been created.')


def load(saver, sess, ckpt_path):
    '''Load trained weights.

    Args:
      saver: TensorFlow Saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    saver.restore(sess, ckpt_path)


    print("Restored model parameters from {}".format(ckpt_path))



def __load__(saver,sess,ckpt_dir):
    import re
    print(" [*] Reading checkpoints...")
    # checkpoint_dir = os.path.join(ckpt_dir, self.model_dir)
    checkpoint_dir = ckpt_dir

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print(" [*] Success to read {}".format(ckpt_name))
        return counter
    else:
        print(" [!] Failed to find a checkpoint")
        return 0