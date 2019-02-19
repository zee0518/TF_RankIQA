#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
created evaluate.py by rjw at 19-1-10 in WHU.
"""
import argparse
import os
import tensorflow as tf
import numpy as np

from src.utils.logger import setup_logger
from src.net.model import VggNetModel
from src.metrics.srocc import evaluate_metric

BASE_PATH = '/media/rjw/Ran-software/dataset/iqa_dataset'


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

# specifying default parameters
def process_command_args():

    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Tensorflow RankIQA evaluating")

    ## Path related arguments
    parser.add_argument('--exp_name',type=str,default="finetuneiqa",help='experiment name')
    parser.add_argument('--data_dir',type=str,default=BASE_PATH,help='the root path of dataset')
    parser.add_argument('--test_list', type=str, default='tid2013_test.txt', help='data list for read image.')
    parser.add_argument('--ckpt_dir',type=str,default=os.path.abspath('..')+'/experiments/',help='the path of ckpt file')

    ## dataset related arguments
    parser.add_argument('--dataset', default='LIVE', type=str, choices=["LIVE", "CSIQ", "tid2013", "CLIVE"],
                        help='datset choice')
    parser.add_argument('--crop_width',type=int,default=224,help='train patch width')
    parser.add_argument('--crop_height',type=int,default=224,help='train patch height')


    args = parser.parse_args()
    return args



def get_image_list(args):
    test_image_paths = []
    test_scores = []
    f = open(args.test_list, 'r')
    for line in f:
        image_path, image_score = line.strip("\n").split()
        test_image_paths.append(image_path)
        test_scores.append(image_score)
    f.close()
    test_image_paths = np.array(test_image_paths)
    test_scores = np.array(test_scores, dtype='float32')

    return test_image_paths, test_scores


def parse_test_data(filename, score):
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
    image = tf.image.resize_images(image, (224, 224))
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    image = tf.expand_dims(image, dim=0)

    return image, score

def evaluate(args):

    graph=tf.Graph()

    with graph.as_default() as g:

        images = tf.placeholder(tf.float32, [None, 224, 224, 3])
        model = VggNetModel(num_classes=1)
        y_hat = model.inference(images, False)
        y_hat = tf.reshape(y_hat, [-1, ])
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:

        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(args.ckpt_dir) # load up level directory :+'/iqa_model_final.ckpt'
        if ckpt and ckpt.model_checkpoint_path:
            logger.info('loading checkpoint:' + ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            logger.info("please loading checkpoint!")

        test_image_paths, test_scores = get_image_list(args)
        score_set = []
        label_set = []
        for i in range(len(test_image_paths)):
            image_tensor, score = parse_test_data(str(test_image_paths[i]),float(test_scores[i]))
            image = sess.run(image_tensor)

            predict_score = sess.run(y_hat, feed_dict={images: image})
            label_set.append(score)
            score_set.append(predict_score[0])
            if i % 50 ==0:
                logger.info("image:{}/{}, true score:{}".format(i,len(test_image_paths),score))
                logger.info("image:{}/{}, predict_score:{}".format(i,len(test_image_paths),predict_score[0]))

        srocc, krocc, plcc, rmse, mse = evaluate_metric(label_set, score_set)

        logger.info("SROCC_v: %.3f\t KROCC: %.3f\t PLCC_v: %.3f\t RMSE_v: %.3f\t mse: %.3f\n" % (srocc, krocc, plcc, rmse, mse))
        logger.info("Test finish!")



def main():
    args = process_command_args()

    if args.dataset=='tid2013':
        args.test_list=os.path.abspath('..')+'/data/ft_tid2013_test.txt'
    elif args.dataset=='LIVE':
        args.test_list = os.path.abspath('..')+'/data/ft_live_test.txt'
    elif args.dataset=='CSIQ':
        args.test_list = 'ft_csiq_test.txt'
    else:
        logger.info("datasets is not in LIVE, CSIQ, tid2013")

    output_dir = os.path.join(args.ckpt_dir, args.dataset)
    args.data_dir=os.path.join(args.data_dir,args.dataset)
    args.ckpt_dir=os.path.join(args.ckpt_dir,args.dataset,args.exp_name)


    global logger
    logger = setup_logger("TF_IQA_"+args.dataset+"_evaluating", output_dir,"evaluate_")
    logger.info(args)

    evaluate(args)

if __name__=="__main__":
    main()