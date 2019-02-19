#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
# from torchvision import datasets, models, transforms


def fully_connection(input, hiden_units, keep_prob):
    """ fully connection architecture.

    Args:
        :param input: tensor - (batch_size, ?)
        :param hiden_units: int.
        :param keep_prob: tensor. scalar.

    Returns:
        :return: tensor - (batch_size, 1)
    """
    shape = input.get_shape().as_list()
    if len(shape) == 4:
        size = shape[-1] * shape[-2] * shape[-3]
    else:
        size = shape[1]
    input = tf.reshape(input, [-1, size])
    input_units = input.get_shape()[1].value
    weights_initializer = tf.truncated_normal_initializer(stddev=1.0 / np.sqrt(float(input_units)))
    with tf.variable_scope('block1'):
        weights = tf.get_variable(name='weights',
                                  shape=[input_units, hiden_units],
                                  initializer=weights_initializer, dtype=tf.float32)
        biases = tf.get_variable(name='biases', shape=[hiden_units],
                                 initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        hidden1 = tf.nn.relu(tf.matmul(input, weights) + biases)
        hidden1_dropout = tf.nn.dropout(hidden1, keep_prob)

    with tf.variable_scope('block2'):
        weights = tf.get_variable(name='weights',
                                  shape=[hiden_units, 10],
                                  initializer=weights_initializer, dtype=tf.float32)
        biases = tf.get_variable(name='biases', shape=[10],
                                 initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        output = tf.matmul(hidden1_dropout, weights) + biases

    return output

class vgg16:

    def __init__(self,imgs):
        self.parameters=[]
        self.imgs=imgs
        self.conv_layers()
        self.fc_layers()
        self.probs=tf.nn.softmax(self.fc8)

    def maxpool(self,name,input,trainable):
        out=tf.nn.max_pool(input,[1,2,2,1],[1,2,2,1],padding="SAME",name=name)
        return out

    def conv(self,name,input,out_channel,trainable):
        in_channel=input.get_shape()[-1]
        with tf.variable_scope(name):
            kernel=tf.get_variable("weights",[3,3,in_channel,out_channel],dtype=tf.float32,trainable=False)
            biases=tf.get_variable("biases",[out_channel],dtype=tf.float32,trainable=False)
            conv_res=tf.nn.conv2d(input,kernel,[1,1,1,1],padding="SAME")
            res=tf.nn.bias_add(conv_res,biases)
            out=tf.nn.relu(res,name=name)
        self.parameters+=[kernel,biases]
        return out

    def fc(self,name,input,out_channel,trainable=True):
        shape=input.get_shape().as_list()
        if len(shape)==4:
            size=shape[-1]*shape[-2]*shape[-3]
        else:
            size=shape[1]
        input_data_flat=tf.reshape(input,[-1,size])
        with tf.variable_scope(name):
            weights=tf.get_variable(name="weights",shape=[size,out_channel],dtype=tf.float32,trainable=trainable)
            biases=tf.get_variable(name="biases",shape=[out_channel],dtype=tf.float32,trainable=trainable)
            res=tf.matmul(input_data_flat,weights)
            out=tf.nn.relu(tf.nn.bias_add(res,biases))
        self.parameters+=[weights,biases]
        return out

    def conv_layers(self):
        # zero-mean input:self.imgs
        self.conv1_1=self.conv("conv1_1",self.imgs,64,trainable=False)
        self.conv1_2=self.conv("conv1_2",self.conv1_1,64,trainable=False)
        self.pool1=self.maxpool("pool1",self.conv1_2,trainable=False)

        self.conv2_1=self.conv("conv2_1",self.pool1,128,trainable=False)
        self.conv2_2=self.conv("conv2_2",self.conv2_1,128,trainable=False)
        self.pool2=self.maxpool("pool2",self.conv2_2,trainable=False)

        self.conv3_1=self.conv("conv3_1",self.pool2,256,trainable=False)
        self.conv3_2=self.conv("conv3_2",self.conv3_1,256,trainable=False)
        self.conv3_3=self.conv("conv3_3",self.conv3_2,256,trainable=False)
        self.pool3=self.maxpool("pool3",self.conv3_3,trainable=256)

        self.conv4_1=self.conv("conv4_1",self.pool3,512,trainable=False)
        self.conv4_2=self.conv("conv4_2",self.conv4_1,512,trainable=False)
        self.conv4_3=self.conv("conv4_3",self.conv4_2,512,trainable=False)
        self.pool4=self.maxpool("pool4",self.conv4_3,trainable=False)

        self.conv5_1=self.conv("conv5_1",self.pool4,512,trainable=False)
        self.conv5_2=self.conv("conv5_2",self.conv5_1,512,trainable=False)
        self.conv5_3=self.conv("conv5_3",self.conv5_2,512,trainable=False)
        self.pool5=self.maxpool("pool5",self.conv5_3,trainable=False)

    def fc_layers(self):
        self.fc6=self.fc("fc6",self.pool5,4096,trainable=False)
        self.fc7=self.fc("fc7",self.fc6,4096,trainable=True)
        self.dropout=tf.nn.dropout(self.fc7,1.0)
        self.fc8=self.fc("fc8",self.dropout,10)

    def load_weights(self,weight_files,sess):
        weights=np.load(weight_files)
        keys=sorted(weights.keys())
        for i,k in enumerate(keys):
            if i not in [30,31]:
                sess.run(self.parameters[i].assign(weights[k]))
        print("------------load pretrian moedle done-----------")


if __name__=="__main__":

    with tf.Graph().as_default() ,tf.Session() as sess:

        images = np.random.random([4, 224, 224, 3])
        img = tf.cast(images, tf.float32)
        vgg = vgg16(img)
        scores_hat = vgg.probs
        print(vgg)
        print(scores_hat)

        sess.run(tf.global_variables_initializer())
        scores_hat=sess.run(scores_hat)
        print(scores_hat)
        # print(scores_hat.graph)

        variables_names = [v.name for v in tf.trainable_variables()]
        values = sess.run(variables_names)
        for k,v in zip(variables_names,values):
            print("Variable:",k)
            print("Shape:",v.shape)


