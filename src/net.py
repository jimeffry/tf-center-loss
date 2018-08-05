# auther : lxy
# time : 2017.12.15 /09:56
#project:
# tool: python2
#version: 0.1
#modify:
#name: center loss
#citations: https://github.com/ydwen/caffe-face
#############################
import numpy as np
import tensorflow as tf
import tflearn
import tensorflow.contrib.slim as slim
import tensorflow.contrib as tf_cb

class face_net(object):
    def __init__(self,input_images,num_class,state,wdecay=0.0005):
        self.num_classes = num_class
        self.input_images = input_images
        self.train = True if state == 'train' else False
        self.state = state
        self.w_regularizer = tf_cb.layers.l2_regularizer(wdecay)
        self.w_initializer = tf_cb.layers.xavier_initializer()
        #self.fc2 = self.drop if state == 'train' else self.fc1_r
        #self.conv_bn(self,in_data,in_filter,out_filter,name,std_num,ker_size)
        #self.basic_block(self,in_data,in_planes,out_planes,name,std_num,expand)
        #self.res_block(self,in_data,in_planes,out_planes,name) 
    def parametric_relu(self,_x):
        '''
        alphas = tf.get_variable(name='alphase', _x.get_shape()[-1],\
                       initializer=tf.constant_initializer(0.0),\
                       trainable=True,\
                        dtype=tf.float32)
        '''
        alphas=0.2
        #pos = tf.nn.relu(_x)
        pos = tf.nn.relu6(_x)
        #neg = alphas * (_x - abs(_x)) * 0.5
        return pos

    def conv_bn(self,data_in,in_filter,out_filter,name,std_num=1,kernel_size=3):
        with tf.name_scope(name) as scope:
            conv = tf_cb.layers.conv2d(data_in,out_filter,kernel_size,stride=std_num,\
                                activation_fn=None,weights_regularizer=self.w_regularizer,\
                                weights_initializer=self.w_initializer,scope=scope+'conv')
            act =  self.parametric_relu(conv)
            '''
            kernel = tf.Variable(tf.truncated_normal(shape=[ker_size,ker_size,in_filter,out_filter],\
                     dtype=tf.float32,stddev=0.005),trainable=self.train,name='weights')
            conv = tf.nn.conv2d(in_data,kernel,strides=[1,std_num,std_num,1],padding='SAME',\
                     data_format='NHWC',name='conv')
            biase = tf.Variable(tf.constant(0.0,shape=[out_filter],dtype=tf.float32),\
                      trainable=self.train,name='biase')
            bias = tf.nn.bias_add(conv,biase)
            bn = tf.contrib.layers.batch_norm(inputs=bias,\
                                            decay=0.95,\
                                            center=True,
                                            scale=True,
                                            is_training=self.train,
                                            updates_collections=None,
                                            scope=(name+'batch_norm'))
                                            '''
        return act

    def basic_block(self,in_data,in_planes,out_planes,name,std_num=1,expand=False):
        with tf.name_scope(name+'branch2a') as scope:
            conv = self.conv_bn(in_data,in_planes,out_planes,scope,std_num)
            #conv1_out = tflearn.prelu(conv,name='conv1_out')
            conv1_out = self.parametric_relu(conv)
            #tf.summary.histogram(scope, conv1_out)
            #conv1_out = tf.contrib.keras.layers.LeakyReLU(conv,0.2)
        with tf.name_scope(name+'branch2b') as scope:
            conv2_out = self.conv_bn(conv1_out,out_planes,out_planes,scope)
            #tf.summary.histogram(scope, conv2_out)
        with tf.name_scope(name+'branch1') as scope:
            if expand:
                conv3_out = self.conv_bn(in_data,in_planes,out_planes,scope,std_num,1)
            else :
                conv3_out = in_data
        res = tf.add(conv2_out, conv3_out)
        #prelu = tflearn.prelu(res,weights_init=0.1,trainable=self.train,name='prelu')
        #prelu = tflearn.prelu(res,name='prelu')
        prelu = self.parametric_relu(res)
        #tf.summary.histogram(name+'resout', prelu)
        #prelu = tf.contrib.keras.layers.LeakyReLU(res,0.2)
        print("name",name,"out",prelu)
        #self.prelu_out = prelu
        return prelu

    def res_block(self,in_data,in_planes,out_planes,std_num,name):
        res1 = self.basic_block(in_data,in_planes,out_planes,name+'_1',std_num,True)
        res2 = self.basic_block(res1,out_planes,out_planes,name+'_2')
        return res2

    def get_resnet18(self):
        '''
        the resnet18 architect is below:
        https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        the graph about caffe arcitect below:
        https://dgschwend.github.io/netscope/#/preset/resnet-50

        conv1 = self.conv_bn(self.input_images,3,64,'conv1',2,7)
        #prelu1 = tflearn.prelu(conv1,name='prelu1')
        prelu1 = self.parametric_relu(conv1)
        pool1 = tf.nn.max_pool(prelu1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',\
                    data_format='NHWC',name='pool1')
        '''
        conv1 = self.conv_bn(self.input_images,3,64,'conv1',1,7)
        prelu1 = self.parametric_relu(conv1)
        res1 = self.res_block(prelu1, 64, 64,2,'res1')
        #self.res1 = res1
        res2 = self.res_block(res1, 64, 128,2,'res2')
        res3 = self.res_block(res2, 128, 256,2,'res3')
        res4 = self.res_block(res3, 256, 512,2,'res4')
        pool2 = tf.nn.avg_pool(res4,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID',\
                    data_format='NHWC',name='pool2')
        print("pool2",pool2)
        flatten = tf.contrib.layers.flatten(pool2,scope='flatten')
        feature = tf.contrib.layers.fully_connected(flatten,num_outputs=512,\
                    activation_fn=None,\
                    weights_initializer=self.w_initializer,\
                    trainable=self.train,scope='fc1')
        self.fc1_r = self.parametric_relu(feature)
        #tf.summary.histogram('Fc1', fc1_r)
        print("fc1",self.fc1_r)
        #fc1_r = tf.contrib.keras.layers.LeakyReLU(feature,0.2)
        self.drop = tf.layers.dropout(self.fc1_r,0.5)
        self.fc2 = self.drop if self.state == 'train' else self.fc1_r
        prob = tf.contrib.layers.fully_connected(self.fc2,num_outputs=self.num_classes,\
                    activation_fn=None,\
                    weights_initializer=self.w_initializer,\
                    trainable=self.train,scope='fc2')
        #tf.summary.histogram('Fc2_x', x)
        print("fc2",prob)
        return prob,feature
