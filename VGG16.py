import numpy as np
import tensorflow as tf
import tensorlayer as tl
from utils import *
from tensorlayer.layers import *

class vgg16:
    def __init__(self, imgs):
        self.imgs = imgs
        self.convlayers()
        self.deconde()

    def convlayers(self):
        self.parameters = []

        with tf.name_scope('enconde') as scope:
            # zero-mean input
            with tf.name_scope('preprocess') as scope:
                mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
                images = self.imgs-mean

            # conv1_1
            with tf.name_scope('conv1_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,stddev=1e-1),name='weights')
                conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv1_1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv1_2
            with tf.name_scope('conv1_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,stddev=1e-1),name='weights')
                conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv1_2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # pool1

            self.pool1 = tf.nn.max_pool(self.conv1_2,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool1')

            # conv2_1
            with tf.name_scope('conv2_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,stddev=1e-1),name='weights')
                conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv2_1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv2_2
            with tf.name_scope('conv2_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv2_2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # pool2

            self.pool2 = tf.nn.max_pool(self.conv2_2,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME',
                                   name='pool2')

            # conv3_1
            with tf.name_scope('conv3_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,stddev=1e-1),name='weights')
                conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv3_1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv3_2
            with tf.name_scope('conv3_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,stddev=1e-1),name='weights')
                conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv3_2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv3_3
            with tf.name_scope('conv3_3') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,stddev=1e-1),name='weights')
                conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv3_3 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # pool3

            self.pool3 = tf.nn.max_pool(self.conv3_3,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME',
                                   name='pool3')

            # conv4_1
            with tf.name_scope('conv4_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,stddev=1e-1), name='weights')
                conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv4_1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv4_2
            with tf.name_scope('conv4_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,stddev=1e-1),name='weights')
                conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv4_2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv4_3
            with tf.name_scope('conv4_3') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,stddev=1e-1),name='weights')
                conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv4_3 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # pool4
            self.pool4 = tf.nn.max_pool(self.conv4_3,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 1,1, 1],
                                   padding='SAME',
                                   name='pool4')

            # conv5_1
            with tf.name_scope('conv5_1') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,stddev=1e-1), name='weights')
                conv = tf.nn.atrous_conv2d(self.pool4, kernel,2, padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv5_1 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv5_2
            with tf.name_scope('conv5_2') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,stddev=1e-1),name='weights')
                conv = tf.nn.atrous_conv2d(self.conv5_1, kernel, 2, padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv5_2 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

            # conv5_3
            with tf.name_scope('conv5_3') as scope:
                kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,stddev=1e-1), name='weights')
                conv = tf.nn.atrous_conv2d(self.conv5_2, kernel, 2, padding='SAME')
                biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),name='biases')
                out = tf.nn.bias_add(conv, biases)
                self.conv5_3 = tf.nn.relu(out, name=scope)
                self.parameters += [kernel, biases]

    def deconde(self):

        with tf.variable_scope("deconde"):
            image_size = 400
            s2, s4, s8, s16,s32 = int(image_size/2), int(image_size/4), int(image_size/8), int(image_size/16),int(image_size/32)
            batch_size = tf.shape(self.conv5_3)[0]
            w_init = tf.contrib.layers.xavier_initializer_conv2d()
            
            #28
            with tf.name_scope('deconv5') as scope:
                conv5_1 = InputLayer(self.conv5_1,name='g/h0/conv5_1')
                conv5_1 = Conv2d(conv5_1, 32, (1,1), (1, 1),padding='SAME', W_init=w_init, name='g/h1/conv51')
                conv5_2 = InputLayer(self.conv5_2,name='g/h0/conv5_2')
                conv5_2 = Conv2d(conv5_2, 32, (1,1), (1, 1),padding='SAME', W_init=w_init, name='g/h1/conv52')
                conv5_3 = InputLayer(self.conv5_3,name='g/h0/conv5_3')
                conv5_3 = Conv2d(conv5_3, 32, (1,1), (1, 1),padding='SAME', W_init=w_init, name='g/h1/conv53')
                concat=tl.layers.ElementwiseLayer([conv5_1,conv5_2,conv5_3],combine_fn=tf.add,name='g/h1/concat')
                concat=Conv2d(concat, 1, (1,1), (1, 1),padding='SAME', W_init=w_init, name='g/h1/conv')
                bilinear_init = deconv2d_bilinear_upsampling_initializer(shape=[16,16,1,1])
                self.deconv1=tl.layers.DeConv2dLayer(concat,shape = [16,16,1,1],output_shape = [batch_size,image_size,image_size,1],
                            strides=[1,8,8, 1],W_init=bilinear_init,padding='SAME',act=tf.identity, name='g/h1/decon2d')

            #28
            with tf.name_scope('deconv4') as scope:
                conv4_1 = InputLayer(self.conv4_1,name='g/h2/conv4_1')
                conv4_1 = Conv2d(conv4_1, 32, (1,1), (1, 1),padding='SAME', W_init=w_init, name='g/h2/conv41')
                conv4_2 = InputLayer(self.conv4_2,name='g/h2/conv4_2')
                conv4_2 = Conv2d(conv4_2, 32, (1,1), (1, 1),padding='SAME', W_init=w_init, name='g/h2/conv42')
                conv4_3 = InputLayer(self.conv4_3,name='g/h2/conv4_3')
                conv4_3 = Conv2d(conv4_3, 32, (1,1), (1, 1),padding='SAME', W_init=w_init, name='g/h2/conv43')
                concat=tl.layers.ElementwiseLayer([conv4_1,conv4_2,conv4_3],combine_fn=tf.add,name='g/h2/concat')
                concat=Conv2d(concat, 1, (1,1), (1, 1), act=None, padding='SAME', W_init=w_init, name='g/h2/conv')
                bilinear_init = deconv2d_bilinear_upsampling_initializer(shape=[16,16,1,1])
                self.deconv2=tl.layers.DeConv2dLayer(concat,shape = [16,16,1,1],output_shape = [batch_size,image_size,image_size,1],
                            strides=[1,8,8, 1],W_init=bilinear_init,padding='SAME',act=tf.identity, name='g/h2/decon2d')

            #56
            with tf.name_scope('deconv3') as scope:
                conv3_1 = InputLayer(self.conv3_1,name='g/h3/conv3_1')
                conv3_1 = Conv2d(conv3_1, 32, (1,1), (1, 1),padding='SAME', W_init=w_init, name='g/h3/conv31')
                conv3_2 = InputLayer(self.conv3_2,name='g/h3/conv3_2')
                conv3_2 = Conv2d(conv3_2, 32, (1,1), (1, 1),padding='SAME', W_init=w_init, name='g/h3/conv32')
                conv3_3 = InputLayer(self.conv3_3,name='g/h3/conv3_3')
                conv3_3 = Conv2d(conv3_3, 32, (1,1), (1, 1),padding='SAME', W_init=w_init, name='g/h3/conv33')
                concat=tl.layers.ElementwiseLayer([conv3_1,conv3_2,conv3_3],combine_fn=tf.add,name='g/h3/concat')
                concat=Conv2d(concat, 1, (1,1), (1, 1),padding='SAME', W_init=w_init, name='g/h3/conv311')
                bilinear_init = deconv2d_bilinear_upsampling_initializer(shape=[8,8,1,1])
                self.deconv3=tl.layers.DeConv2dLayer(concat,shape = [8,8,1,1],output_shape = [batch_size,image_size,image_size,1],
                            strides=[1, 4,4, 1],W_init=bilinear_init,padding='SAME',act=tf.identity, name='g/h3/decon2d')

            #112
            with tf.name_scope('deconv2') as scope:
                conv2_1 = InputLayer(self.conv2_1,name='g/h4/conv2_1')
                conv2_1 = Conv2d(conv2_1, 32, (1,1), (1, 1),padding='SAME', W_init=w_init, name='g/h4/conv22')
                conv2_2 = InputLayer(self.conv2_2,name='g/h4/conv2_2')
                conv2_2 = Conv2d(conv2_2, 32, (1,1), (1, 1),padding='SAME', W_init=w_init, name='g/h4/conv21')
                concat=tl.layers.ElementwiseLayer([conv2_1,conv2_2],combine_fn=tf.add,name='g/h4/concat')
                concat=Conv2d(concat, 1, (1,1), (1, 1),padding='SAME', W_init=w_init, name='g/h4/conv2')
                bilinear_init = deconv2d_bilinear_upsampling_initializer(shape=[4,4,1,1])
                self.deconv4=tl.layers.DeConv2dLayer(concat,shape = [4,4,1,1],output_shape = [batch_size,image_size,image_size,1],
                            strides=[1, 2,2, 1],W_init=bilinear_init,padding='SAME',act=tf.identity, name='g/h4/decon2d')

            #224
            with tf.name_scope('deconv1') as scope:
                conv1_1 = InputLayer(self.conv1_1,name='g/h5/conv1_1')
                conv1_1 = Conv2d(conv1_1, 32,(1,1), (1, 1),padding='SAME', W_init=w_init, name='g/h5/conv12')
                conv1_2 = InputLayer(self.conv1_2,name='g/h5/conv1_2')
                conv1_2 = Conv2d(conv1_2, 32, (1,1), (1, 1),padding='SAME', W_init=w_init, name='g/h5/conv11')
                concat=tl.layers.ElementwiseLayer([conv1_1,conv1_2],combine_fn=tf.add,name='g/h5/concat')
                self.deconv5=Conv2d(concat, 1, (1,1), (1, 1),padding='SAME', W_init=w_init, name='g/h5/conv1')
 
            self.deconv5.outputs.set_shape([None,image_size,image_size,1])
            self.deconv4.outputs.set_shape([None,image_size,image_size,1]) 
            self.deconv3.outputs.set_shape([None,image_size,image_size,1]) 
            self.deconv2.outputs.set_shape([None,image_size,image_size,1]) 
            self.deconv1.outputs.set_shape([None,image_size,image_size,1]) 
            self.fuse=ConcatLayer([self.deconv5,self.deconv4,self.deconv3,self.deconv2,self.deconv1],concat_dim=3,name='g/fuse')
            self.pred=Conv2d(self.fuse, 1, (1,1), (1, 1),padding='SAME', W_init=w_init, name='g/pre/conv')


