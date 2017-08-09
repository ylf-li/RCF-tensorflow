
import glob
import os
import sys
import random
import shutil
import cv2
import numpy as np
from utils import *
from model import *
from VGG16 import vgg16

import tensorflow as tf
import tensorflow.contrib as tc
import tensorlayer as tl
from tensorlayer.layers import *
from scipy.misc import imread, imresize,imsave
import matplotlib.pyplot as plt


batch_size=6
epochs=50

learning_rate=5e-6

weight_decay=0.0002
crop_size=400

Img_path='data/BSD500/images/'
GT_path='data/BSD500/gts/'

# Img_path='data/PASCAL/images/'
# GT_path='data/PASCAL/gts/'

summaries_dir='summary'
shutil.rmtree('results_fuse')
os.mkdir('results_fuse')
shutil.rmtree('results_res5')
os.mkdir('results_res5')
shutil.rmtree('results_res1')
os.mkdir('results_res1')
shutil.rmtree('summary')
os.mkdir('summary')

GT_list=glob.glob(os.path.join(GT_path,'*.png'))
global_steps = tf.Variable(0, trainable=False)

input_img=tf.placeholder(tf.float32,[None,crop_size,crop_size,3],name='input_image')
GT = tf.placeholder(tf.float32,[None,crop_size,crop_size],name='GT')

pos_mask = tf.placeholder(tf.float32,[None,crop_size,crop_size],name='Pen_pos')
neg_mask = tf.placeholder(tf.float32,[None,crop_size,crop_size],name='Pen_neg')
pos_weight = tf.placeholder(tf.float32,[None,],name='pos_w')
neg_weight = tf.placeholder(tf.float32,[None,],name='neg_w')

print('loadind model VGG')
weights = np.load('vgg16_weights.npz')
keys = sorted(weights.keys())
vgg=vgg16(input_img)

pred=tf.squeeze(vgg.pred.outputs)
Contour5=tf.squeeze(vgg.deconv5.outputs)
Contour4=tf.squeeze(vgg.deconv4.outputs)
Contour3=tf.squeeze(vgg.deconv3.outputs)
Contour2=tf.squeeze(vgg.deconv2.outputs)
Contour1=tf.squeeze(vgg.deconv1.outputs)


dis_G,logit_dis_G = discriminator_simplified_api(tf.concat(3,[tf.sigmoid(vgg.pred.outputs),input_img]),\
                                is_train=True,reuse=False)
dis_GT,logit_dis_GT = discriminator_simplified_api(tf.concat(3,[tf.expand_dims(GT,-1),input_img]),\
                                is_train=True,reuse=True)

vgg_vars = tl.layers.get_variables_with_name('enconde',True,True)


gen_vars5 = tl.layers.get_variables_with_name('deconde/deconv5',True,True)
regularizer5=tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(weight_decay),gen_vars5+vgg_vars)

gen_vars4 = tl.layers.get_variables_with_name('deconde/deconv4',True,True)
regularizer4=tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(weight_decay),gen_vars4+vgg_vars)

gen_vars3 = tl.layers.get_variables_with_name('deconde/deconv3',True,True)
regularizer3=tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(weight_decay),gen_vars3+vgg_vars)

gen_vars2 = tl.layers.get_variables_with_name('deconde.deconv2',True,True)
regularizer2=tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(weight_decay),gen_vars2+vgg_vars)

gen_vars1 = tl.layers.get_variables_with_name('deconde/deconv1',True,True)
regularizer1=tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(weight_decay),gen_vars1+vgg_vars)

gen_vars6 = tl.layers.get_variables_with_name('deconde', True, True)
regularizer6=tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(weight_decay),gen_vars6+vgg_vars)



# Context loss
loss_context=tf.nn.weighted_cross_entropy_with_logits(Contour1,GT,10)
loss_context1 = tf.reduce_mean(loss_context)
tf.summary.scalar('loss_context1', loss_context1)

loss_context=tf.nn.weighted_cross_entropy_with_logits(Contour2,GT,10)
loss_context2 = tf.reduce_mean(loss_context)
tf.summary.scalar('loss_context2', loss_context2)

loss_context=tf.nn.weighted_cross_entropy_with_logits(Contour3,GT,10)
loss_context3 = tf.reduce_mean(loss_context)
tf.summary.scalar('loss_context3', loss_context3)

loss_context=tf.nn.weighted_cross_entropy_with_logits(Contour4,GT,10)
loss_context4 = tf.reduce_mean(loss_context)
tf.summary.scalar('loss_context4', loss_context4)

loss_context=tf.nn.weighted_cross_entropy_with_logits(Contour5,GT,10)
loss_context5 = tf.reduce_mean(loss_context)
tf.summary.scalar('loss_context5', loss_context5)

loss_context=tf.nn.weighted_cross_entropy_with_logits(pred,GT,10)
loss_context6 = tf.reduce_mean(loss_context)
tf.summary.scalar('loss_context6', loss_context6)


learning_rate = tf.train.exponential_decay(learning_rate, global_steps,60000, 0.1, staircase=True)
loss_optim1 = tf.train.AdamOptimizer(learning_rate).minimize(loss_context1, var_list=gen_vars1+vgg_vars,global_step=global_steps)
loss_optim2 = tf.train.AdamOptimizer(learning_rate).minimize(loss_context2, var_list=gen_vars2+vgg_vars,global_step=global_steps)
loss_optim3 = tf.train.AdamOptimizer(learning_rate).minimize(loss_context3, var_list=gen_vars3+vgg_vars,global_step=global_steps)
loss_optim4 = tf.train.AdamOptimizer(learning_rate).minimize(loss_context4, var_list=gen_vars4+vgg_vars,global_step=global_steps)
loss_optim5 = tf.train.AdamOptimizer(learning_rate).minimize(loss_context5, var_list=gen_vars5+vgg_vars,global_step=global_steps)
loss_optim6 = tf.train.AdamOptimizer(learning_rate).minimize(loss_context6, var_list=gen_vars6+vgg_vars,global_step=global_steps)


num_batch=int(len(GT_list)/batch_size)
merged = tf.summary.merge_all()

iter_step=0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    train_writer = tf.summary.FileWriter('summary', sess.graph) 
    for i,k in enumerate(keys[:25]):
         sess.run(vgg.parameters[i].assign(weights[k]))
    all_names=[]
    for epoch in xrange(epochs):
        np.random.shuffle(GT_list)
        index_begin=0
        for index,batch_indx in enumerate(xrange(num_batch)):
            resall_res5=[]
            resall_fuse=[]
            resall_res1=[]
            GT_batch=GT_list[index_begin:index_begin+batch_size]
            index_begin=index_begin+batch_size

            img_names=[ os.path.basename(x).split('.png')[0] for x in GT_batch]
            gts_path =[os.path.join(GT_path,x+'.png')  for x in img_names ]
            img_path = [os.path.join(Img_path,x.replace('gt','data')+'.jpg')  for x in img_names ]

            imgs_raw=[cv2.imread(x) for x in img_path]
            gts_raw=[cv2.imread(x,0)/255.0 for x in gts_path]

            imgs=np.array(imgs_raw)
            gts=np.array(gts_raw)

            gts[gts>0.5]=1.0

            lr,g,loss1,loss2,loss3,loss4,loss5,loss6,res1,res2,res3,summary,_ ,_,_,_,_,_= sess.run([learning_rate,global_steps,\
              loss_context1,loss_context2,loss_context3,loss_context4,loss_context5,loss_context6,\
              Contour5,Contour1,pred,merged,loss_optim1,loss_optim2,loss_optim3,loss_optim4,loss_optim5,loss_optim6],\
                    feed_dict={GT:gts,vgg.imgs:imgs})
            resall_res5.extend(res1)
            resall_fuse.extend(res2)
            resall_res1.extend(res3)
            train_writer.add_summary(summary,iter_step)
            save=[imsave(os.path.join('results_res5','{}.png'.format(index)),\
                       1./(1+np.exp(-resall_res5[i]))) for i in np.arange(0,len(resall_res5))]
            save=[imsave(os.path.join('results_fuse','{}.png'.format(index)),\
                       1./(1+np.exp(-resall_fuse[i]))) for i in np.arange(0,len(resall_fuse))]
            save=[imsave(os.path.join('results_res1','{}.png'.format(index)),\
                       1./(1+np.exp(-resall_res1[i]))) for i in np.arange(0,len(resall_res1))]
            if((index+1)%500==0):
                saver.save(sess,"/opt/code/contour/models/PASCAL/models_BSDS500_HED_weighted.ckpt")
            print("Epoch: [%2d/%2d] [%4d/%4d] lr: %.8f gs: %d l1: %.6f,l2: %.6f,l3: %.6f,l4: %.6f,l5: %.6f,l6: %.6f," \
                %(epoch, epochs, index, num_batch, lr,g,loss1,loss2,loss3,loss4,loss5,loss6))