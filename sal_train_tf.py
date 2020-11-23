import tensorflow as tf
import numpy as np
from functools import reduce

import os
from math import ceil
import glob
import load_data

os.environ["CUDA_VISIBLE_DEVICES"]="0"
is_training = True
VGG_MEAN = [103.939, 116.779, 123.68]
#trainable=True
trainable1 = True
var_dict = {}
data_dict = np.load('vgg16.npy', encoding='latin1').item()
lr = 5*1e-3
offset = tf.constant(0.5)
rgb = tf.placeholder(tf.float32,[None,224,224,3],name='input_image')


#GTc = tf.placeholder(tf.float32,[5,480,640],name='cground_truth')
GTf = tf.placeholder(tf.float32,[5,480,640],name='fground_truth')


rgb_scaled = rgb * 255.0
red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])

def max_pool(bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def mean_pool(bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def conv_layer(bottom, in_channels, out_channels, name,train):
        with tf.variable_scope(name):
            filt, conv_biases = get_conv_var(3, in_channels, out_channels, name,train)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu
def pred_conv_layer(bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt,conv_biases= get_conv_var1(1, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='VALID')
            #bias = tf.nn.bias_add(conv, conv_biases)
            #relu = tf.nn.sigmoid(bias)

            return conv

def get_conv_var(filter_size, in_channels, out_channels, name,train):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = get_var(initial_value, name, 0, name + "_filters",train)

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = get_var(initial_value, name, 1, name + "_biases",train)

        return filters, biases

def get_conv_var1(filter_size, in_channels, out_channels, name):
        initial_value = tf.ones([filter_size, filter_size, in_channels, out_channels])
        filters = get_var1(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = get_var1(initial_value, name, 1, name + "_biases")

        return filters, biases

def get_var1(initial_value, name, idx, var_name):
        print(trainable1)
        if data_dict is not None and name in data_dict:
            value = data_dict[name][idx]
        else:
            value = initial_value

        if trainable1:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

def get_var(initial_value, name, idx, var_name,train):
        print(train)
        if data_dict is not None and name in data_dict:
            value = data_dict[name][idx]
        else:
            value = initial_value

        if train==True:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):

    initial = tf.constant(0.1,shape=shape)

    return tf.Variable(initial)

def conv2d(x,W):
    #stride [1,x_movement,y_movement,1]#
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')


def reduce_var(x,axis=None):
    m = tf.reshape(tf.reduce_mean(x,axis = axis),(-1,1,1))
    devs_squared = tf.square(x-m)
    return tf.reduce_mean(devs_squared,axis = axis)

def reduce_std(x,axis=None):
    return tf.sqrt(reduce_var(x,axis=axis))

def reduce_mean(x,axis=None):
    return tf.reduce_mean(x,axis = axis)


def nss(sal,fix):
    m = tf.reshape(reduce_mean(sal,axis = [1,2]),(-1,1,1))
    std = tf.reshape(reduce_std(sal,axis=[1,2]),(-1,1,1))
    sal = (sal-m)/std
    s = tf.reduce_sum(fix,[1,2])
    ns = sal*fix
    sns = tf.reduce_sum(ns,[1,2])
    nssa = sns/s
    return tf.reduce_mean(nssa)

#vgg architecture
#percept

        
conv1_1 = conv_layer(bgr, 3, 64, "conv1_1",False)
conv1_2 = conv_layer(conv1_1, 64, 64, "conv1_2",False)
pool1 = max_pool(conv1_2, 'pool1')

conv2_1 = conv_layer(pool1, 64, 128, "conv2_1",False)
conv2_2 = conv_layer(conv2_1, 128, 128, "conv2_2",False)
pool2 = max_pool(conv2_2, 'pool2')

conv3_1 = conv_layer(pool2, 128, 256, "conv3_1",False)
conv3_2 = conv_layer(conv3_1, 256, 256, "conv3_2",False)
conv3_3 = conv_layer(conv3_2, 256, 256, "conv3_3",False)

pool3 = max_pool(conv3_3, 'pool3')

conv4_1 = conv_layer(pool3, 256, 512, "conv4_1",False)
conv4_2 = conv_layer(conv4_1, 512, 512, "conv4_2",False)
conv4_3 = conv_layer(conv4_2, 512, 512, "conv4_3",False)

pool4 = max_pool(conv4_3, 'pool4')

conv5_1 = conv_layer(pool4, 512, 512, "conv5_1",False)
conv5_2 = conv_layer(conv5_1, 512, 512, "conv5_2",False)
conv5_3 = conv_layer(conv5_2, 512, 512, "conv5_3",True)
print(conv5_3.shape)


pred = pred_conv_layer(conv5_3, 512, 1, "pred2")


  
#pool5 = max_pool(conv5_3, 'pool5')
pred = tf.squeeze(pred,axis=-1)
ma = tf.reshape(tf.reduce_max(pred,[1,2]),[-1,1,1])
mi = tf.reshape(tf.reduce_min(pred,[1,2]),[-1,1,1])

pred = (pred-mi)/(ma-mi)
pred = tf.reshape(pred,(5,14,14,1))
u_pred = tf.image.resize_images(pred,(480,640))

u_pred = tf.squeeze(u_pred,axis=-1)
#error = tf.square(u_pred-GTc)*tf.exp(GTc)
#loss1 = tf.reduce_mean(tf.reduce_sum(error,[1,2]))
loss = -nss(u_pred,GTf)
#loss = 0.0001*loss1-0.5*loss2

loss_output1 = tf.Variable(0.0)
loss_output2 = tf.Variable(0.0)



tf.summary.scalar('trtloss',loss_output1)
tf.summary.scalar('tetloss',loss_output2)
train_op = tf.train.AdamOptimizer(lr).minimize(loss)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)


sess.run(tf.global_variables_initializer())

train_writer1 = tf.summary.FileWriter("/home/sen/Desktop/sal_vis/model1/log/traint/",sess.graph)

test_writer1 = tf.summary.FileWriter("/home/sen/Desktop/sal_vis/model1/log/testt/",sess.graph)

merge_op = tf.summary.merge_all()
saver = tf.train.Saver(max_to_keep=20)

tfiles = glob.glob('./train_im/'+'*.jpg')
print(len(tfiles))
vfiles = glob.glob('./val_im/'+'*.jpg')
print(len(vfiles))


for i in range(20):
    t_L1 = np.zeros(1)
    shuffle = np.random.permutation(10000)
    for j in range(2000):#1349
        ind = shuffle[j*5+0:j*5+5]
        image,_,label2 = load_data.read_train_data(tfiles,ind)
        #print(np.max(image[:]))
        feed_dict = {rgb: image,GTf:label2}
        #print(feature.shape)#
        L,_ = sess.run([loss,train_op], feed_dict=feed_dict)        
        if j%5==0:
            #print(L)
            tr_Loss1 = sess.run(merge_op, feed_dict={loss_output1: L})
            train_writer1.add_summary(tr_Loss1,i*2000+j)          
    saver.save(sess,'./model1/model/',global_step=i*2000+j)
    lr = lr*0.8
    s1 = np.arange(1000)
    #is_training = False
    for j in range(200):

        ind1 = s1[j*5+0:j*5+5]
        image,_,label2 = load_data.read_val_data(vfiles,ind1)
        feed_dict = {rgb: image,GTf:label2}
    
        L = sess.run(loss, feed_dict=feed_dict)
        t_L1 += L
        if j==199:
            t_L1 = t_L1/200
            te_Loss1 = sess.run(merge_op, feed_dict={loss_output2: t_L1[0]})
            test_writer1.add_summary(te_Loss1,(i+1)*2000)
            #test_writer.add_summary(te_loss2,(i+1)*2000)
            #test_writer.add_summary(10*te_loss3,(i+1)*2000)
