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
from read_tfrecord_v2 import read_single_tfrecord
from net import *
from Center_loss_custom import *
from mnist import mnist_data
import argparse


CENTER_LOSS_ALPHA = 0.9

def argument():
    parser = argparse.ArgumentParser(description="face resnet center loss")
    parser.add_argument('--batch_size',type=int,default=16,help='the batch_size num')
    parser.add_argument('--epoch_num',type=int,default=10,\
                        help='the epoch num should bigger than 10000')
    parser.add_argument('--save_model_name',type=str,default='./face_model/model.ckpt',\
                        help='model Parameters saved name and directory')
    parser.add_argument('--lr',type=float,default=0.001,help='the Learning rate begin')
    parser.add_argument('--sta',type=str,default='train',help="input should 'train' or 'test' ")
    parser.add_argument('--img_shape',type=int,default='300',help="the input image reshape size")
    args = parser.parse_args()
    return args

def build_network(input_images, labels):
    num_class = 526
    sta = 'train'
    ratio = 0.003
    
    net = face_net(input_images,num_class,sta)
    #logits, features = net.inference()
    logits, features = net.get_resnet18()
    #res1 = net.res1

    assert num_class== net.num_classes,"net class should be equal to loss"

    with tf.name_scope('loss'):
        with tf.name_scope('center_loss'):
            center_loss, centers, centers_update_op = get_center_loss(features,logits, labels, CENTER_LOSS_ALPHA, num_class)
        with tf.name_scope('softmax_loss'):
            #softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
            labels_onehot = tf.one_hot(labels,on_value=1,off_value=0,depth=num_class)
            entropy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_onehot, logits=logits)
            print("entropy_loss ",entropy_loss.shape)
            softmax_loss = tf.reduce_mean(entropy_loss)
        with tf.name_scope('total_loss'):
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            total_loss = softmax_loss + ratio * center_loss+0.01 * sum(regularization_losses)
            #total_loss = softmax_loss
    with tf.name_scope('acc'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.arg_max(logits, 1),tf.int32), labels), tf.float32))
    with tf.name_scope('pred_class'):
        pred_class = tf.arg_max(logits, 1)
    with tf.name_scope('loss/'):
        tf.summary.scalar('CenterLoss', center_loss)
        tf.summary.scalar('SoftmaxLoss', softmax_loss)
        tf.summary.scalar('TotalLoss', total_loss)
    #return total_loss, accuracy, centers_update_op, center_loss, softmax_loss,pred_class
    return total_loss

def make_parallel(model,num_gpus,**kwargs):
    in_splits = {}
    for k,v in kwargs.items():
        in_splits[k] = tf.split(v,num_gpus)
    out_splits = []
    for i in range(num_gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU",device_index=i)):
            with tf.variable_scope(tf.get_variable_scope(),reuse=i>0):
                out_splits.append(model(**{k:v[i] for k,v in in_splits.items()}))
    return tf.stack(out_splits,axis=0)


def main():
    LAMBDA = 0.001
    num_class = 526
    args = argument()
    checkpoint_dir = args.save_model_name
    lr = args.lr
    batch_size = args.batch_size
    epoch_num = args.epoch_num
    sta = args.sta
    img_shape = args.img_shape
    num_gpus = 4
    #train_batch_loader = BatchLoader("./data/facescrub_train.list", batch_size,img_shape)
    #test_batch_loader = BatchLoader("./data/facescrub_val.list", batch_size,img_shape)
    #(Height,Width) = (train_batch_loader.height,train_batch_loader.width)
    #train_batch_loader = mnist_data(batch_size)
    tfrecord_file = './data/MegaFace_train.tfrecord_shuffle'
    val_file = './data/MegaFace_val.tfrecord_shuffle'
    image_batch, label_batch = read_single_tfrecord(tfrecord_file, batch_size, img_shape)
    val_image_batch, val_label_batch = read_single_tfrecord(val_file, batch_size, img_shape)
    print("img shape",img_shape)
    with tf.name_scope('input'):
        input_images = tf.placeholder(tf.float32, shape=(batch_size,img_shape,img_shape,3), name='input_images')
        labels = tf.placeholder(tf.int32, shape=(batch_size), name='labels')
        learn_rate = tf.placeholder(tf.float32,shape=(None),name='learn_rate')
    with tf.name_scope('var'):
        global_step = tf.Variable(0, trainable=False, name='global_step')
    #total_loss, accuracy, centers_update_op, center_loss, softmax_loss,pred_class = build_network(input_images,labels)
    #total_loss, accuracy, centers_update_op, center_loss, softmax_loss,pred_class = make_parallel(build_network,num_gpus,input_images=input_images,labels=labels)
    total_loss = make_parallel(build_network,num_gpus,input_images=input_images,labels=labels)
    #optimizer = tf.train.AdamOptimizer(learn_rate)
    optimizer = tf.train.GradientDescentOptimizer(learn_rate)
    #with tf.control_dependencies([centers_update_op]):
    train_op = optimizer.minimize(tf.reduce_mean(total_loss), colocate_gradients_with_ops=True)
    #train_op = optimizer.minimize(total_loss, global_step=global_step)
    summary_op = tf.summary.merge_all()

    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./tmp/face_log', sess.graph)
        saver = tf.train.Saver()
        #begin
        coord = tf.train.Coordinator()
        #begin enqueue thread
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        step = sess.run(global_step)
        epoch_idx =0
        graph_step=0
        item = './data/facescrub_train.list'    
        imagelist = open(item, 'r')
        files_item = imagelist.readlines()
        file_len = len(files_item)
        batch_num = np.ceil(file_len / batch_size)
        while epoch_idx <= epoch_num:
            step = 0
            ckpt_fg = 'True'
            ps_loss=0.0
            pc_loss=0.0
            acc_sum = 0.0
            while step < batch_num:
                train_img_batch, train_label_batch = sess.run([image_batch,label_batch])
                #print("data in ",in_img[0,:2,:2,0])
                _, summary_str,Center_loss = sess.run(
                    [train_op, summary_op,total_loss],
                    feed_dict={
                        input_images: train_img_batch,
                        labels: train_label_batch,
                        learn_rate: lr
                        })
                step += 1
                #print("step",step, str(Softmax_loss),str(Center_loss))
                #print("res1",res1_o[0,:20])
                #print("step label",step, str(batch_labels))
                graph_step+=1
                if step %10 ==0 :
                    writer.add_summary(summary_str, global_step=graph_step)
                pc_loss+=Center_loss
                #ps_loss+=Softmax_loss
                #acc_sum+=train_acc
                if step % 100 == 0:
                    #lr = lr*0.1
                    #c_loss+=c_loss
                    #s_loss+=s_loss
                    print ("****** Epoch {} Step {}: ***********".format(str(epoch_idx),str(step)) )
                    print ("center loss: {}".format(pc_loss/100.0))
                    print ("softmax_loss: {}".format(ps_loss/100.0))
                    print ("train_acc: {}".format(acc_sum/100.0))
                    print ("*******************************")
                    if (Center_loss<0.1  and ckpt_fg=='True'):
                        print("******************************************************************************")
                        saver.save(sess, checkpoint_dir, global_step=epoch_idx)
                        ckpt_fg = 'False'
                    ps_loss=0.0
                    pc_loss=0.0
                    acc_sum=0.0

            epoch_idx +=1

            if epoch_idx % 5 ==0:
                print("******************************************************************************")
                saver.save(sess, checkpoint_dir, global_step=epoch_idx)

            #writer.add_summary(summary_str, global_step=step)
            if epoch_idx % 5 == 0:
                lr = lr*0.5

            if epoch_idx:
                val_img_batch,val_label_batch = sess.run([val_image_batch,val_label_batch])
                vali_acc = sess.run(
                    total_loss,
                    feed_dict={
                        input_images: val_img_batch,
                        labels: val_label_batch
                    })
                print(("epoch: {}, train_acc:{:.4f}, vali_acc:{:.4f}".
                      format(epoch_idx, Center_loss, vali_acc)))
        coord.join(threads)
        sess.close()
if __name__ == '__main__':
    main()
