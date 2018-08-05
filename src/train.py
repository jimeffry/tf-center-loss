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
from batch_loader import BatchLoader
from net import *
from Center_loss_custom import CenterLoss,get_softmax_cross_loss
import argparse


def argument():
    parser = argparse.ArgumentParser(description="face resnet center loss")
    parser.add_argument('--batch_size',type=int,default=16,help='the batch_size num')
    parser.add_argument('--epoch_num',type=int,default=10,\
                        help='the epoch num should bigger than 10000')
    parser.add_argument('--save_model_name',type=str,default='../face_model/model_ckpt',\
                        help='model Parameters saved name and directory')
    parser.add_argument('--lr',type=float,default=0.001,help='the Learning rate begin')
    parser.add_argument('--sta',type=str,default='train',help="input should 'train' or 'test' ")
    parser.add_argument('--img_size',default=[112,96],nargs="+",help="the input image reshape size")
    parser.add_argument('--pretrained',type=int,default=None,help="pretrained model num")
    parser.add_argument('--train_file',type=str,default='train.txt',help="train img list")
    parser.add_argument('--val_file',type=str,default='val.txt',help="val img list")
    args = parser.parse_args()
    return args

def build_network(input_images, labels,num_class,sta,loss_op,ratio=0.003):
    net = face_net(input_images,num_class,sta)
    #logits, features = net.inference()
    pred_logits, features = net.get_resnet18()
    res1 = net.fc1_r
    assert num_class== net.num_classes,"net class should be equal to loss"
    with tf.name_scope('loss'):
        with tf.name_scope('center_loss'):
            center_loss, centers, centers_update_op = loss_op.get_center_loss(features,pred_logits, labels)
        with tf.name_scope('softmax_loss'):
            #softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
            #labels_onehot = tf.one_hot(labels,on_value=1,off_value=0,depth=num_class)
            #entropy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_onehot, logits=logits)
            #entropy_loss = tf.losses.softmax_cross_entropy(labels_onehot,pred_logits)
            entropy_loss = get_softmax_cross_loss(pred_logits,labels,num_class)
            print("entropy_loss ",entropy_loss.shape)
            softmax_loss = tf.reduce_mean(entropy_loss)
        with tf.name_scope('total_loss'):
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            total_loss = softmax_loss + ratio * center_loss+ratio * sum(regularization_losses)
            #total_loss = softmax_loss
    with tf.name_scope('acc'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.arg_max(pred_logits, 1),tf.int32), labels), tf.float32))
    with tf.name_scope('pred_class'):
        pred_class = tf.arg_max(pred_logits, 1)
    with tf.name_scope('loss/'):
        tf.summary.scalar('CenterLoss', center_loss)
        tf.summary.scalar('SoftmaxLoss', softmax_loss)
        tf.summary.scalar('TotalLoss', total_loss)
    return total_loss, accuracy, centers_update_op, center_loss, softmax_loss,pred_class,centers

def make_parallel(model,num_gpus,**kwargs):
    in_splits = {}
    for k,v in kwargs.items():
        in_splits[k] = tf.split(v,num_gpus)
    out_splits = {}
    for i in range(num_gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU",device_index=i)):
            with tf.variable_scope(tf.get_variable_scope(),reuse=i>0):
                out_splits.append(model(**{k:v[i] for k,v in in_splits.items()}))
    return tf.concat(out_splits,axis=0)


def main():
    LAMBDA = 1e-8
    center_alpha = 0.9
    num_class = 10000
    embd_size = 512
    args = argument()
    checkpoint_dir = args.save_model_name
    lr = args.lr
    batch_size = args.batch_size
    epoch_num = args.epoch_num
    sta = args.sta
    img_shape = args.img_size
    train_file = args.train_file
    val_file = args.val_file
    train_batch_loader = BatchLoader(train_file, batch_size,img_shape)
    test_batch_loader = BatchLoader(val_file, batch_size,img_shape)
    #(Height,Width) = (train_batch_loader.height,train_batch_loader.width)
    #train_batch_loader = mnist_data(batch_size)
    print("img shape",img_shape)
    with tf.name_scope('input'):
        input_images = tf.placeholder(tf.float32, shape=(batch_size,img_shape[0],img_shape[1],3), name='input_images')
        labels = tf.placeholder(tf.int32, shape=(batch_size), name='labels')
        learn_rate = tf.placeholder(tf.float32,shape=(None),name='learn_rate')
    with tf.name_scope('var'):
        global_step = tf.Variable(0, trainable=False, name='global_step')
    loss_op = CenterLoss(center_alpha,num_class,embd_size)
    #with tf.device('/gpu:0'):
    total_loss, accuracy, centers_update_op, center_loss, softmax_loss,pred_class,res1 = build_network(input_images,labels,num_class,sta,loss_op,ratio=LAMBDA)
    optimizer = tf.train.AdamOptimizer(learn_rate)
    #optimizer = tf.train.GradientDescentOptimizer(learn_rate)
    with tf.control_dependencies([centers_update_op]):
        train_op = optimizer.minimize(total_loss, global_step=global_step)
    #train_op = optimizer.minimize(total_loss, global_step=global_step)
    summary_op = tf.summary.merge_all()
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        #sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('../tmp/face_log', sess.graph)
        saver = tf.train.Saver()
        if args.pretrained is not None :
            model_path = args.save_model_name+'-'+str(args.pretrained)
            #saver.restore(sess,'./face_model/high_score-60')
            saver.restore(sess,model_path)
        else:
            init = tf.global_variables_initializer()
            sess.run(init)
        step = sess.run(global_step)
        epoch_idx =0
        graph_step=0
        while epoch_idx <= epoch_num:
            step = 0
            ckpt_fg = 'True'
            ps_loss=0.0
            pc_loss=0.0
            acc_sum = 0.0
            while step < train_batch_loader.batch_num:
                batch_images, batch_labels = train_batch_loader.next_batch()
                #batch_images, batch_labels = train_batch_loader.get_batchdata()
                in_imgs=(batch_images - 127.5) * 0.0078125
                #print("data in ",in_img[0,:2,:2,0])
                _, summary_str, train_acc, Center_loss, Softmax_loss,Pred_class,res1_o = sess.run(
                    [train_op, summary_op, accuracy, center_loss, softmax_loss,pred_class,res1],
                    feed_dict={
                        input_images: in_imgs,
                        labels: batch_labels,
                        learn_rate: lr
                        })
                step += 1
                #print("step",step, str(Softmax_loss),str(Center_loss))
                #print("step label",step, str(batch_labels))
                graph_step+=1
                if step %100 ==0 :
                    writer.add_summary(summary_str, global_step=graph_step)
                pc_loss+=Center_loss
                ps_loss+=Softmax_loss
                acc_sum+=train_acc
                if step % 1000 == 0:
                    #lr = lr*0.1
                    #c_loss+=c_loss
                    #s_loss+=s_loss
                    print ("****** Epoch {} Step {}: ***********".format(str(epoch_idx),str(step)) )
                    print ("center loss: {}".format(pc_loss/1000.0))
                    print ("softmax_loss: {}".format(ps_loss/1000.0))
                    print ("train_acc: {}".format(acc_sum/1000.0))
                    print("centers",res1_o[0,:5])
                    print ("*******************************")
                    #if (acc_sum/100.0) >= 0.98 and (pc_loss/100.0)<40 and (ps_loss/100.0) <0.1 and ckpt_fg=='True':
                    if  ckpt_fg=='True':
                        print("******************************************************************************")
                        saver.save(sess, checkpoint_dir, global_step=epoch_idx)
                        ckpt_fg = 'False'
                    ps_loss=0.0
                    pc_loss=0.0
                    acc_sum=0.0

            epoch_idx +=1

            if epoch_idx % 10 ==0:
                print("******************************************************************************")
                saver.save(sess, checkpoint_dir, global_step=epoch_idx)

            #writer.add_summary(summary_str, global_step=step)
            if epoch_idx % 5 == 0:
                lr = lr*0.5

            if epoch_idx:
                batch_images, batch_labels = test_batch_loader.next_batch()
                #batch_images,batch_labels = train_batch_loader.get_valdata()
                vali_image = (batch_images - 127.5) * 0.0078125
                vali_acc = sess.run(
                    accuracy,
                    feed_dict={
                        input_images: vali_image,
                        labels: batch_labels
                    })
                print(("epoch: {}, train_acc:{:.4f}, vali_acc:{:.4f}".
                      format(epoch_idx, train_acc, vali_acc)))
        sess.close()
if __name__ == '__main__':
    main()
