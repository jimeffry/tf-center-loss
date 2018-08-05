import numpy as np
import tensorflow as tf
import tflearn
from batch_loader import BatchLoader
from net import *
from Center_loss_custom import *
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
CENTER_LOSS_ALPHA = 0.0

def build_network(input_images, labels,num_class,ratio=0.5):
    with tf.device('/gpu:0'):
        net = face_net(input_images)
        logits, features = net.inference()

    assert num_class== net.num_classes,"net class should be equal to loss"

    with tf.name_scope('loss'):
        with tf.name_scope('center_loss'):
            center_loss, centers, centers_update_op = get_center_loss(features,logits, labels, CENTER_LOSS_ALPHA, num_class)
        with tf.name_scope('softmax_loss'):
            softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        with tf.name_scope('total_loss'):
            total_loss = softmax_loss + ratio * center_loss
    with tf.name_scope('acc'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(logits, 1), labels), tf.float32))
    with tf.name_scope('loss/'):
        tf.summary.scalar('CenterLoss', center_loss)
        tf.summary.scalar('SoftmaxLoss', softmax_loss)
        tf.summary.scalar('TotalLoss', total_loss)
    return logits, features, total_loss, accuracy, centers_update_op, center_loss, softmax_loss

def main():
    LAMBDA = 0.0
    num_class = 526
    checkpoint_dir = "../model/"
    with tf.name_scope('input'):
        input_images = tf.placeholder(tf.float32, shape=(None,100,100,3), name='input_images')
        labels = tf.placeholder(tf.int64, shape=(None), name='labels')
    logits, features, total_loss, accuracy, centers_update_op, center_loss, softmax_loss = build_network(input_images,labels,num_class,ratio=LAMBDA)
    global_step = tf.Variable(0, trainable=False, name='global_step')
    train_batch_loader = BatchLoader("../data/facescrub_train.list", 16)
    test_batch_loader = BatchLoader("../data/facescrub_val.list", 16)
    optimizer = tf.train.AdamOptimizer(0.001)
    with tf.control_dependencies([centers_update_op]):
        train_op = optimizer.minimize(total_loss, global_step=global_step)
    summary_op = tf.summary.merge_all()

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('../tmp/face_log', sess.graph)
        saver = tf.train.Saver()
        step = sess.run(global_step)
        while step <= 80000:
            batch_images, batch_labels = train_batch_loader.next_batch()
            # print batch_images.shape
            # print batch_labels.shape
            _, summary_str, train_acc, Center_loss, Softmax_loss = sess.run(
                [train_op, summary_op, accuracy, center_loss, softmax_loss],
                feed_dict={
                    input_images: (batch_images - 127.5) * 0.0078125, # - mean_data,
                    labels: batch_labels,
                })
            step += 1
            print("step",step)
            if step % 100 == 0:
                print ("********* Step %s: ***********" % str(step))
                print ("center loss: %s" % str(Center_loss))
                print ("softmax_loss: %s" % str(Softmax_loss))
                print ("train_acc: %s" % str(train_acc))
                print ("*******************************")

            if step % 10000 == 0:
                saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=step)

            writer.add_summary(summary_str, global_step=step)

            if step % 2000 == 0:
                batch_images, batch_labels = test_batch_loader.next_batch()
                vali_image = (batch_images - 127.5) * 0.0078125
                vali_acc = sess.run(
                    accuracy,
                    feed_dict={
                        input_images: vali_image,
                        labels: batch_labels
                    })
                print(("step: {}, train_acc:{:.4f}, vali_acc:{:.4f}".
                      format(step, train_acc, vali_acc)))
        sess.close()

if __name__ == '__main__':
    main()
