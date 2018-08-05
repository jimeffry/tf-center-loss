# auther : lxy
# time : 2018.2.23 /16:50
#project:
# tool: python2
#version: 0.1
#modify:
#name: center loss
#citations: https://github.com/ydwen/caffe-face
#############################
# ==============================================================================

"""A binary to train center_loss using multiple GPUs with synchronous updates.

Accuracy:
cifar10_multi_gpu_train.py achieves ~86% accuracy after 100K steps (256
epochs of data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
--------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
2 Tesla K20m  | 0.13-0.20              | ~84% at 30K steps  (2.5 hours)
3 Tesla K20m  | 0.13-0.18              | ~84% at 30K steps
4 Tesla K20m  | ~0.10                  | ~84% at 30K steps

Usage:

"""
from datetime import datetime
import os.path
import re
import time

import numpy as np
#from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
from read_tfrecord_v2 import read_single_tfrecord
from net import *
from Center_loss_custom import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './tmp/face_log',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 2,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_float('learn_rate', 0.001,
                            """learning rate.""")
tf.app.flags.DEFINE_string('tower_name', 'tower',
                           """loss tower name """)
tf.app.flags.DEFINE_integer('batch_size',128,
                           """batch size for training""")

def tower_loss(scope, input_images, labels):
    """Calculate the total loss on a single tower running the CIFAR model.

    Args:
      scope: unique prefix string identifying the face tower, e.g. 'tower_0'
      images: Images. 4D tensor of shape [batch_size, height, width, 3].
      labels: Labels. 1D tensor of shape [batch_size].

    Returns:
       Tensor of shape [] containing the total loss for a batch of data
    """
    num_class = 526
    sta = 'train'
    ratio = 0.003  
    net = face_net(input_images,num_class,sta)
    #logits, features = net.inference()
    logits, features = net.get_resnet18()
    assert num_class== net.num_classes,"net class should be equal to loss"
    labels = tf.cast(labels,tf.int32)
    with tf.name_scope('center_loss'):
        center_loss, centers, centers_update_op = get_center_loss(features,logits, labels, 0.9, num_class)
    with tf.name_scope('softmax_loss'):
        #softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        labels_onehot = tf.one_hot(labels,on_value=1,off_value=0,depth=num_class)
        entropy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_onehot, logits=logits)
        #print("entropy_loss ",entropy_loss.shape)
        softmax_loss = tf.reduce_mean(entropy_loss)
    with tf.name_scope(scope):
        #regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        #total_loss = softmax_loss + ratio * center_loss+0.01 * sum(regularization_losses)
        total_loss = softmax_loss + ratio * center_loss
        #total_loss = softmax_loss
    with tf.name_scope('acc'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.arg_max(logits, 1),tf.int32), labels), tf.float32))
    with tf.name_scope('pred_class'):
        pred_class = tf.arg_max(logits, 1)
    '''
    with tf.name_scope('loss/'):
        tf.summary.scalar('CenterLoss', center_loss)
        tf.summary.scalar('SoftmaxLoss', softmax_loss)
        tf.summary.scalar('TotalLoss', total_loss)
    '''

    # Build inference Graph.
    #logits = cifar10.inference(images)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    #_ = cifar10.loss(logits, labels)

    # Assemble all of the losses for the current tower only.
    #losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    #total_loss = tf.add_n(losses, name='total_loss')

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
    #loss_name = re.sub('%s_[0-9]*/' % FLAGS.tower_name, '', total_loss.op.name)
    #tf.summary.scalar(loss_name, total_loss)
    return total_loss,centers_update_op


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, tm in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      #expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      #print("gradient shape ",tf.shape(g))
      print("var name ",tm)
      grads.append(g)

    # Average over the 'tower' dimension.
    #grad = tf.concat(axis=0, values=grads)
    grad = tf.stack(grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def train():
  """Train face_reg for a number of steps."""
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    # Calculate the learning rate schedule.
    item = './data/facescrub_train.list'    
    imagelist = open(item, 'r')
    files_item = imagelist.readlines()
    file_len = len(files_item)
    num_batches_per_epoch = (file_len /FLAGS.batch_size)
    decay_steps = int(num_batches_per_epoch * 10)
    batch_size = FLAGS.batch_size
    img_shape = 300

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(FLAGS.learn_rate,
                                    global_step,
                                    decay_steps,
                                    0.1,
                                    staircase=True)

    # Create an optimizer that performs gradient descent.
    opt = tf.train.GradientDescentOptimizer(lr)

    # Get images and labels for CIFAR-10.
    tfrecord_file = './data/MegaFace_train.tfrecord_shuffle'
    val_file = './data/MegaFace_val.tfrecord_shuffle'
    images, labels = read_single_tfrecord(tfrecord_file, batch_size, img_shape)
    val_image_batch, val_label_batch = read_single_tfrecord(val_file, batch_size, img_shape)
    batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
          [images, labels], capacity=2 * FLAGS.num_gpus)
    # Calculate the gradients for each model tower.
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
      for i in range(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % ('tower', i)) as scope:
            # Dequeues one batch for the GPU
            image_batch, label_batch = batch_queue.dequeue()
            # Calculate the loss for one tower of the CIFAR model. This function
            # constructs the entire CIFAR model but shares the variables across
            # all towers.
            loss,center_op = tower_loss(scope, image_batch, label_batch)

            # Reuse variables for the next tower.
            tf.get_variable_scope().reuse_variables()

            # Retain the summaries from the final tower.
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

            # Calculate the gradients for the batch of data on this CIFAR tower.
            with tf.control_dependencies([center_op]):
                grads = opt.compute_gradients(loss)

            # Keep track of the gradients across all towers.
            tower_grads.append(grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    #print("gradient shape ",tf.shape(tower_grads))
    grads = average_gradients(tower_grads)

    # Add a summary to track the learning rate.
    summaries.append(tf.summary.scalar('learning_rate', lr))

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      summaries.append(tf.summary.histogram(var.op.name, var))

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op, variables_averages_op)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries)

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    for step in range(FLAGS.max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / FLAGS.num_gpus

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % (num_batches_per_epoch /FLAGS.num_gpus) == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
  #cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
