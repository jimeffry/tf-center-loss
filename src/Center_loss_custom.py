# auther : lxy
# time : 2017.12.15 /09:56
#project:
# tool: python2
#version: 0.1
#modify:
#name: center loss
#citations: "A Discriminative Feature Learning Approach for Deep Face Recognition"
#############################
import numpy as np
import tensorflow as tf
import tflearn

class CenterLoss(object):
    def __init__(self,alpha, num_classes,embd_size):
        self.alpha = alpha
        self.num_classes = num_classes
        self.cnt_call = 0
        self.embd_size = embd_size
        self.centers = tf.get_variable('centers', [num_classes, embd_size], dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=False)
    def get_center_loss(self,features, logits,labels):
        '''
        get center loss and update center
        Arguments:
            features: Tensor,is fc_output, shape [batch_size, feature_length]
            labels: Tensor,shape [batch_size]
            alpha: 0-1
            num_classes: how many classes
        the result:
            loss: Tensor,center_loss
            centers: Tensor,feature centers
            centers_update_op:  update feature centers
        '''
        print ("center loss feature",features.get_shape())
        len_features = features.get_shape()[1]
        batch_size = features.get_shape()[0]
        assert len_features == self.embd_size
        #len_features = features.shape[1]
        #batch_size = features.shape[0]
        #tf.constant_initializer(0)
        #labels = tf.argmax(labels,axis=1)
        labels = tf.reshape(labels, [-1])
        alpha_sel = tf.equal(tf.cast(tf.arg_max(logits, 1),tf.int32), tf.cast(labels,tf.int32))
        a = tf.constant(self.alpha,shape=[batch_size],dtype=tf.float32)
        print ("alpha_sel shape",alpha_sel.get_shape())
        alpha_mask = tf.where(alpha_sel,x= a,y=tf.zeros_like(a))
        #test_al = tf.where(features >0.1)

        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])

        centers_batch = tf.gather(self.centers, labels)
        diff = (centers_batch - features)
        diff = diff / tf.cast((1 + appear_times), tf.float32)
        #loss = tf.nn.l2_loss(diff)
        loss = tf.reduce_sum(tf.abs(diff))
        print("uni_label",unique_label.get_shape())
        print("uni_idx",unique_idx.get_shape())
        print("uni_cont",unique_count.get_shape())
        #diff = diff / tf.cast((1 + appear_times), tf.float32)
        #alpha_mask = tf.concat(alpha_mask,alpha_mask)
        #alpha_mask = tf.stack([alpha_mask,alpha_mask])
        alpha_mask,_ = tf.meshgrid(alpha_mask,tf.constant(0,shape=[len_features],dtype=tf.float32))
        diff = tf.multiply( diff,tf.transpose(alpha_mask))

        centers_update_op = tf.scatter_sub(self.centers, labels, diff)
        #centers_update_op = tf.scatter_add(centers, labels, diff)
        self.centers = centers_update_op
        return loss, self.centers, centers_update_op

def get_softmax_cross_loss(logits,labels,num_class):
    #lg_shape = tf.shape(logits)
    lg_shape = logits.get_shape()
    #lb_shape = tf.shape(labels)
    lb_shape = labels.get_shape()
    print("softmax logit ",lg_shape)
    print("softmax label ",lb_shape)
    assert lg_shape[1] == num_class
    assert lg_shape[0] == lb_shape[0]
    labels_onehot = tf.one_hot(labels,on_value=1,off_value=0,depth=num_class)
    softmax_out = tf.nn.softmax(logits)
    log_soft = tf.log(tf.clip_by_value(softmax_out,1e-8,1.0))
    labels_onehot = tf.cast(labels_onehot,tf.float32)
    log_pred = tf.multiply(log_soft,labels_onehot)
    batch_loss = -tf.reduce_sum(log_pred)
    return batch_loss


if __name__ == '__main__':
    #features_t = np.random.rand(3,2)
    #label = np.random.randint(0,10,size=(10))
    features_t = np.array([[0.5,0.2],[0.11,0.03],[0.4,0.6]])
    label_i = np.array([0,1,2],dtype=np.int32)
    logit_i = np.array([[0.2,0.1,0.1],[0.5,0.4,0.3],[0.1,0.6,0.8]],dtype=np.float32)
    batch_size = 3
    features = tf.placeholder(tf.float32, shape=(batch_size,2), name='input_images')
    logits = tf.placeholder(tf.float32, shape=(batch_size,3), name='input_f')
    label = tf.placeholder(tf.int32, shape=(batch_size), name='input_l')
    los,cent,op,c_index = get_center_loss(features,logits,label,0.9,3)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ind = sess.run([c_index], feed_dict={features: features_t, logits:logit_i, label:label_i})
        #print tf.maximum(ind[0],1)
    print (ind)
    print (ind[0][:,0])
    f_t = np.array([[[0.5,0.2],[0.11,0.03],[0.4,0.6]],[[0.5,0.2],[0.11,0.03],[0.4,0.6]],[[0.5,0.2],[0.11,0.03],[0.4,0.6]]])
    print (f_t[ind[0][:,0],ind[0][:,1],:])
    c= f_t[ind[0][:,0],ind[0][:,1],:]
    print (c[:,0])
