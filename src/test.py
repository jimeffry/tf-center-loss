# auther : lxy
# time : 2017.12.26 /09:56
#project:
# tool: python2
#version: 0.1
#modify:
#name: center loss
#citations: https://github.com/ydwen/caffe-face
#############################
import numpy as np
import tensorflow as tf
from L2_distance import *
from batch_loader import BatchLoader
from get_faces import FaceDetector
from net import *
import argparse
import cv2
import os
import string
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
#####################
import time

Test_Img = 'File'
SV_IMG = 0
def args():
    parser = argparse.ArgumentParser(description='face_reg test mode')
    parser.add_argument('--model_prefix',type=str,default='../face_model/model.ckpt', help='the prefix of the model to load')
    parser.add_argument('--load_epoch',type = int,default=5, help='load the model on an epoch ')
    parser.add_argument('--batch_size',type=int,default=1,help='numbers of images loaded once')
    parser.add_argument('--image1',type=str,default='../images/test1.png',help='image directory')
    parser.add_argument('--image2',type=str,default='../images/test2.png',help='image directory')
    parser.add_argument('--image_shape',type=int,default=112,help='the net input image shape')
    parser.add_argument('--img_dir',type=str,default='../images',help='image path includes images')
    parser.add_argument('--file_txt',type=str,default='./highway.txt',help='annotation files')
    arg = parser.parse_args()
    return arg


def build_network(input_images, labels,num_class,sta='test'):
    with tf.device('/gpu:0'):
        net = face_net(input_images,num_class,sta)
        logits, features = net.get_resnet18()
        res1 = net.res1
    with tf.name_scope('pred_class'):
        pred_class = tf.arg_max(logits, 1)
        pred_class = tf.cast(pred_class,tf.int32)
    with tf.name_scope('acc'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_class, labels), tf.float32))
    return  features,accuracy, pred_class,res1

def save_image(path_dir,cropface):
    cv2.imwrite(path_dir,cropface)

def mkdir_(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def check_ckpt(file_name):
    print_tensors_in_checkpoint_file(file_name, 0, all_tensors=0)

def face_dect():
    prefix = ["../models/MTCNN_bright_model/PNet_landmark/PNet", \
            "../models/MTCNN_bright_model/RNet_landmark/RNet", \
            "../models/MTCNN_bright_model/ONet_landmark/ONet"]
    detector = FaceDetector(prefix)
    return detector

def get_img(img,bbox,img_size):
    x1,y1 = bbox[0],bbox[1]
    x2,y2 = bbox[2],bbox[3]
    x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])
    img_roi = img[y1: (y2 +1),x1: (x2 +1),:]
    crop_img = cv2.resize(img_roi,(img_size,img_size))
    return crop_img

def main():
    params = args()
    model_prefix = params.model_prefix
    load_epoch = params.load_epoch
    batch_size = params.batch_size
    img1_path = params.image1
    img2_path = params.image2
    img_shape = params.image_shape
    img_dir = params.img_dir
    txt_path =  params.file_txt
    # in out files
    file_ = open(txt_path,'r')
    lines_ = file_.readlines()
    result_ = open("result.txt",'w')
    #
    if Test_Img == 'True':
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        img1 = cv2.resize(img1,(img_shape,img_shape))
        img2 = cv2.resize(img2,(img_shape,img_shape))
        img1 = np.expand_dims(img1,0)
        img2 = np.expand_dims(img2,0)

    test_batch_loader = BatchLoader("../data/facescrub_val.list", batch_size,img_shape)
    
    tf.reset_default_graph()
    with tf.name_scope('input'):
        input_images = tf.placeholder(tf.float32, shape=(batch_size,img_shape,img_shape,3), name='input_images')
        labels = tf.placeholder(tf.int32,shape=(batch_size),name='labels')
    features,accuracy,pred_class,res1 = build_network(input_images,labels,526,'test')
    check_ckpt(model_prefix+'-'+str(load_epoch))
    #detect
    detector = face_dect()
    #
    with tf.Session() as sess:
        restore_model = tf.train.Saver()
        #restore_model = tf.train.import_meta_graph(model_prefix+'-'+str(load_epoch) +'.meta')
        restore_model.restore(sess,model_prefix+'-'+str(load_epoch))
        print("face model restore over")
        '''
        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        key_list = []
        var_dic = dict()
        for v_name in tf.global_variables():
            i=0
            print("name : ",v_name.name[:-2],v_name.shape) 
            print("shape",all_vars[i])
            key_list.append(v_name.name[:-2])
            i+=1
            #print(tf.get_variable_scope())
        #all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        vas= sess.run([all_vars])
        print(len(vas))
        for i in range(len(vas)):
            cur_name = key_list[i]
            cur_var = vas[i]
            print("name ,shape : ",cur_name,np.shape(cur_var))
            var_dic[cur_name] = cur_var
        '''
        #restore_model = tf.train.import_meta_graph(model_prefix+'-'+str(load_epoch)+'.meta')
        #restore_model.restore(sess,model_prefix+'-'+str(load_epoch))
        if  Test_Img =='False' :
            iter_num = 0
            accuracy_sum = 0
            while iter_num < test_batch_loader.batch_num:
                batch_images,batch_labels = test_batch_loader.next_batch()
                images_in = (batch_images - 127.5)*0.0078125
                feat,batch_accuracy = sess.run([features,accuracy],
                                            feed_dict={input_images:images_in,
                                            labels:batch_labels})
                accuracy_sum +=batch_accuracy
                iter_num += 1
                if iter_num % 10==0:
                    print("step ",iter_num, batch_accuracy)
            print("image num: ",test_batch_loader.data_num,"the test accuracy: ",accuracy_sum / (test_batch_loader.batch_num))
        elif Test_Img == 'True':
            with tf.name_scope('valdata'):
                label_t = np.zeros([1])
                feat1 = sess.run([features],feed_dict={input_images:img1,labels:label_t})
                feat2 = sess.run([features],feed_dict={input_images:img2,labels:label_t})
                distance = L2_distance(feat1,feat2,512)
            print("the 2 image dis: ",distance)
            for rot,fdir,fname in os.walk(img_dir):
                if len(fname) !=0 :
                    break
            img_list = []
            print (fname)
            for i in range(len(fname)):
                org_img = cv2.imread(os.path.join(rot,fname[i]))
                img_org = cv2.resize(org_img,(img_shape,img_shape))
                img_org = np.expand_dims(img_org,0)
                img_list.append(img_org)
            for i in range(len(fname)):
                img1 = img_list[i]
                feat1 = sess.run([features],feed_dict={input_images:img1,labels:label_t})
                j = i+1
                while j < len(fname):
                    img2 = img_list[j]
                    t1 = time.time()
                    feat2 = sess.run([features],feed_dict={input_images:img2,labels:label_t})
                    t2 = time.time()
                    print("one image time ",t2-t1)
                    distance = L2_distance(feat1,feat2,512)
                    t3 = time.time()
                    print("compere time ",t3-t2)
                    print(i , j , distance)
                    j +=1
        elif Test_Img == 'File':
            label_t = np.ones([1])
            for i in range(len(lines_)):
                feat_vec = []
                feat1_fg = 0
                feat2_fg = 0
                line_1 = lines_[i]
                line_1 = string.strip(line_1)
                line_s = line_1.split(',')
                dir_path_save = line_s[0][:-4]
                dir_path_save = "../cropface/"+dir_path_save
                mkdir_(dir_path_save)
                for j in range(len(line_s)):
                    feat_vec2 = []
                    if j == 0:
                        #print("line ",line_s)
                        img1_pic = line_s[0]
                        img1_path = os.path.join(img_dir,img1_pic)
                        img1 = cv2.imread(img1_path)
                        bboxes_1 = detector.get_face(img1)
                        if bboxes_1 is not None:
                            for k in range(bboxes_1.shape[0]):
                                crop_img1 = get_img(img1,bboxes_1[k],img_shape)
                                if k==0 and SV_IMG:
                                    img_save_path = dir_path_save+'/'+line_s[0][:-4]+".jpg"
                                    save_image(img_save_path,crop_img1)
                                crop_img1 = (crop_img1 - 127.5)*0.0078125
                                crop_img1 = np.expand_dims(crop_img1,0)
                                feat1 = sess.run([features],feed_dict={input_images:crop_img1,labels:label_t})
                                print("a feature shape ",np.shape(feat1))
                                feat_vec.append(feat1)
                                feat_fg =1
                        else:
                            print("a no face detect ")
                            break
                    else:
                        img2_pic = line_s[j]
                        img2_path = os.path.join(img_dir,img2_pic)
                        img2 = cv2.imread(img2_path)
                        bboxes_2 = detector.get_face(img2)
                        if bboxes_2 is not None:
                            for k in range(bboxes_2.shape[0]):
                                crop_img2 = get_img(img2,bboxes_2[k],img_shape)
                                if SV_IMG:
                                    img_save_path = dir_path_save+'/'+line_s[j][:-4]+"-"+str(k)+".jpg"
                                    save_image(img_save_path,crop_img2)
                                crop_img2 = (crop_img2 - 127.5)*0.0078125
                                crop_img2 = np.expand_dims(crop_img2,0)
                                feat2 = sess.run([features],feed_dict={input_images:crop_img2,labels:label_t})
                                print("b feature shape ",np.shape(feat2))
                                feat_vec2.append(feat2)
                                feat2_fg = 1
                        else:
                            print("b no face detect ")
                            continue
                    if j>0:
                        t2 = time.time()
                        distance = L2_distance(feat_vec[0],feat_vec2[0],512)
                        print("distance is ",distance)
                        t3 = time.time()
                        print("compere time ",t3-t2)
                        result_.write("{} ".format(distance))
                result_.write("\n")
                print(feat2)
    file_.close()
    result_.close()
    
if __name__ =='__main__':
    main()
