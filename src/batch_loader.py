"""
Batch Loader by Donny You
"""
import cv2
import numpy as np
import numpy.random as nr
from random import shuffle
import os


class BatchLoader(object):

    def __init__(self, file_path, batch_size,img_shape):
        self.batch_size = batch_size
        self.labels, self.im_list = self.image_dir_processor(file_path)
        self.idx = 0
        self.data_num = len(self.labels)
        self.rnd_list = np.arange(self.data_num)
        shuffle(self.rnd_list)
        self.img_temp = cv2.imread(self.im_list[0])
        self.height,self.width,_= np.shape(self.img_temp)
        self.batch_num = np.ceil(self.data_num / batch_size)
        self.img_h,self.img_w = img_shape
        #self.labels = tf.one_hot(self.labels,on_value=1,off_value=0,depth=526)
        #print(self.labels)

    def next_batch(self):
        batch_images = []
        batch_labels = []

        for i in xrange (self.batch_size):
            if self.idx < self.data_num:
                cur_idx = self.rnd_list[self.idx]
                im_path = self.im_list[cur_idx]
                image = cv2.imread(im_path)
                row,col,chal = np.shape(image)
                if not chal == 3:
                    '''
                    temp_im = np.zeros([300,300,3],dtype=np.float32)
                    temp_im[:,:,0]=image
                    temp_im[:,:,1]=image
                    temp_im[:,:,2]=image
                    image = temp_im
                    '''
                    continue
                if row != self.img_h or col !=self.img_w :
                    image = cv2.resize(image,(self.img_w,self.img_h))
                #mean_data = np.mean(image)
                #max_d = np.max(image)/2
                #image = (image - mean_data)/max_d
                #print(mean_data, max_d)
                batch_images.append(image)
                #onehot_label = np.zeros([self.data_num])
                #onehot_label[self.labels[cur_idx]] = 1
                batch_labels.append(self.labels[cur_idx])

                self.idx +=1
            elif self.data_num % self.batch_size !=0:
                remainder = self.data_num % self.batch_size
                patch_num = self.batch_size - remainder
                for j in xrange(patch_num):
                    cur_idx = self.rnd_list[j]
                    im_path = self.im_list[cur_idx]
                    image = cv2.imread(im_path)
                    row,col,chal = np.shape(image)
                    if not chal == 3:
                        '''
                        temp_im = np.zeros([300,300,3],dtype=np.float32)
                        temp_im[:,:,0]=image
                        temp_im[:,:,1]=image
                        temp_im[:,:,2]=image
                        image = temp_im
                        '''
                        continue
                    if row != self.img_h or col !=self.img_w :
                        image = cv2.resize(image,(self.img_w,self.img_h))
                    #mean_data = np.mean(image)
                    #max_d = np.max(image)/2
                    #image = (image - mean_data)/max_d
                    #print(mean_data, max_d)
                    batch_images.append(image)
                    #onehot_label = np.zeros([self.data_num])
                    #onehot_label[self.labels[cur_idx]] = 1
                    #batch_labels.append(onehot_label)
                    batch_labels.append(self.labels[cur_idx])
                self.idx = 0
                shuffle(self.rnd_list)
                break
            else:
                self.idx = 0
                shuffle(self.rnd_list)
        #print("indx",self.idx,"batch",len(batch_images))
        batch_images = np.array(batch_images).astype(np.float32)
        batch_labels = np.array(batch_labels).astype(np.int32)
        return batch_images, batch_labels

    def image_dir_processor(self, file_path):
        labels = []
        im_path_list = []
        if not os.path.exists(file_path):
            print ("File %s not exists." % file_path)
            exit()

        with open(file_path, "r") as fr:
            for line in fr.readlines():
                terms = line.rstrip().split()
                label = int(terms[1])
                im_path_list.append(terms[0])
                labels.append(label)

        return labels, im_path_list

if __name__ =='__main__':
    data = BatchLoader("./data/facescrub_train.list", 128,112)
    a,b = data.next_batch()
    print(b.shape)
