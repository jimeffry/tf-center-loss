# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2017/08/04 14:09
#project: Face recognize
#company: Senscape
#rversion: 0.1
#tool:   python 2.7
#modified:2018/06/14 09:24
#description  generate txt file,for example: img1.png  img1.png img2.png  ... img_n.png
####################################################
import numpy as np 
import os
import sys
import argparse
import shutil

def parms():
    parser = argparse.ArgumentParser(description='gen img lists for confusion matrix test')
    parser.add_argument('--img-dir',type=str,dest="img_dir",default='/home/lxy/Pictures',\
                        help='the directory should include 2 more Picturess')
    parser.add_argument('--file-in',type=str,dest="file_in",default="train.txt",\
                        help='img paths saved file')
    parser.add_argument('--save-dir',type=str,dest="save_dir",default='/home/lxy/Pictures',\
                        help='img saved dir')
    parser.add_argument('--base-label',type=int,dest="base_label",default=0,\
                        help='the base label')
    parser.add_argument('--out-file',type=str,dest="out_file",default="train.txt",\
                        help='img paths saved file')
    return parser.parse_args()

def get_from_dir():
    args = parms()
    out_file = open('confuMax_test.txt','w')
    directory = args.img_dir
    filename = []
    for root,dirs,files in os.walk(directory):
        for file_1 in files:
            print("file:", file_1)
            filename.append(file_1)
    filename = np.sort(filename)
    print(len(filename))
    print(filename)
    for i in range(0,len(filename)):
        #print(len(filename))
        #out_file.write("{},".format(filename[i]))
        if i < len(filename):
            out_file.write("{},".format(filename[i]))
            #print(filename)
            for j in range(0,len(filename)):
                if j < len(filename)-1:
                    out_file.write("{},".format(filename[j]))
                else:
                    out_file.write("{}".format(filename[j]))
            out_file.write("\n")
    out_file.close()


def get_from_txt(file_in):
    f_in = open(file_in,'r')
    f_out = open('confuMax_test.txt','w')
    filenames = f_in.readlines()
    filenames = np.sort(filenames)
    print("total ",len(filenames))
    #print(filenames[0])
    for i in range(len(filenames)):
        if i < len(filenames):
            f_out.write("{},".format(filenames[i].strip()))
            #print(filename)
            for j in range(0,len(filenames)):
                if j < len(filenames)-1:
                    f_out.write("{},".format(filenames[j].strip()))
                else:
                    f_out.write("{}".format(filenames[j].strip()))
            f_out.write("\n")
    f_in.close()
    f_out.close()

def generate_list_from_dir(dirpath):
    f_w = open("ms_celeb_1m.txt",'w')
    files = os.listdir(dirpath)
    total_ = len(files)
    print("total id ",len(files))
    idx =0
    file_name = []
    total_cnt = 0
    for file_cnt in files:
        img_dir = os.path.join(dirpath,file_cnt)
        imgs = os.listdir(img_dir)
        idx+=1
        sys.stdout.write("\r>>convert  %d/%d" %(idx,total_))
        sys.stdout.flush()
        #label_ = int(file_cnt)+10575
        for img_one in imgs:
            img_path = os.path.join(file_cnt,img_one)
            #file_name.append(img_path)
            #f_w.write("{} {}\n".format(img_path,label_))
            total_cnt+=1
            f_w.write("{}\n".format(img_path))
    #file_name = np.sort(file_name)
    #print(file_name)
    #f_n = open("filenames.txt",'w')
    print("total img ",total_cnt)
    cnt = 0
    '''
    for i in range(len(file_name)):
        cnt+=1
        f_n.write("'{}',".format(file_name[i]))
        if cnt == 20 :
            cnt = 0
            f_n.write("\n")
    '''
    #for i in range(len(file_name)):
        #cnt+=1
        #f_n.write("{}\n".format(file_name[i]))
    f_w.close()
    #f_n.close()

def read_1(file_in):
    f_= open(file_in,'r')
    lines = f_.readlines()
    print(lines[-1])
    f_.close()

def gen_filefromdir(base_dir):
    f_w = open("data0.txt",'w')
    files = os.listdir(base_dir)
    total_ = len(files)
    print("total id ",len(files))
    for file_cnt in files:
        f_w.write("{}\n".format(file_cnt))
    f_w.close()

def merge2txt(file1,file2):
    f1 = open(file1,'r')
    f2 = open(file2,'r')
    f_out = open("db_test.txt",'w')
    id_files = f1.readlines()
    imgs = f2.readlines()
    for img_gal in imgs:
        f_out.write("{},".format(img_gal.strip()))
        for j in range(len(id_files)):
            if j < len(id_files)-1:
                f_out.write("{},".format(id_files[j].strip()))
            else:
                f_out.write("{}".format(id_files[j].strip()))
        f_out.write("\n")
    f1.close()
    f2.close()
    f_out.close()
    print("over")

def gen_dirfromtxt(file_p,base_dir,save_dir):
    f_in = open(file_p,'r')
    file_lines = f_in.readlines()
    file_lines = np.sort(file_lines)
    file_cnt = 0
    total_cnt = 0
    print("total id ",len(file_lines))
    for line_one in file_lines: 
        if total_cnt % 10000==0:
            dist_dir = os.path.join(save_dir,str(file_cnt))
            if not os.path.exists(dist_dir):
                os.makedirs(dist_dir)
            file_cnt+=1
            print(file_cnt)
        total_cnt+=1
        line_one = line_one.strip()
        org_path = os.path.join(base_dir,line_one)
        dist_path = os.path.join(dist_dir,line_one)
        shutil.copyfile(org_path,dist_path)
    f_in.close()


def generate_label_from_dir(dirpath,base_label,out_file):
    f_w = open(out_file,'w')
    f2_w = open("MS_1M_val.txt",'w')
    files = os.listdir(dirpath)
    files = np.sort(files)
    total_ = len(files)
    print("total id ",len(files))
    idx =0
    cnt = 0
    for file_cnt,line_one in enumerate(files):
        img_dir = os.path.join(dirpath,line_one)
        imgs = os.listdir(img_dir)
        cnt = 0
        idx+=1
        sys.stdout.write("\r>>convert  %d/%d" %(idx,total_))
        sys.stdout.flush()
        label_ = int(file_cnt)+base_label
        if int(file_cnt) == 10000:
            break
        for img_one in imgs:
            if len(img_one)<=0:
                continue
            img_path = os.path.join(img_dir,img_one)
            if cnt == len(imgs)-1:
                f2_w.write("{} {}\n".format(img_path,label_))
                break
            f_w.write("{} {}\n".format(img_path,label_))
            '''
            if cnt ==100000:
                #f2_w.write("{} {}\n".format(img_path,label_))
                f2 = 0
            else:
                f_w.write("{} {}\n".format(img_path,label_))
            '''
            cnt+=1
    f_w.close()
    print("total ",cnt)
    #f2_w.close()

if __name__ == "__main__":
    args = parms()
    txt_file = args.file_in
    #get_from_txt(txt_file)
    img_dir = args.img_dir
    save_dir = args.save_dir
    base_label = args.base_label
    out_file = args.out_file
    #generate_list_from_dir(img_dir)
    #read_1(txt_file)
    fil2="id_crop.txt"
    fil1="prison_train.txt"
    #gen_filefromdir(img_dir)
    #merge2txt(fil1,fil2)
    #gen_dirfromtxt(txt_file,img_dir,save_dir)
    generate_label_from_dir(img_dir,base_label,out_file)
