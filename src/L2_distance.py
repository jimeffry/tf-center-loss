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


def L2_distance(feature1,feature2,lenth):
    _,_,len1 = np.shape(feature1)
    _,_,len2 = np.shape(feature2)
    #print("feature shape: ",np.shape(feature1))
    assert len1 ==lenth and len2==lenth
    mean1 = np.mean(feature1)
    mean2 = np.mean(feature2)
    print("feature mean: ",mean1, mean2)
    f_center1 = feature1-mean1
    f_center2 = feature2-mean2
    std1 = np.sum(np.power(f_center1,2))
    std2 = np.sum(np.power(f_center2,2))
    std1 = np.sqrt(std1)
    std2 = np.sqrt(std2)
    norm1 = f_center1/std1
    norm2 = f_center2/std2
    loss =np.sqrt(np.sum(np.power((norm1-norm2),2)))
    return loss
