#!pip install opencv-python

import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def read_ct_img_bydir(target_dir):
    img=cv2.imdecode(np.fromfile(target_dir,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
    #img = corp_margin(img)
    img=cv2.resize(img,(200,200))
    
    return img


def get_data():
    '''Loads raw data from image names of each class folder and assigns a label to it.'''
    
    target_dir1='/home/adduser/code/NinLev/Covid_19/raw_data/labeled_CT_data/niCT'
    target_dir2='/home/adduser/code/NinLev/Covid_19/raw_data/labeled_CT_data/pCT'
    target_dir3='/home/adduser/code/NinLev/Covid_19/raw_data/labeled_CT_data/nCT'
    
    target_list1=[target_dir1+file for file in os.listdir(target_dir1)]
    target_list2=[target_dir2+file for file in os.listdir(target_dir2)]
    target_list3=[target_dir3+file for file in os.listdir(target_dir3)]

    target_list=target_list1+target_list2+target_list3

    # Assign labels: 0: 'non informative CT'
    #                1: 'positive Covid-19 CT'
    #                2: 'negative Covid-19 CT'
    y_list=to_categorical(np.concatenate(np.array([[0]*len(target_list1),
                                               [1]*len(target_list2),
                                               [2]*len(target_list3)])),3)

    X=np.array([read_ct_img_bydir(file) for file in target_list])[:,:,:,np.newaxis]

    return X, y_list