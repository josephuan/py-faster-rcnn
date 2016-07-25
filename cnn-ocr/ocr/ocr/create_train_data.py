# -*- coding: utf-8 -*-
# 图像都是行优先的，访问图像都按照，(y,x)坐标来访问
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os
import codecs

from  math import pow
import skimage.io as io
from skimage import transform as tf

from cnn_ocr import text_detection_image, char2label, chars

import time
import datetime  
import random  

import Image,ImageDraw

from xml_jason_converter import pythonXmlToJson, pythonJsonToXml

import math

from labelFile import LabelFile, LabelFileError

import xmltodict

import cv2

photo_base_dir=u'D:/hwang/fridge-photos/'
#photo_base_dir=u'D:/冰箱图/'

model_path = '../../'
#model_path = './'

det_char_w = 32 #检测网络的input dim
recog_char_w = 114 #32 #114 #识别网络的input dim
recog_char_ext_ratio = 0.125 #1/8 扩展的比率

testing_samples_ratio = 0.1 #othres are training samples
val_samples_ratio = 0.1 #othres are training samples


def get_time_rand_num():
    nowTime=datetime.datetime.now().strftime("%Y%m%d%H%M%S");#生成当前时间  
    randomNum=random.randint(0,1000);#生成的随机整数n，其中0<=n<=1000 
    if randomNum<=100:  
        randomNum=str(0)+str(randomNum); 
    elif randomNum<=10: 
        randomNum=str(0)+str(0)+str(randomNum);   
    uniqueNum=str(nowTime)+str(randomNum);  
    return uniqueNum;  

#直接从图像集中建立testing samples list
def dir_to_det_trainimg_datas(listfile_filename, label):
       
    
     # listfile_filename所在目录
    save_directory = listfile_filename[:-len(listfile_filename.split('/')[-1])]

                
    file_object_train = codecs.open(save_directory+'detector_train_list.txt', 'w','utf-8')
    file_object_test = codecs.open(save_directory+'detector_test_list.txt', 'w','utf-8')

     # 载入detector训练文件列表        
    with open(listfile_filename) as f:
        gt_lines = [[str(x) for x in line.strip().split(' ')] for line in f]
       
    index_in_all = 0  
    for gt_line in gt_lines:
           
        # 处理空行的情况
        if len(gt_line)<1:
            continue  

        print save_directory+gt_line[0]  

        # 获取对应图像
        save_filename = save_directory+gt_line[0] 
        
         
        index_in_all=index_in_all+1
          
        if 0==index_in_all%(int(1/testing_samples_ratio)):
            file_object_test.write(save_filename[3:]+u' '+str(label)+u'\r\n')
        else:
            file_object_train.write(save_filename[3:]+u' '+str(label)+u'\r\n')
                    
                 
    #关闭文件
    file_object_train.close()     
    file_object_test.close()                    


    return 0

# 直接从图像集中建立training samples list
def mat_det_trainimg_data_prep_dir(photo_folder, listfile_filename):
       
    
     # listfile_filename所在目录,去掉d:/
    save_directory = listfile_filename[:-len(listfile_filename.split('/')[-1])]+u'det_samples/'

    if False==os.path.exists(save_directory):
       os.makedirs(save_directory) 
   
        
    file_object = codecs.open(save_directory+listfile_filename.split('/')[-1], 'w','utf-8')

     # 载入detector训练文件列表        
    with open(listfile_filename) as f:
        gt_lines = [[str(x) for x in line.strip().split(' ')] for line in f]
       
    for gt_line in gt_lines:
           
        # 处理空行的情况
        if len(gt_line)<2:
            continue  

        print photo_folder+gt_line[0]  

        # 新建字符对应目录，保存所有检测到的图像块
        save_directory1 = save_directory+gt_line[1]+u'/'
        if False==os.path.exists(save_directory1):
            os.makedirs(save_directory1)                        
            
        # 获取对应图像
        save_filename = save_directory1+gt_line[0][:-4]+u'_'+str(get_time_rand_num())+u'.jpg' 
        
        imgs = skimage.io.imread(photo_folder+gt_line[0],as_grey=False)
        skimage.io.imsave(save_filename,imgs)        
        
          
        file_object.write(save_filename[3:]+u' '+gt_line[1]+u'\r\n')
                    
                 
    #关闭文件
    file_object.close()                   


    return 0
    
# 直接从图像集中建立training samples list
def mat_recog_trainimg_data_prep_dir(photo_folder, listfile_filename):
       
    
     # listfile_filename所在目录,去掉d:/
    save_directory = listfile_filename[:-len(listfile_filename.split('/')[-1])]+u'recog_samples/'

    if False==os.path.exists(save_directory):
       os.makedirs(save_directory) 
   
        
    file_object = codecs.open(save_directory+listfile_filename.split('/')[-1], 'w','utf-8')

     # 载入detector训练文件列表        
    with open(listfile_filename) as f:
        gt_lines = [[str(x) for x in line.strip().split(' ')] for line in f]
       
    for gt_line in gt_lines:
           
        # 处理空行的情况
        if len(gt_line)<2:
            continue                    
            
        print photo_folder+gt_line[0]
        
        # 新建字符对应目录，保存所有检测到的图像块
        save_directory1 = save_directory+gt_line[1]+u'/'
        if False==os.path.exists(save_directory1):
            os.makedirs(save_directory1)               
       
        
        # 获取对应图像
        save_filename = save_directory1+gt_line[0][:-4]+u'_'+str(get_time_rand_num())+u'.jpg' 
        
        imgs = skimage.io.imread(photo_folder+gt_line[0],as_grey=False)
        skimage.io.imsave(save_filename,imgs)
        
        
          
        file_object.write(save_filename[3:]+u' '+gt_line[1]+u'\r\n')
                    
                 
    #关闭文件
    file_object.close()                   


    return 0

#还需要增加1/N份作为testing样本的功能    
def det_data_prep_dir(photo_folder, mark_folder):
     
    #用来存储宽度和高度序列，用来制作直方图
    all_box_width=[]
    all_box_height=[]
    
     #写入训练图像文件的列表，文件名 类别号
    save_directory = mark_folder+u'det_samples/'
    if False==os.path.exists(save_directory):
        os.makedirs(save_directory)         
        
    file_object_train = codecs.open(save_directory+'detector_train_list.txt', 'w','utf-8')
    file_object_test = codecs.open(save_directory+'detector_test_list.txt', 'w','utf-8')

     # 载入gt标记文件列表
    gt_filelist = mark_folder+u'filelist.txt'
    try:
        gt_filenames = np.loadtxt(gt_filelist, str, delimiter='\t')
    except:   
        gt_filenames = np.loadtxt(gt_filelist, str, delimiter='\t')
       
    index_in_all = 0
    for gt_filename in gt_filenames:
        photo_filename = photo_folder+gt_filename[:-3]+u'.jpg'
        print photo_filename
        
        # load mark files,获取图像中box的列表
        imgs = skimage.io.imread(photo_filename,as_grey=False)
        
        with open(mark_folder+gt_filename) as f:
            gt_lines = [[str(x) for x in line.strip().split('\t')] for line in f]
       
        for gt_line in gt_lines:
           
            # 处理空行的情况
            if len(gt_line)<2:
                continue

            box=gt_line[2:6]
        
            all_box_height.append(int(box[3]))
            all_box_width.append(int(box[2]))
        
            # 新建字符对应目录，保存所有检测到的图像块
            save_directory = mark_folder+u'det_samples/'+str(char2label(unicode(gt_line[6].strip(),"utf-8")))+u'/'
#            save_directory = mark_folder+u'det_samples/'+unicode(gt_line[6].strip(),"utf-8")+u'/'            
            
            if False==os.path.exists(save_directory):
                os.makedirs(save_directory)                  
                            
            im_crop = imgs[int(box[1]):int(box[1])+int(box[3]),int(box[0]):int(box[0])+int(box[2])]

            scaled_im_crop= tf.resize(im_crop,(det_char_w,det_char_w))            
            
            save_filename = save_directory+photo_filename.split('/')[-1][:-4]+u'_crop'+str(get_time_rand_num())+u'.jpg'           
            
#            skimage.io.imsave(save_filename,scaled_im_crop)
             # 保留原始尺寸，在create——data，生成图像时统一resize
            skimage.io.imsave(save_filename,im_crop)
            
            index_in_all=index_in_all+1
                    
            if 0==index_in_all%(int(1/testing_samples_ratio)):
                file_object_test.write(save_filename[3:]+u' 0\r\n')
            else:
                file_object_train.write(save_filename[3:]+u' 0\r\n')
                 
                    
                 
    #关闭文件
    file_object_train.close()     
    file_object_test.close()       
                 
#    num_bins = 50
#    plt.hist(all_box_width, num_bins, normed=1, facecolor='green', alpha=0.5)
#    plt.hist(all_box_height, num_bins, normed=1, facecolor='blue', alpha=0.5)
#    plt.title('Histogram')
#    plt.show()

    return 0
    
def recog_data_prep_dir(photo_folder, mark_folder):
     
    #用来存储宽度和高度序列，用来制作直方图
    all_box_width=[]
    all_box_height=[]
    
     #写入训练图像文件的列表，文件名 类别号
    save_directory = mark_folder+u'recog_samples/'
    if False==os.path.exists(save_directory):
        os.makedirs(save_directory) 
    file_object_train = codecs.open(save_directory+'classifier_train_list.txt', 'w','utf-8')
    file_object_test = codecs.open(save_directory+'classifier_test_list.txt', 'w','utf-8')


     # 载入gt标记文件列表
    gt_filelist = mark_folder+u'filelist.txt'
    try:
        gt_filenames = np.loadtxt(gt_filelist, str, delimiter='\t')
    except:   
        gt_filenames = np.loadtxt(gt_filelist, str, delimiter='\t')
       
    index_in_all = 0
    for gt_filename in gt_filenames:
        photo_filename = photo_folder+gt_filename[:-3]+u'.jpg'
        print photo_filename
        
        # load mark files,获取图像中box的列表
        imgs = skimage.io.imread(photo_filename,as_grey=False)
        
        with open(mark_folder+gt_filename) as f:
            gt_lines = [[str(x) for x in line.strip().split('\t')] for line in f]
       
        for gt_line in gt_lines:
           
            # 处理空行的情况
            if len(gt_line)<2:
                continue

            box=gt_line[2:6]
        
            all_box_height.append(int(box[3]))
            all_box_width.append(int(box[2]))
        
            # 新建字符对应目录，保存所有检测到的图像块
            save_directory = mark_folder+u'recog_samples/'+str(char2label(unicode(gt_line[6].strip(),"utf-8")))+u'/'
            if False==os.path.exists(save_directory):
                os.makedirs(save_directory)                  
                   
            recog_ext_offset_h = int(int(box[3])*recog_char_ext_ratio)
            recog_ext_offset_w = int(int(box[2])*recog_char_ext_ratio)
            
            if recog_ext_offset_h < 1:
                recog_ext_offset_h = 1
            if recog_ext_offset_w < 1:
                recog_ext_offset_w = 1
                
            recog_ext_stride_h = int(int(box[3])/recog_char_w)
            recog_ext_stride_w = int(int(box[2])/recog_char_w)
            
            if recog_ext_stride_h < 1:
                recog_ext_stride_h = 1
            if recog_ext_stride_w < 1:
                recog_ext_stride_w = 1
            
            top0 = int(box[1]) - recog_ext_offset_h
            left0 = int(box[0]) - recog_ext_offset_w
            bottom0 = int(box[1])+int(box[3]) + recog_ext_offset_h
            right0 = int(box[0])+int(box[2]) + recog_ext_offset_w

            if top0 < 0:
                top0 = 0
            if left0 < 0:
                left0 = 0
            if bottom0 > imgs.shape[0]:
                bottom0 = imgs.shape[0]
            if right0 > imgs.shape[1]:
                right0 = imgs.shape[1]
            
            im_crop = imgs[top0:bottom0,left0:right0]
            
            for y_slice in range(0,recog_ext_offset_h*2+1,recog_ext_stride_h):
                for x_slice in range(0,recog_ext_offset_w*2+1,recog_ext_stride_w):
                    # 裁剪图片
                    blob_img=im_crop[y_slice:(y_slice+int(box[3])), x_slice:(x_slice+int(box[2]))]
                    
                    scaled_blob_img= tf.resize(blob_img,(recog_char_w,recog_char_w))     
                
                    save_filename = save_directory+photo_filename.split('/')[-1][:-4]+u'_crop'+str(get_time_rand_num())+u'.jpg'           
            
#                    skimage.io.imsave(save_filename,scaled_blob_img) 
                    # 保留原始尺寸，在create——data，生成图像时统一resize
                    skimage.io.imsave(save_filename,blob_img) 
                    
                   
                    
                    index_in_all=index_in_all+1
                    
                    if 0==index_in_all%(int(1/testing_samples_ratio)):
                        file_object_test.write(save_filename[3:]+u' '+str(char2label(unicode(gt_line[6].strip(),"utf-8")))+u'\r\n')
                    else:
                        file_object_train.write(save_filename[3:]+u' '+str(char2label(unicode(gt_line[6].strip(),"utf-8")))+u'\r\n')
                 
    #关闭文件
    file_object_train.close()     
    file_object_test.close()     
    
#    num_bins = 50
#    plt.hist(all_box_width, num_bins, normed=1, facecolor='green', alpha=0.5)
#    plt.hist(all_box_height, num_bins, normed=1, facecolor='blue', alpha=0.5)
#    plt.title('Histogram')
#    plt.show()

    return 0
    
# 对于有obj_mark文件的处理情况    
# gt_mark_folder: 标记的文字的位置
# obj_mark_folder: 标记的物体框的位置
# overlapThresh: cnn detect box和gt标记的框重合门限，如果大于，就不作为夫样本
def neg_trainimg_prep_dir_objmark(photo_folder, gt_mark_folder, obj_mark_folder,overlapThresh):

 #写入训练图像文件的列表，文件名 类别号
    save_directory = gt_mark_folder+u'neg_det_samples/'
    if False==os.path.exists(save_directory):
        os.makedirs(save_directory) 
        
    if False==os.path.exists(save_directory+u'1/'):
        os.makedirs(save_directory+u'1/') 
        
    file_object_train = codecs.open(save_directory+'detector_train_list.txt', 'w','utf-8')
    file_object_test = codecs.open(save_directory+'detector_test_list.txt', 'w','utf-8')


    # load目标类别列表
    type_filelist = obj_mark_folder+u'filelist.txt'
    try:
       type_names = np.loadtxt(type_filelist, str, delimiter='\t')
    except:   
       type_names = np.loadtxt(type_filelist, str, delimiter='\t')

    index_in_all=0
    # load 每一类的图像列表
    for type_name in type_names:
        mark_filelist = obj_mark_folder+unicode(type_name,"utf-8")+u'/filelist.txt'
        try:
           mark_filenames = np.loadtxt(mark_filelist, str, delimiter='\t')
        except:   
           mark_filenames = np.loadtxt(mark_filelist, str, delimiter='\t')

        for mark_filename in mark_filenames:
            photo_filename = photo_folder+mark_filename[:-4]
            print photo_filename
            
            gt_filename = mark_filename[:-8]+u'.gt'
            
            # photo not exist
            if False==os.path.exists(gt_mark_folder+gt_filename):
                continue
            
            with open(gt_mark_folder+gt_filename) as f:
                gt_lines = [[str(x) for x in line.strip().split('\t')] for line in f]
                
            gt_boxes=[]
            for gt_line in gt_lines:           
                # 处理空行的情况
                if len(gt_line)<2:
                    continue
                box=gt_line[2:6]
                gt_boxes.append(box)
                
            # 计算各个字符窗口的面积
            # 获取窗口的坐标，是一个向量
            gt_boxes_x1 = [[int(box[0]) for box in gt_boxes]]
            gt_boxes_y1 = [[int(box[1]) for box in gt_boxes]]
            gt_boxes_w = [[int(box[2]) for box in gt_boxes]]
            gt_boxes_h = [[int(box[3]) for box in gt_boxes]]
            
            gt_boxes_x2 = [[int(gt_boxes_x1[0][i]+gt_boxes_w[0][i]) for i in range(len(gt_boxes_w[0]))]]
            gt_boxes_y2 = [[int(gt_boxes_y1[0][i]+gt_boxes_h[0][i]) for i in range(len(gt_boxes_h[0]))]]
            
    
            
            area = [[int(gt_boxes_w[0][i]*gt_boxes_h[0][i]) for i in range(len(gt_boxes_w[0]))]]

            mark_filename = obj_mark_folder+unicode(type_name,"utf-8")+u'/'+mark_filename
            
            # load mark files,获取图像中box的列表
            obj_box=[]
            with open(mark_filename) as f:    
                array = [[int(x) for x in line.split()] for line in f]
                
            # mark is empty line
            if len(array)<1:
                continue
            
            if len(array[0])<4:
                continue
                
            for i in range(len(array[0])/4):
                box=[]
                for j in range(4):
                   box.append(array[0][i*4+j])
                obj_box.append(box)
             
            # photo not exist
            if False==os.path.exists(photo_filename):
                continue
            
            imgs = skimage.io.imread(photo_filename,as_grey=False)            
           

            for i in range(0,len(obj_box)):
                 box=obj_box[i]      

                 # right小于left或者bottom小于top
                 if box[2] <box[0] or box[3] <box[1]:
                     continue
                 
                 im_crop = imgs[box[1]:box[3],box[0]:box[2]]
                 save_filename = save_directory+unicode(type_name,"utf-8")+u'_'+photo_filename.split('/')[-1][:-4]+u'_crop'+str(i)+u'.jpg'

                 skimage.io.imsave(save_filename,im_crop)                 

                 detected_char_boxes = text_detection_image(save_filename)

                 rows,cols,chs = im_crop.shape

                 # 新建目录，保存所有检测到的图像块
                 for box in detected_char_boxes:                    
                    # 计算和gt里面char box的重合区域，如果重合大于一定比率就不保存为夫样本
                    #找出窗口的最大的坐标位置，以及最小的坐标位置。
                    xx1 = np.maximum(box[1]+obj_box[i][0], gt_boxes_x1[0])
                    yy1 = np.maximum(box[0]+obj_box[i][1], gt_boxes_y1[0])
                    xx2 = np.minimum(box[3]+obj_box[i][0], gt_boxes_x2[0])
                    yy2 = np.minimum(box[2]+obj_box[i][1], gt_boxes_y2[0])
                     
                    #计算窗口的长和宽
                    w = np.maximum(0, xx2 - xx1 + 1)
                    h = np.maximum(0, yy2 - yy1 + 1)
                     
                    #计算重合率 ,占gt_box的面积比率                   
#                    overlap = (w * h) / (area)
#                    overlap_idxs = np.where(overlap > overlapThresh)[1]
                    
                    #计算重合率，占detected_char_boxe的面积比率，
                    #适合输入gt_box为整个text的区域
                    overlap = (w * h) / ((box[3]-box[1])*(box[2]-box[0]))
                    overlap_idxs = np.where(overlap > overlapThresh)[0]           
                    
                    
                    # 计算和gt里面char box的重合区域，如果重合大于一定比率就不保存为夫样本
                    if len(overlap_idxs)>0 :
                        continue                        
                    
                    if box[0] <0 or box[1] <0 or box[2] >=rows  or box[3] >=cols:
                        continue
                    im_crop_s = im_crop[box[0]:box[2],box[1]:box[3]]
                    if im_crop_s.shape[0] == 0 or im_crop_s.shape[1] == 0:
                        continue

#                    save_directory =photo_base_dir+u'neg_det_samples/'+photo_filename.split('/')[-1][:-4]+u'/'
#                    if False==os.path.exists(save_directory):
#                        os.makedirs(save_directory)     

#                    skimage.io.imsave(save_directory+'detcrop_img_'+str(get_time_rand_num())+'.jpg', im_crop_s)
                    
                    save_filename=save_directory+u'1/'+photo_filename.split('/')[-1][:-4]+'_detcrop_img_'+str(get_time_rand_num())+'.jpg'
                    
                    skimage.io.imsave(save_filename, im_crop_s)
    
                    index_in_all=index_in_all+1
                    
                    if 0==index_in_all%(int(1/testing_samples_ratio)):
                        file_object_test.write(save_filename[3:]+u' 1\r\n')
                    else:
                        file_object_train.write(save_filename[3:]+u' 1\r\n')
    #关闭文件
    file_object_train.close()     
    file_object_test.close()      
    
    return 0

   
# 对于有obj_mark文件的处理情况    
# obj_mark_folder: 标记的物体框的位置
# 保存所有在obj mark框中检测到文字块
def neg_trainimg_prep_dir_only_objmark(photo_folder, obj_mark_folder):

    #写入训练图像文件的列表，文件名 类别号
    save_directory = obj_mark_folder+u'neg_det_samples/'
    if False==os.path.exists(save_directory):
        os.makedirs(save_directory) 
        
    # load目标类别列表
    type_filelist = obj_mark_folder+u'filelist.txt'
    try:
       type_names = np.loadtxt(type_filelist, str, delimiter='\t')
    except:   
       type_names = np.loadtxt(type_filelist, str, delimiter='\t')


    # load 每一类的图像列表
    for type_name in type_names:
        mark_filelist = obj_mark_folder+unicode(type_name,"utf-8")+u'/filelist.txt'
        try:
           mark_filenames = np.loadtxt(mark_filelist, str, delimiter='\t')
        except:   
           mark_filenames = np.loadtxt(mark_filelist, str, delimiter='\t')

        for mark_filename in mark_filenames:
            photo_filename = photo_folder+mark_filename[:-4]
            print photo_filename
            
           
            mark_filename = obj_mark_folder+unicode(type_name,"utf-8")+u'/'+mark_filename
            
            # load mark files,获取图像中box的列表
            obj_box=[]
            with open(mark_filename) as f:    
                array = [[int(x) for x in line.split()] for line in f]
                
            # mark is empty line
            if len(array)<1:
                continue
            
            if len(array[0])<4:
                continue
                
            for i in range(len(array[0])/4):
                box=[]
                for j in range(4):
                   box.append(array[0][i*4+j])
                obj_box.append(box)
             
            # photo not exist
            if False==os.path.exists(photo_filename):
                continue
            
            imgs = skimage.io.imread(photo_filename,as_grey=False)            
           

            for i in range(0,len(obj_box)):
                 box=obj_box[i]      

                 # right小于left或者bottom小于top
                 if box[2] <box[0] or box[3] <box[1]:
                     continue
                 
                 im_crop = imgs[box[1]:box[3],box[0]:box[2]]
                 save_filename = save_directory+unicode(type_name,"utf-8")+u'_'+photo_filename.split('/')[-1][:-4]+u'_crop'+str(i)+u'.jpg'

                 skimage.io.imsave(save_filename,im_crop)                 

                 detected_char_boxes = text_detection_image(save_filename)

                 rows,cols,chs = im_crop.shape

                 # 新建目录，保存所有检测到的图像块
                 for box in detected_char_boxes:                    
                                         
                    
                    if box[0] <0 or box[1] <0 or box[2] >=rows  or box[3] >=cols:
                        continue
                    
                    im_crop_s = im_crop[box[0]:box[2],box[1]:box[3]]
                    if im_crop_s.shape[0] == 0 or im_crop_s.shape[1] == 0:
                        continue

                    # 每个文件单独文件夹保存crop图像
                    save_directory1 =save_directory+photo_filename.split('/')[-1][:-4]+u'/'
                    if False==os.path.exists(save_directory1):
                        os.makedirs(save_directory1)     

                    skimage.io.imsave(save_directory1+'detcrop_img_'+str(get_time_rand_num())+'.jpg', im_crop_s)
                    
                   
    
    return 0
    
# 暂时不用，对大图有问题，会检测出太多的text box，超过int_max    
# 对于没有obj_mark文件的处理情况    
# gt_mark_folder: 标记的文字的位置
# obj_mark_folder: 标记的物体框的位置
# overlapThresh: cnn detect box和gt标记的框重合门限，如果大于，就不作为夫样本
def neg_trainimg_prep_dir_no_objmark(photo_folder, mark_folder,overlapThresh):


     # 载入gt标记文件列表
    gt_filelist = mark_folder+u'filelist.txt'
    try:
        gt_filenames = np.loadtxt(gt_filelist, str, delimiter='\t')
    except:   
        gt_filenames = np.loadtxt(gt_filelist, str, delimiter='\t')
       
    for gt_filename in gt_filenames:
        photo_filename = photo_folder+gt_filename[:-3]+u'.jpg'
        print photo_filename
        
        # load mark files      
        with open(mark_folder+gt_filename) as f:
            gt_lines = [[str(x) for x in line.strip().split('\t')] for line in f]
       
       
        gt_boxes=[]
        for gt_line in gt_lines:
           
            # 处理空行的情况
            if len(gt_line)<2:
                continue

            box=gt_line[2:6]
            gt_boxes.append(box)
        

        imgs = skimage.io.imread(photo_filename,as_grey=False)
        detected_char_boxes = text_detection_image(photo_filename)

        rows,cols,chs = imgs.shape

        # 计算各个字符窗口的面积
        # 获取窗口的坐标，是一个向量
        x1 = gt_boxes[:][0]
        y1 = gt_boxes[:][1]
        x2 = gt_boxes[:][2]
        y2 = gt_boxes[:][3]
    
        area = (x2 - x1 + 1) * (y2 - y1 + 1)

        # 新建目录，保存所有检测到的图像块
        for box in detected_char_boxes:
           
                    
            # 计算和gt里面char box的重合区域，如果重合大于一定比率就不保存为夫样本
            #找出窗口的最大的坐标位置，以及最小的坐标位置。
            xx1 = np.maximum(box[1], gt_boxes[:,0])
            yy1 = np.maximum(box[0], gt_boxes[:,1])
            xx2 = np.minimum(box[3], gt_boxes[:,2])
            yy2 = np.minimum(box[2], gt_boxes[:,3])
                     
            #计算窗口的长和宽
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
                     
            #计算重合率
            overlap = (w * h) / (area)
            overlap_idxs = np.where(overlap > overlapThresh)[0]
                    
            # 计算和gt里面char box的重合区域，如果重合大于一定比率就不保存为夫样本
            if len(overlap_idxs)>0 :
                continue                        
                    
            if box[0] <0 or box[1] <0 or box[2] >=rows  or box[3] >=cols:
                continue
            im_crop_s = imgs[box[0]:box[2],box[1]:box[3]]
            if im_crop_s.shape[0] == 0 or im_crop_s.shape[1] == 0:
                continue

#           save_directory =photo_base_dir+u'neg_det_samples/'+photo_filename.split('/')[-1][:-4]+u'/'
#           if False==os.path.exists(save_directory):
#               os.makedirs(save_directory)     

#           skimage.io.imsave(save_directory+'detcrop_img_'+str(get_time_rand_num())+'.jpg', im_crop_s)
                    
                 
            if False==os.path.exists(photo_base_dir+u'neg_det_samples/'):
                os.makedirs(photo_base_dir+u'neg_det_samples/')  
                
            skimage.io.imsave(photo_base_dir+u'neg_det_samples/'+photo_filename.split('/')[-1][:-4]+'_detcrop_img_'+str(get_time_rand_num())+'.jpg', im_crop_s)
           
    return 0
    
def draw_box_with_objmarks(photo_folder, gt_mark_folder, obj_mark_folder):
    
    #写入训练图像文件的列表，文件名 类别号
    save_directory = gt_mark_folder+u'det_samples_gt/'
    if False==os.path.exists(save_directory):
        os.makedirs(save_directory) 
           
    # load目标类别列表
    type_filelist = obj_mark_folder+u'filelist.txt'
    try:
       type_names = np.loadtxt(type_filelist, str, delimiter='\t')
    except:   
       type_names = np.loadtxt(type_filelist, str, delimiter='\t')
  
    # load 每一类的图像列表
    for type_name in type_names:
        mark_filelist = obj_mark_folder+unicode(type_name,"utf-8")+u'/filelist.txt'
        try:
           mark_filenames = np.loadtxt(mark_filelist, str, delimiter='\t')
        except:   
           mark_filenames = np.loadtxt(mark_filelist, str, delimiter='\t')

        for mark_filename in mark_filenames:
            mark_filename_inst=mark_filename         
            
            photo_filename = photo_folder+mark_filename[:-4]
            print photo_filename
            
             # photo not exist
            if False==os.path.exists(photo_filename):
                continue
            
            # photo not exist
            photo_filename1=save_directory+mark_filename[:-4]
            if False==os.path.exists(photo_filename1):
                imgs = io.imread(photo_filename,as_grey=False)
                io.imsave(photo_filename1,imgs) 
              
            im_crop = io.imread(photo_filename,as_grey=False)
            imgs = Image.open(photo_filename1)
            draw = ImageDraw.Draw(imgs)
            
            mark_filename = obj_mark_folder+unicode(type_name,"utf-8")+u'/'+mark_filename
            
            # load mark files,获取图像中box的列表
            obj_box=[]
            with open(mark_filename) as f:    
                array = [[int(x) for x in line.split()] for line in f]
                
                
            # mark is empty line
            if len(array)<1:
                continue
            
            if len(array[0])<4:
                continue
                
            for i in range(len(array[0])/4):
                box=[]
                for j in range(4):
                   box.append(array[0][i*4+j])
                obj_box.append(box)
            
            for i in range(0,len(obj_box)):
                 box=obj_box[i]      

                 left=box[0]
                 top=box[1]
                 right=box[2]
                 bottom=box[3] 
                 # right小于left或者bottom小于top, 调换一下
                 if box[2] <box[0] or box[3] <box[1]:
                     left=min(box[0],box[2])   
                     right=max(box[0],box[2]) 
                     top=min(box[1],box[3]) 
                     bottom=max(box[1],box[3]) 
    
                     box[0]=left
                     box[1]=top
                     box[2]=right
                     box[3]=bottom 
                
                 draw.rectangle(box,outline=(0,255,0))
                 imgs.save(photo_filename1)  
    
#                 if False==os.path.exists(gt_mark_folder+mark_filename_inst[:-4]+'.lif'):
#                     continue
#                 
#                 # 载入 char lif file
#                 charbox_str=''
#                 char_labelFile = LabelFile(gt_mark_folder+mark_filename_inst[:-4]+'.lif')
#                 for char_shape in char_labelFile.shapes: 
#                
#                    p_x=[p[0] for p in char_shape[1]]
#                    p_y=[p[1] for p in char_shape[1]]  
#              
#                    # coordination of character
#                    left1 = int(np.min(p_x))
#                    top1 = int(np.min(p_y))
#            
#                    right1 = int(np.max(p_x))
#                    bottom1 = int(np.max(p_y))   
#               
#                    area = (right1 - left1 + 1) * (bottom1 - top1 + 1)
#                        
#                    # 计算和gt里面char box的重合区域，如果完全重合就认为是这个out box里的字
#                    #找出窗口的最大的坐标位置，以及最小的坐标位置。
#                    xx1 = np.maximum(left, left1)
#                    yy1 = np.maximum(top, top1)
#                    xx2 = np.minimum(right, right1)
#                    yy2 = np.minimum(bottom, bottom1)
#                         
#                    #计算窗口的长和宽
#                    w = np.maximum(0, xx2 - xx1 + 1)
#                    h = np.maximum(0, yy2 - yy1 + 1)
#                         
#                    #计算重合率
#                    overlap = (w * h*1.0) / (area)
#                    
#                    
#                    #print overlap        
#                    # 完全重合就认为是这个obj box里的字
#                    if overlap == 1.0 :
#                        charbox_str+=(str(left1-left)+' '+str(top1-top)+' '+str(right1-left)+' '+str(bottom1-top)+'\r\n')
#                 
#                 # obj box里没有文字就跳过
#                 if len(charbox_str)<2:
#                     continue
#                 
#                 # 截取object box图像    
#                 im_crop_s = im_crop[box[1]:box[3],box[0]:box[2]]
#                 if im_crop_s.shape[0] == 0 or im_crop_s.shape[1] == 0:
#                     continue                 
#
#                 io.imsave(photo_filename1[:-4]+'_'+str(i)+'.jpg', im_crop_s)
#                 
#                 #write to file
#                 charbox_filename= photo_filename1[:-4]+'_'+str(i)+'.txt'
#                 charbox_file_object = codecs.open(charbox_filename, 'a','utf-8')    
#                
#                 charbox_file_object.write(charbox_str)
#                
#                 charbox_file_object.close()     
    '''                 
    # 载入gt标记文件列表
    gt_filelist = gt_mark_folder+u'filelist.txt'
    try:
        gt_filenames = np.loadtxt(gt_filelist, str, delimiter='\t')
    except:   
        gt_filenames = np.loadtxt(gt_filelist, str, delimiter='\t')
       

    for gt_filename in gt_filenames:
               
       photo_filename = photo_folder+gt_filename[:-3]+u'.jpg'
       print photo_filename
            
       # photo not exist
       if False==os.path.exists(photo_filename):
            continue
            
        # photo not exist
       photo_filename1=save_directory+gt_filename[:-3]+u'.jpg'
       if False==os.path.exists(photo_filename1):
           imgs = skimage.io.imread(photo_filename,as_grey=False)
           skimage.io.imsave(photo_filename1,imgs) 
                
       imgs = Image.open(photo_filename1)
       draw = ImageDraw.Draw(imgs)
            
       with open(gt_mark_folder+gt_filename) as f:
           gt_lines = [[str(x) for x in line.strip().split('\t')] for line in f]
       
       gt_boxes=[]
       
       for gt_line in gt_lines:
           
           # 处理空行的情况
           if len(gt_line)<2:
               continue   
           
           box=gt_line[2:6]
           gt_boxes_x1 = int(box[0])
           gt_boxes_y1 =int(box[1])
            
           gt_boxes_x2 = int(box[0])+int(box[2])
           gt_boxes_y2 = int(box[1])+int(box[3])
           
           box_true=[]
           box_true.append(gt_boxes_x1);
           box_true.append(gt_boxes_y1);
           box_true.append(gt_boxes_x2);
           box_true.append(gt_boxes_y2);
           
           gt_boxes.append(box_true)
            
       for i in range(0,len(gt_boxes)):
            box=gt_boxes[i]      

            # right小于left或者bottom小于top
            if box[2] <box[0] or box[3] <box[1]:
               continue   
                
            draw.rectangle(box,outline="rgb(0,255,0)")
            imgs.save(photo_filename1)  
        '''
    
    return 0
    
def find_all_chinese_chars(in_char_list, gt_mark_folder):


    out_char_list=in_char_list
    
     # 载入gt标记文件列表
    gt_filelist = gt_mark_folder+u'filelist.txt'
    try:
        gt_filenames = np.loadtxt(gt_filelist, str, delimiter='\t')
    except:   
        gt_filenames = np.loadtxt(gt_filelist, str, delimiter='\t')
       
    allcharnames=[]
    for gt_filename in gt_filenames:
        
        print gt_filename
               
        with open(gt_mark_folder+gt_filename) as f:
            gt_lines = [[str(x) for x in line.strip().split('\t')] for line in f]
       
        for gt_line in gt_lines:
           
            # 处理空行的情况
            if len(gt_line)<2:
                continue
            
            allcharnames.append(unicode(gt_line[6],"utf-8"))
            


    for chn_char in allcharnames:
       
            if len(out_char_list)<1:
                out_char_list=out_char_list+chn_char
  
            find_same_char=False
            for list_char in out_char_list:
                if chn_char==list_char:
                    find_same_char=True
                    break             
            
            if False==find_same_char:
                out_char_list=out_char_list+chn_char
                
#    file_object = codecs.open(gt_mark_folder+'out_char_list.txt', 'w','utf-8')
#    
#    for x in out_char_list:    
#        file_object.write(x+'\r\n')
#    
#    file_object.close()    
#
#
#    file_object = codecs.open('out_char_list.txt', 'w','utf-8')
#    
#    ichar=0
#    for x in chars:    
#        file_object.write(str(ichar)+u'\r\n')
#        ichar=ichar+1
#    
#    file_object.close() 
            
    return out_char_list
    
def find_all_chinese_words(gt_mark_folder, lexword_file):

    # load lexicon_word   
#    try:
#       lexicon_words_list = np.loadtxt(lexword_file, str, delimiter='\t')
#    except:   
#       lexicon_words_list = np.loadtxt(lexword_file, str, delimiter='\t')
#
#    lexicon_words=[]
#    
#    for lex_word in lexicon_words_list:
#        lexicon_words.append(unicode(lex_word,'utf-8'))
   
#     # 载入gt标记文件列表
#    gt_filelist = gt_mark_folder+u'filelist.txt'
#    try:
#        gt_filenames = np.loadtxt(gt_filelist, str, delimiter='\t')
#    except:   
#        gt_filenames = np.loadtxt(gt_filelist, str, delimiter='\t')
#       
#    allwords=[]
#   
#    for gt_filename in gt_filenames:
#        
#        print gt_filename
#               
#        with open(gt_mark_folder+gt_filename) as f:
#            gt_lines = [[str(x) for x in line.strip().split('\t')] for line in f]
#        
#        allcharnames=[]
#        for gt_line in gt_lines:
#           
#            # 处理空行的情况
#            if len(gt_line)<2:
#                continue
#            
#            allcharnames.append(unicode(gt_line[6],"utf-8"))
#            
#        if len(allcharnames)>1:
#            allwords.append(allcharnames)         
#
#    for word in allwords:
#       
#            if len(lexicon_words)<1:
#                lexicon_words.append(''.join(word)) 
#  
#            find_same_word=False
#            for lex_word in lexicon_words:
#                if lex_word==''.join(word):
#                    find_same_word=True
#                    break             
#            
#            if False==find_same_word:
#                lexicon_words.append(''.join(word))    

#    for word in lexicon_words:
#       
#            rev_word= word[::-1]
#  
#            find_same_word=False
#            for lex_word in lexicon_words:
#                if lex_word==rev_word:
#                    find_same_word=True
#                    break             
#            
#            if False==find_same_word:
#                lexicon_words.append(rev_word)   

    with open(lexword_file) as f:
        lexword_lines = [[unicode(str(x),'utf-8') for x in line.strip().split('\t')] for line in f]

    # 去掉words中重复的
    allwords=[]
    for word in lexword_lines:           
  
            find_same_word=False
            for lex_word in allwords:
                if lex_word[0]==word[0] or lex_word[0][::-1]==word[0]:
                    find_same_word=True
                    break                               
            
            if False==find_same_word:
                allwords.append(word)   
                
    file_object = codecs.open('out_word_list.txt', 'w','utf-8')
    
    for word in allwords:    
        file_object.write(word[0]+'\t'+word[1]+'\r\n')
        file_object.write(word[0][::-1]+'\t'+word[1]+'\r\n')
    
    file_object.close()    
            
    return allwords

# 以旋转后的textbox中心扩展到长边500的图像
def generate_MTR500_objs_photo(photo_folder,mark_folder):

    
    min_ext_box_size = 224
    
    #写入训练图像文件的列表，文件名 类别号
    save_directory = photo_folder+u'ext_samples_gt/'
    if False==os.path.exists(save_directory):
        os.makedirs(save_directory) 
    
     # 载入gt标记文件列表
    gt_filelist = mark_folder+u'filelist.txt'
    try:
        gt_filenames = np.loadtxt(gt_filelist, str, delimiter='\t')
    except:   
        gt_filenames = np.loadtxt(gt_filelist, str, delimiter='\t')
       
       
   
    for gt_filename in gt_filenames:
        photo_filename = photo_folder+gt_filename[:-3]+u'.jpg'
        print photo_filename
        
        # load mark files,获取图像中box的列表
        imgs = skimage.io.imread(photo_filename,as_grey=False)
        imw=imgs.shape[1]
        imh=imgs.shape[0]
        
        boxs = []
        with open(mark_folder+gt_filename) as f:
            gt_lines = [[str(x) for x in line.strip().split(' ')] for line in f]
       
        for gt_line in gt_lines:
           
            # 处理空行的情况
            if len(gt_line)<2:
                continue

            box=gt_line[0:7]
            boxs.append(box)
       
        for box in boxs:
            
            ext_box_size = 500   
            
            
            #计算窗口的长和宽
            left = float(box[2])
            top = float(box[3])

            right = float(box[2])+float(box[4])
            bottom = float(box[3])+float(box[5])
            
            box_o_x = float(box[2])+float(box[4])*0.5
            box_o_y =  float(box[3])+float(box[5])*0.5
            
            rot_theta = float(box[6])
            
            cosA = math.cos(rot_theta)
            sinA = math.sin(rot_theta)
            
            # 绕中心旋转方框
            o_rot_lefttop_x = int(cosA*left-sinA*top-box_o_x*cosA+box_o_y*sinA+box_o_x)
            o_rot_lefttop_y = int(sinA*left+cosA*top-box_o_x*sinA-box_o_y*cosA+box_o_y)
            
            o_rot_leftbottom_x = int(cosA*left-sinA*bottom-box_o_x*cosA+box_o_y*sinA+box_o_x)
            o_rot_leftbottom_y = int(sinA*left+cosA*bottom-box_o_x*sinA-box_o_y*cosA+box_o_y)
            
            o_rot_righttop_x = int(cosA*right-sinA*top-box_o_x*cosA+box_o_y*sinA+box_o_x)
            o_rot_righttop_y = int(sinA*right+cosA*top-box_o_x*sinA-box_o_y*cosA+box_o_y)
            
            o_rot_rightbottom_x = int(cosA*right-sinA*bottom-box_o_x*cosA+box_o_y*sinA+box_o_x)
            o_rot_rightbottom_y = int(sinA*right+cosA*bottom-box_o_x*sinA-box_o_y*cosA+box_o_y)

            
            if o_rot_lefttop_x<0:
                o_rot_lefttop_x=0
            if o_rot_leftbottom_x<0:
                o_rot_leftbottom_x=0
            if o_rot_righttop_x<0:
                o_rot_righttop_x=0
            if o_rot_rightbottom_x<0:
                o_rot_rightbottom_x=0
                
            if o_rot_lefttop_y<0:
                o_rot_lefttop_y=0
            if o_rot_leftbottom_y<0:
                o_rot_leftbottom_y=0
            if o_rot_righttop_y<0:
                o_rot_righttop_y=0
            if o_rot_rightbottom_y<0:
                o_rot_rightbottom_y=0
                
            if o_rot_lefttop_x>imw-1:
                o_rot_lefttop_x=imw-1
            if o_rot_leftbottom_x>imw-1:
                o_rot_leftbottom_x=imw-1
            if o_rot_righttop_x>imw-1:
                o_rot_righttop_x=imw-1
            if o_rot_rightbottom_x>imw-1:
                o_rot_rightbottom_x=imw-1
                
            if o_rot_lefttop_y>imh-1:
                o_rot_lefttop_y=imh-1
            if o_rot_leftbottom_y>imh-1:
                o_rot_leftbottom_y=imh-1
            if o_rot_righttop_y>imh-1:
                o_rot_righttop_y=imh-1
            if o_rot_rightbottom_y>imh-1:
                o_rot_rightbottom_y=imh-1
                          
            
            p_x=[o_rot_lefttop_x,o_rot_leftbottom_x,o_rot_righttop_x,o_rot_rightbottom_x]
            p_y=[o_rot_lefttop_y,o_rot_leftbottom_y,o_rot_righttop_y,o_rot_rightbottom_y]
            
            
            
            #计算窗口的长和宽
            left = np.min(p_x)
            top = np.min(p_y)

            right = np.max(p_x)
            bottom = np.max(p_y)
            
            # 扩展后窗口的中心点坐标为
            o_x = (left+right)/2
            o_y = (top+bottom)/2
            
            ext_box_size=max(ext_box_size,bottom-top+1)
            ext_box_size=max(ext_box_size,right-left+1)
            
            if bottom-top>right-left:
                oro_y=o_y-ext_box_size/2
                oro_x=o_x-ext_box_size*(right-left)/(bottom-top)/2
                # 右下角
                ere_y=oro_y+ext_box_size-1
                ere_x=oro_x+ext_box_size*(right-left)/(bottom-top)-1
                
            else:
                oro_x=o_x-ext_box_size/2
                oro_y=o_y-ext_box_size*(bottom-top)/(right-left)/2
                # 右下角
                ere_x=oro_x+ext_box_size-1
                ere_y=oro_y+ext_box_size*(bottom-top)/(right-left)-1
                
            if oro_x<0:
                oro_x=0
            if oro_y<0:
                oro_y=0
            if ere_x<0:
                ere_x=0
            if ere_y<0:
                ere_y=0
                
            if oro_y>imh-1:
                oro_y=imh-1
            if oro_x>imw-1:
                oro_x=imw-1
            if ere_y>imh-1:
                ere_y=imh-1
            if ere_x>imw-1:
                ere_x=imw-1
                
                
            newimw=ere_x-oro_x+1
            newimh=ere_y-oro_y+1
            
#            if int(newimw) <min_ext_box_size or  int(newimh)<min_ext_box_size:
#                continue
                
             # 保存扩大后的图片和box gt文件            
            crop_photo_filename = save_directory+gt_filename[:-3]+'_crop_'+box[0]+u'.jpg'

            im_crop = imgs[int(oro_y):int(ere_y)+1,int(oro_x):int(ere_x)+1]          
            skimage.io.imsave(crop_photo_filename,im_crop)
            
            crop_gt_filename = save_directory+gt_filename[:-3]+'_crop_'+box[0]+u'.gt'
            gt_file_object = codecs.open(crop_gt_filename, 'w','utf-8')
    
        
            if im_crop.shape[0]>0 and im_crop.shape[1]>0:  

                # 计算得到了子图的左上角坐标和宽高，计算所有box在子图中的位置
                indexbox=0                
                for box in boxs:                
                
                    #计算窗口的长和宽
                    left = float(box[2])
                    top = float(box[3])
        
                    right = float(box[2])+float(box[4])
                    bottom = float(box[3])+float(box[5])
                    
                    box_o_x = float(box[2])+float(box[4])*0.5
                    box_o_y =  float(box[3])+float(box[5])*0.5
                    
                    rot_theta = float(box[6])
                    
                    cosA = math.cos(rot_theta)
                    sinA = math.sin(rot_theta)
                    
                    # 绕中心旋转方框
                    o_rot_lefttop_x = int(cosA*left-sinA*top-box_o_x*cosA+box_o_y*sinA+box_o_x)
                    o_rot_lefttop_y = int(sinA*left+cosA*top-box_o_x*sinA-box_o_y*cosA+box_o_y)
                    
                    o_rot_leftbottom_x = int(cosA*left-sinA*bottom-box_o_x*cosA+box_o_y*sinA+box_o_x)
                    o_rot_leftbottom_y = int(sinA*left+cosA*bottom-box_o_x*sinA-box_o_y*cosA+box_o_y)
                    
                    o_rot_righttop_x = int(cosA*right-sinA*top-box_o_x*cosA+box_o_y*sinA+box_o_x)
                    o_rot_righttop_y = int(sinA*right+cosA*top-box_o_x*sinA-box_o_y*cosA+box_o_y)
                    
                    o_rot_rightbottom_x = int(cosA*right-sinA*bottom-box_o_x*cosA+box_o_y*sinA+box_o_x)
                    o_rot_rightbottom_y = int(sinA*right+cosA*bottom-box_o_x*sinA-box_o_y*cosA+box_o_y)
        
                    
                    if o_rot_lefttop_x<0:
                        o_rot_lefttop_x=0
                    if o_rot_leftbottom_x<0:
                        o_rot_leftbottom_x=0
                    if o_rot_righttop_x<0:
                        o_rot_righttop_x=0
                    if o_rot_rightbottom_x<0:
                        o_rot_rightbottom_x=0
                        
                    if o_rot_lefttop_y<0:
                        o_rot_lefttop_y=0
                    if o_rot_leftbottom_y<0:
                        o_rot_leftbottom_y=0
                    if o_rot_righttop_y<0:
                        o_rot_righttop_y=0
                    if o_rot_rightbottom_y<0:
                        o_rot_rightbottom_y=0
                        
                    if o_rot_lefttop_x>imw-1:
                        o_rot_lefttop_x=imw-1
                    if o_rot_leftbottom_x>imw-1:
                        o_rot_leftbottom_x=imw-1
                    if o_rot_righttop_x>imw-1:
                        o_rot_righttop_x=imw-1
                    if o_rot_rightbottom_x>imw-1:
                        o_rot_rightbottom_x=imw-1
                        
                    if o_rot_lefttop_y>imh-1:
                        o_rot_lefttop_y=imh-1
                    if o_rot_leftbottom_y>imh-1:
                        o_rot_leftbottom_y=imh-1
                    if o_rot_righttop_y>imh-1:
                        o_rot_righttop_y=imh-1
                    if o_rot_rightbottom_y>imh-1:
                        o_rot_rightbottom_y=imh-1
                                  
                    
                    p_x=[o_rot_lefttop_x,o_rot_leftbottom_x,o_rot_righttop_x,o_rot_rightbottom_x]
                    p_y=[o_rot_lefttop_y,o_rot_leftbottom_y,o_rot_righttop_y,o_rot_rightbottom_y]
                    
                    
                    
                    #计算窗口的长和宽
                    left = np.min(p_x)
                    top = np.min(p_y)
        
                    right = np.max(p_x)
                    bottom = np.max(p_y)
                
                    left = left-oro_x
                    top = top-oro_y

                    right = right-oro_x
                    bottom = bottom-oro_y  
                    
                    
            
            
                    if left<0:
                        left=0
                    if top<0:
                        top=0
                    if right<0:
                        right=0
                    if bottom<0:
                        bottom=0
                    
                    if top>newimh-1:
                        top=newimh-1
                    if left>newimw-1:
                        left=newimw-1
                    if bottom>newimh-1:
                        bottom=newimh-1
                    if right>newimw-1:
                        right=newimw-1    
            
                    if int(right)-int(left)+1>1 and  int(bottom)-int(top)+1>1:
     
                        gt_file_object.write(str(indexbox)+'\t'+box[1]+'\t')
                                       
                        gt_file_object.write(str(int(left))+u'\t'+str(int(top))+u'\t'+str(int(right)-int(left)+1)+u'\t'+str(int(bottom)-int(top)+1)+u'\t0.000000\r\n')
    
                        indexbox=indexbox+1
                    
            #关闭文件
            gt_file_object.close()
        
    a=1
        

#把数据的宽高比至少约束在0.117-15.500才能保证训练    
def generat_MTR500_objxmls(photo_folder,mark_folder):

    min_w_h_ratio=0.117
    max_w_h_ratio=15.500
    
    min_img_w_h_ratio=0.25
    max_img_w_h_ratio=4.0
    
    too_big_overlap=0.8

    # 保存到xml文件
    def make_json_dict(imgs,photofilename,imw,imh,boxs):
        source_dict=dict(database='The VOC2007 Database',
                 annotation='PASCAL VOC2007',
                 image='manual_collect',
                 flickrid='0000000')
        owner_dict=dict(flickrid='0000000',
                name='hwang')
        size_dict=dict(width=imw,
               height=imh,
               depth=3)   

        def format_object(box,imw,imh):          
        
            #计算窗口的长和宽
            left = float(box[2])
            top = float(box[3])

            right = float(box[2])+float(box[4])
            bottom = float(box[3])+float(box[5])
            
            box_o_x = float(box[2])+float(box[4])*0.5
            box_o_y =  float(box[3])+float(box[5])*0.5
            
            rot_theta = float(box[6])
            
            cosA = math.cos(rot_theta)
            sinA = math.sin(rot_theta)
            
            # 绕中心旋转方框
            o_rot_lefttop_x = int(cosA*left-sinA*top-box_o_x*cosA+box_o_y*sinA+box_o_x)
            o_rot_lefttop_y = int(sinA*left+cosA*top-box_o_x*sinA-box_o_y*cosA+box_o_y)
            
            o_rot_leftbottom_x = int(cosA*left-sinA*bottom-box_o_x*cosA+box_o_y*sinA+box_o_x)
            o_rot_leftbottom_y = int(sinA*left+cosA*bottom-box_o_x*sinA-box_o_y*cosA+box_o_y)
            
            o_rot_righttop_x = int(cosA*right-sinA*top-box_o_x*cosA+box_o_y*sinA+box_o_x)
            o_rot_righttop_y = int(sinA*right+cosA*top-box_o_x*sinA-box_o_y*cosA+box_o_y)
            
            o_rot_rightbottom_x = int(cosA*right-sinA*bottom-box_o_x*cosA+box_o_y*sinA+box_o_x)
            o_rot_rightbottom_y = int(sinA*right+cosA*bottom-box_o_x*sinA-box_o_y*cosA+box_o_y)

            
            if o_rot_lefttop_x<0:
                o_rot_lefttop_x=0
            if o_rot_leftbottom_x<0:
                o_rot_leftbottom_x=0
            if o_rot_righttop_x<0:
                o_rot_righttop_x=0
            if o_rot_rightbottom_x<0:
                o_rot_rightbottom_x=0
                
            if o_rot_lefttop_y<0:
                o_rot_lefttop_y=0
            if o_rot_leftbottom_y<0:
                o_rot_leftbottom_y=0
            if o_rot_righttop_y<0:
                o_rot_righttop_y=0
            if o_rot_rightbottom_y<0:
                o_rot_rightbottom_y=0
                
            if o_rot_lefttop_x>imw-1:
                o_rot_lefttop_x=imw-1
            if o_rot_leftbottom_x>imw-1:
                o_rot_leftbottom_x=imw-1
            if o_rot_righttop_x>imw-1:
                o_rot_righttop_x=imw-1
            if o_rot_rightbottom_x>imw-1:
                o_rot_rightbottom_x=imw-1
                
            if o_rot_lefttop_y>imh-1:
                o_rot_lefttop_y=imh-1
            if o_rot_leftbottom_y>imh-1:
                o_rot_leftbottom_y=imh-1
            if o_rot_righttop_y>imh-1:
                o_rot_righttop_y=imh-1
            if o_rot_rightbottom_y>imh-1:
                o_rot_rightbottom_y=imh-1
                          
            
            p_x=[o_rot_lefttop_x,o_rot_leftbottom_x,o_rot_righttop_x,o_rot_rightbottom_x]
            p_y=[o_rot_lefttop_y,o_rot_leftbottom_y,o_rot_righttop_y,o_rot_rightbottom_y]
            
            #计算窗口的长和宽
            left = np.min(p_x)
            top = np.min(p_y)

            right = np.max(p_x)
            bottom = np.max(p_y)
            
#            im_crop = imgs[int(top):int(bottom)+1,int(left):int(right)+1]
#            skimage.io.imsave(photofilename+'_crop'+box[0]+'.jpg',im_crop)

            bndbox_dict=dict(xmin=int(left),
                             ymin=int(top),
                             xmax=int(right),
                             ymax=int(bottom))

            return dict(name=unicode(u'text'),
                        pose='left',
                        truncated=1,
                        difficult=box[1],
                        bndbox=bndbox_dict)

        objects = [format_object(box, imw, imh) for box in boxs]
                    
        annotation_dict=dict(annotation=dict(folder='VOC2007',
                                     filename=photofilename,
                                     source=source_dict,
                                     owner=owner_dict,
                                     size=size_dict,
                                     segmented=0,
                                     object=objects
                                     )
                            )          

        return annotation_dict

     # 载入gt标记文件列表
    gt_filelist = mark_folder+u'filelist.txt'
    try:
        gt_filenames = np.loadtxt(gt_filelist, str, delimiter='\t')
    except:   
        gt_filenames = np.loadtxt(gt_filelist, str, delimiter='\t')
       
       
   
    for gt_filename in gt_filenames:
        photo_filename = photo_folder+gt_filename[:-3]+u'.jpg'
        print photo_filename
        
        
        
        # load mark files,获取图像中box的列表
        imgs = skimage.io.imread(photo_filename,as_grey=False)
        
        # 图像要满足宽高比要求
        if imgs.shape[1]/imgs.shape[0]<=min_img_w_h_ratio or imgs.shape[1]/imgs.shape[0]>=max_img_w_h_ratio:
            continue
        
        boxs = []
        with open(mark_folder+gt_filename) as f:
            gt_lines = [[str(x) for x in line.strip().split(' ')] for line in f]
       
        for gt_line in gt_lines:
           
            # 处理空行的情况
            if len(gt_line)<2:
                continue

            box=gt_line[0:7]
            
            # 标记是否满足宽高比要求
            if float(box[4])*float(box[5])/(imgs.shape[0]*imgs.shape[1])<too_big_overlap and float(box[4])/float(box[5])>min_w_h_ratio and float(box[4])/float(box[5])<max_w_h_ratio:           
                boxs.append(box)              
        
        if len(boxs)<1:
            continue
        
#        bHasNonDifficText=False
#        for box in boxs:      
#            # 标记是否包含difficult=0的区域，并且满足宽高比要求，如果所有子块都不包含difficult=0的区域，则不保存
#            if box[1]=='0' and float(box[4])*float(box[5])/(imgs.shape[0]*imgs.shape[1])<too_big_overlap and float(box[4])/float(box[5])>min_w_h_ratio and float(box[4])/float(box[5])<max_w_h_ratio:           
#                bHasNonDifficText=True 
#            
#            if box[1]=='0' and False==(float(box[4])*float(box[5])/(imgs.shape[0]*imgs.shape[1])<too_big_overlap and float(box[4])/float(box[5])>min_w_h_ratio and float(box[4])/float(box[5])<max_w_h_ratio):           
#                bHasNonDifficText=False               
#                
#        if False==bHasNonDifficText:
#            continue

        json_dict_str=make_json_dict(imgs,gt_filename[:-3]+u'.jpg', imgs.shape[1],imgs.shape[0],boxs)
        xml_str=pythonJsonToXml(json_dict_str)

        #write to file
        xml_filename= photo_folder+gt_filename[:-3]+u'.xml'
        xml_file_object = codecs.open(xml_filename, 'w','utf-8')    

        xml_file_object.write(xml_str)

        xml_file_object.close()        
    
    a=1
    

def read(filename, default=None):
    try:
        with open(filename, 'rb') as f:
            return f.read()
    except:
        return default


# 转换到lif文件
def generate_MTR500_objs_2lif(photo_folder,mark_folder):    
    
    
    #写入训练图像文件的列表，文件名 类别号
    save_directory = photo_folder+u'ext_samples_gt/'
    if False==os.path.exists(save_directory):
        os.makedirs(save_directory) 
    
    
     # 转换到lif文件
    
    def format_shape(px,py):
    	return dict(label='text',
    				line_color=None,
    				fill_color=None,
    				points=[(px[0], py[0]), (px[2], py[2]),(px[3], py[3]),(px[1], py[1])])
    
   
    	
        	
        
    
     # 载入gt标记文件列表
    gt_filelist = mark_folder+u'filelist.txt'
    try:
        gt_filenames = np.loadtxt(gt_filelist, str, delimiter='\t')
    except:   
        gt_filenames = np.loadtxt(gt_filelist, str, delimiter='\t')
       
       
   
    for gt_filename in gt_filenames:
        photo_filename = photo_folder+gt_filename[:-3]+u'.jpg'
        print photo_filename
        
        # load mark files,获取图像中box的列表
        imgs = io.imread(photo_filename,as_grey=False)
        imw=imgs.shape[1]
        imh=imgs.shape[0]
        
        boxs = []
        with open(mark_folder+gt_filename) as f:
            gt_lines = [[str(x) for x in line.strip().split(' ')] for line in f]
       
        for gt_line in gt_lines:
           
            # 处理空行的情况
            if len(gt_line)<1:
                continue

            box=gt_line[0:7]
            boxs.append(box)
            
        shapes=[]
       
        for box in boxs:
            
            
            
            
            #计算窗口的长和宽
            left = float(box[2])
            top = float(box[3])

            right = float(box[2])+float(box[4])
            bottom = float(box[3])+float(box[5])
            
            box_o_x = float(box[2])+float(box[4])*0.5
            box_o_y =  float(box[3])+float(box[5])*0.5
            
            rot_theta = float(box[6])
            
            cosA = math.cos(rot_theta)
            sinA = math.sin(rot_theta)
            
            # 绕中心旋转方框
            o_rot_lefttop_x = int(cosA*left-sinA*top-box_o_x*cosA+box_o_y*sinA+box_o_x)
            o_rot_lefttop_y = int(sinA*left+cosA*top-box_o_x*sinA-box_o_y*cosA+box_o_y)
            
            o_rot_leftbottom_x = int(cosA*left-sinA*bottom-box_o_x*cosA+box_o_y*sinA+box_o_x)
            o_rot_leftbottom_y = int(sinA*left+cosA*bottom-box_o_x*sinA-box_o_y*cosA+box_o_y)
            
            o_rot_righttop_x = int(cosA*right-sinA*top-box_o_x*cosA+box_o_y*sinA+box_o_x)
            o_rot_righttop_y = int(sinA*right+cosA*top-box_o_x*sinA-box_o_y*cosA+box_o_y)
            
            o_rot_rightbottom_x = int(cosA*right-sinA*bottom-box_o_x*cosA+box_o_y*sinA+box_o_x)
            o_rot_rightbottom_y = int(sinA*right+cosA*bottom-box_o_x*sinA-box_o_y*cosA+box_o_y)

            
            if o_rot_lefttop_x<0:
                o_rot_lefttop_x=0
            if o_rot_leftbottom_x<0:
                o_rot_leftbottom_x=0
            if o_rot_righttop_x<0:
                o_rot_righttop_x=0
            if o_rot_rightbottom_x<0:
                o_rot_rightbottom_x=0
                
            if o_rot_lefttop_y<0:
                o_rot_lefttop_y=0
            if o_rot_leftbottom_y<0:
                o_rot_leftbottom_y=0
            if o_rot_righttop_y<0:
                o_rot_righttop_y=0
            if o_rot_rightbottom_y<0:
                o_rot_rightbottom_y=0
                
            if o_rot_lefttop_x>imw-1:
                o_rot_lefttop_x=imw-1
            if o_rot_leftbottom_x>imw-1:
                o_rot_leftbottom_x=imw-1
            if o_rot_righttop_x>imw-1:
                o_rot_righttop_x=imw-1
            if o_rot_rightbottom_x>imw-1:
                o_rot_rightbottom_x=imw-1
                
            if o_rot_lefttop_y>imh-1:
                o_rot_lefttop_y=imh-1
            if o_rot_leftbottom_y>imh-1:
                o_rot_leftbottom_y=imh-1
            if o_rot_righttop_y>imh-1:
                o_rot_righttop_y=imh-1
            if o_rot_rightbottom_y>imh-1:
                o_rot_rightbottom_y=imh-1
                          
            
            p_x=[o_rot_lefttop_x,o_rot_leftbottom_x,o_rot_righttop_x,o_rot_rightbottom_x]
            p_y=[o_rot_lefttop_y,o_rot_leftbottom_y,o_rot_righttop_y,o_rot_rightbottom_y]
            
            
            
            #计算窗口的长和宽
            left = np.min(p_x)
            top = np.min(p_y)

            right = np.max(p_x)
            bottom = np.max(p_y)
            
            shape=format_shape(p_x,p_y)
            shapes.append(shape)
    
        lf = LabelFile()
             
        try:
            lf.save(save_directory+gt_filename[:-3]+u'.jpg'+LabelFile.suffix, shapes, unicode(photo_filename), read(photo_filename, None), [0,255,0,128], [0,128,255,155])
                
        except LabelFileError:
        	    
            continue
            
            
   
# 转换手工文字标识为xml文件给rcnn训练 
def rcnn_recog_data_2xmls(photo_folder, mark_folder):
     
    min_w_h_ratio=0.117
    max_w_h_ratio=15.500

    min_img_w_h_ratio=0.25
    max_img_w_h_ratio=4.0
    
    too_big_overlap=0.8

    # 保存到xml文件
    def make_json_dict(imgs,photofilename,imw,imh,boxs):
        source_dict=dict(database='The VOC2007 Database',
                 annotation='PASCAL VOC2007',
                 image='manual_collect',
                 flickrid='0000000')
        owner_dict=dict(flickrid='0000000',
                name='hwang')
        size_dict=dict(width=imw,
               height=imh,
               depth=3)   

        def format_object(box,imw,imh):          
        
            #计算窗口的长和宽
            left = float(box[2])
            top = float(box[3])

            right = float(box[2])+float(box[4])
            bottom = float(box[3])+float(box[5])
            
            
            
#            im_crop = imgs[int(top):int(bottom)+1,int(left):int(right)+1]
#            skimage.io.imsave(photofilename+'_crop'+box[0]+'.jpg',im_crop)

            bndbox_dict=dict(xmin=int(left),
                             ymin=int(top),
                             xmax=int(right),
                             ymax=int(bottom))

#            return dict(name=unicode('char_'+str(char2label(unicode(box[6].strip(),"utf-8")))),
            return dict(name=unicode(u'text'),                        
                        pose='left',
                        truncated=1,
                        difficult=0,
                        bndbox=bndbox_dict)

        objects = [format_object(box, imw, imh) for box in boxs]
                    
        annotation_dict=dict(annotation=dict(folder='VOC2007',
                                     filename=photofilename,
                                     source=source_dict,
                                     owner=owner_dict,
                                     size=size_dict,
                                     segmented=0,
                                     object=objects
                                     )
                            )          

        return annotation_dict

     # 载入gt标记文件列表
    gt_filelist = mark_folder+u'filelist.txt'
    try:
        gt_filenames = np.loadtxt(gt_filelist, str, delimiter='\t')
    except:   
        gt_filenames = np.loadtxt(gt_filelist, str, delimiter='\t')
       
       
   
    for gt_filename in gt_filenames:
        
        photo_filename = gt_filename[:-3]       
        
        # 对于labelme的工具标记老版本时*.gt格式，对于新版本时*.jpg.gt格式
        if False==os.path.exists(photo_folder+photo_filename):
            photo_filename = gt_filename[:-3]+u'.jpg'
        
        print photo_folder+photo_filename        
        
        
        # load mark files,获取图像中box的列表
        imgs = skimage.io.imread(photo_folder+photo_filename,as_grey=False)
        
        # 图像要满足宽高比要求
        if imgs.shape[1]/imgs.shape[0]<=min_img_w_h_ratio or imgs.shape[1]/imgs.shape[0]>=max_img_w_h_ratio:
            continue
        
        boxs = []
        with open(mark_folder+gt_filename) as f:
            gt_lines = [[str(x) for x in line.strip().split('\t')] for line in f]
       
        for gt_line in gt_lines:
           
            # 处理空行的情况
            if len(gt_line)<2:
                continue

            box=gt_line[0:7]
            
            # 标记是否满足宽高比要求
            if float(box[4])*float(box[5])/(imgs.shape[0]*imgs.shape[1])<too_big_overlap and float(box[4])/float(box[5])>min_w_h_ratio and float(box[4])/float(box[5])<max_w_h_ratio:           
                boxs.append(box)              
        
        if len(boxs)<1:
            continue
#        bHasNonDifficText=False
#        for box in boxs:      
#            # 标记是否包含difficult=0的区域，并且满足宽高比要求，如果所有子块都不包含difficult=0的区域，则不保存
#            if float(box[4])*float(box[5])/(imgs.shape[0]*imgs.shape[1])<too_big_overlap and float(box[4])/float(box[5])>min_w_h_ratio and float(box[4])/float(box[5])<max_w_h_ratio:           
#                bHasNonDifficText=True 
#            
#            if False==(float(box[4])*float(box[5])/(imgs.shape[0]*imgs.shape[1])<too_big_overlap and float(box[4])/float(box[5])>min_w_h_ratio and float(box[4])/float(box[5])<max_w_h_ratio):           
#                bHasNonDifficText=False               
#                
#        if False==bHasNonDifficText:
#            continue

        json_dict_str=make_json_dict(imgs,photo_filename, imgs.shape[1],imgs.shape[0],boxs)
        xml_str=pythonJsonToXml(json_dict_str)

        #write to file
        xml_filename= mark_folder+photo_filename[:-4]+u'.xml'
        xml_file_object = codecs.open(xml_filename, 'w','utf-8')    

        xml_file_object.write(xml_str)

        xml_file_object.close()        
    
    a=1
 
#统计obj mark标记的框的宽高范围
def objmark_wh_stat(obj_mark_folder):

    #用来存储宽度和高度序列，用来制作直方图
    all_box_width=[]
    all_box_height=[]
    
        
    # load目标类别列表
    type_filelist = obj_mark_folder+u'filelist.txt'
    try:
       type_names = np.loadtxt(type_filelist, str, delimiter='\t')
    except:   
       type_names = np.loadtxt(type_filelist, str, delimiter='\t')


    # load 每一类的图像列表
    for type_name in type_names:
        mark_foldername = obj_mark_folder+unicode(type_name,"utf-8")
       
        #获得目录下所有label files
        mark_filenames=os.listdir(mark_foldername)  
        
        for mark_filename in mark_filenames:
            
            if os.path.splitext(mark_filename)[1] != '.txt' :
                continue    

            if os.path.splitext(mark_filename)[0] == 'filelist' :
                continue    

            if os.path.splitext(mark_filename)[0] == 'filelist~' :
                continue                   
            
           
            mark_filename = obj_mark_folder+unicode(type_name,"utf-8")+u'/'+mark_filename
            
            print mark_filename
            
            # load mark files,获取图像中box的列表
            obj_box=[]
            with open(mark_filename) as f:    
                array = [[int(x) for x in line.split()] for line in f]
                
            # mark is empty line
            if len(array)<1:
                continue
            
            if len(array[0])<4:
                continue
                
            for i in range(len(array[0])/4):
                box=[]
                for j in range(4):
                   box.append(array[0][i*4+j])
                obj_box.append(box)
             
                   
           

            for i in range(0,len(obj_box)):
                 box=obj_box[i]      

                 # right小于left或者bottom小于top
                 if box[2] <box[0] or box[3] <box[1]:
                     continue
                 
                 all_box_height.append(int(box[3]-box[1]))
                 all_box_width.append(int(box[2]-box[0]))
                 
                 
    num_bins = 50
    plt.hist(all_box_width, num_bins, normed=1, facecolor='green', alpha=0.5)
    plt.hist(all_box_height, num_bins, normed=1, facecolor='blue', alpha=0.5)
    plt.title('Histogram')
    plt.show()
    
    return 0
 
# format_marks_folder formated labels lif file folder 
def generate_brandname_productname_folders(format_marks_folder, photo_folder):
    
    brandname_folders=format_marks_folder+'brandname_folders/'
    if False==os.path.exists(brandname_folders):
         os.makedirs(brandname_folders)  
         
    productname_folders=format_marks_folder+'productname_folders/'
    if False==os.path.exists(productname_folders):
         os.makedirs(productname_folders) 
         
    categorized_folder_brandname=format_marks_folder+'categorized_folder_brandname/'
    if False==os.path.exists(categorized_folder_brandname):
         os.makedirs(categorized_folder_brandname)  
                    
    categorized_folder_productname=format_marks_folder+'categorized_folder_productname/'
    if False==os.path.exists(categorized_folder_productname):
         os.makedirs(categorized_folder_productname)  
         
    colorname_folder=format_marks_folder+'categorized_folder_colors/'
    if False==os.path.exists(colorname_folder):
         os.makedirs(colorname_folder)  
         
     # all files
    all_mark_files=os.listdir(format_marks_folder)  
    
    for obj_filename in all_mark_files:
    
        if os.path.splitext(obj_filename)[1] != '.lif' :
            continue
        
        #obj lif file by labelme
        labelFile = LabelFile(format_marks_folder+obj_filename)
        
        #transform1: random set obj in background , random obj size in (80,240)      
        # process only one shape
        index=0
        for shape in labelFile.shapes:
            
            photoname = photo_folder+obj_filename[:-4]
             
            print photoname
       
       
            
            p_x=[p[0] for p in shape[1]]
            p_y=[p[1] for p in shape[1]]  
          
            # coordination of out box
            left = int(np.min(p_x))
            top = int(np.min(p_y))
    
            right = int(np.max(p_x))
            bottom = int(np.max(p_y))   
            
            label = shape[0]
            
            label_items = label.split('-')
    
            
            # brandname-productname-brandbackgroundcolor-prductbackgoundcolor-a(b/c)-x
            if label_items[5]=='1':
                # brandname
                brandname_folder=categorized_folder_brandname+label_items[0]+'_'+label_items[2]+'_'+label_items[4]

                  
                if False==os.path.exists(brandname_folder):
                    os.makedirs(brandname_folder)                
                
#                save_labelimg_name=brandname_folder+'/'+obj_filename[:-8]+'_'+str(index)+'.jpg'
#                
#                imgs=io.imread(photoname)
#                im_crop_s = imgs[top:bottom,left:right]
#                
#                io.imsave(save_labelimg_name, im_crop_s)
                
                # only brandname
                brandname_folder=brandname_folders+label_items[0]
                  
                if False==os.path.exists(brandname_folder):
                    os.makedirs(brandname_folder)   
                
            if label_items[5]=='2':
                # brandname
                productname_folder=categorized_folder_productname+label_items[1]+'_'+label_items[0]+'_'+label_items[3]+'_'+label_items[4]

                  
                if False==os.path.exists(productname_folder):
                    os.makedirs(productname_folder)                
                
#                save_labelimg_name=productname_folder+'/'+obj_filename[:-8]+'_'+str(index)+'.jpg'
#                
#                imgs=io.imread(photoname)
#                im_crop_s = imgs[top:bottom,left:right]
#                
#                io.imsave(save_labelimg_name, im_crop_s) 
                
                # only productname
                productname_folder=productname_folders+label_items[1]
                  
                if False==os.path.exists(productname_folder):
                    os.makedirs(productname_folder)   
                    
            # save colors
            color_folder=colorname_folder+label_items[2]
                  
            if False==os.path.exists(color_folder):
                   os.makedirs(color_folder) 
                   
            color_folder=colorname_folder+label_items[3]
                  
            if False==os.path.exists(color_folder):
                   os.makedirs(color_folder) 
                
            index=index+1
           
def brandname_productname_2label(label):
    # 载入brandnames
    brandnames_filelist = u'D:/hwang/fridge-photos/food_sampels_total_class/brandname_folders/filelist1.txt'
    try:
        brandnames = np.loadtxt(brandnames_filelist, str, delimiter='\t')
    except:   
        brandnames = np.loadtxt(brandnames_filelist, str, delimiter='\t')
        
     # 载入productnames
    productnames_filelist = u'D:/hwang/fridge-photos/food_sampels_total_class/productname_folders/filelist1.txt'
    try:
        productnames = np.loadtxt(productnames_filelist, str, delimiter='\t')
    except:   
        productnames = np.loadtxt(productnames_filelist, str, delimiter='\t')
        
    label_items = label.split('-')
    
    # brandname-productname-brandbackgroundcolor-prductbackgoundcolor-a(b/c)-x
    if label_items[5]=='1':
        index = 0
        for brandname in brandnames:   
            if label_items[0] == unicode(brandname,'utf8'):
                return 'brand'+str(index)
            index=index+1
            
    if label_items[5]=='2':
        index = 0
        for productname in productnames:   
            if label_items[1] == unicode(productname,'utf8'):
                return 'product'+str(index)
            index=index+1
            
    return None
   
# 转换lif标识为xml文件给rcnn训练 
def rcnn_lif_data_2xmls(photo_folder, mark_folder):
     
    min_w_h_ratio=0.117
    max_w_h_ratio=15.500

    min_img_w_h_ratio=0.25
    max_img_w_h_ratio=4.0
    
    too_big_overlap=0.8

    # 保存到xml文件
    def make_json_dict(imgs,photofilename,imw,imh,boxs):
        source_dict=dict(database='The VOC2007 Database',
                 annotation='PASCAL VOC2007',
                 image='manual_collect',
                 flickrid='0000000')
        owner_dict=dict(flickrid='0000000',
                name='hwang')
        size_dict=dict(width=imw,
               height=imh,
               depth=3)   

        def format_object(box,imw,imh):          
        
            #计算窗口的长和宽
            left = float(box[2])
            top = float(box[3])

            right = float(box[2])+float(box[4])
            bottom = float(box[3])+float(box[5])
            
            label=box[6]
           
            
            
            im_crop = imgs[int(top):int(bottom)+1,int(left):int(right)+1]
            io.imsave(photofilename+'_crop'+str(box[0])+'.jpg',im_crop)

            bndbox_dict=dict(xmin=int(left),
                             ymin=int(top),
                             xmax=int(right),
                             ymax=int(bottom))

            class_name=unicode(brandname_productname_2label(label))
            return dict(name=unicode(brandname_productname_2label(label)),
#            return dict(name=unicode(u'text'),                        
                        pose='left',
                        truncated=1,
                        difficult=0,
                        bndbox=bndbox_dict)

        objects = [format_object(box, imw, imh) for box in boxs]
                    
        annotation_dict=dict(annotation=dict(folder='VOC2007',
                                     filename=photofilename,
                                     source=source_dict,
                                     owner=owner_dict,
                                     size=size_dict,
                                     segmented=0,
                                     object=objects
                                     )
                            )          

        return annotation_dict

    # all files
    all_mark_files=os.listdir(mark_folder)  
    
    for obj_filename in all_mark_files:
    
        if os.path.splitext(obj_filename)[1] != '.lif' :
            continue
        
        #obj lif file by labelme
        labelFile = LabelFile(mark_folder+obj_filename)
        
        boxs = []
        #transform1: random set obj in background , random obj size in (80,240)      
        # process only one shape
        for shape in labelFile.shapes:
            
            photoname = photo_folder+obj_filename[:-4]
             
            print photoname
       
       
            
            p_x=[p[0] for p in shape[1]]
            p_y=[p[1] for p in shape[1]]  
          
            # coordination of out box
            left = int(np.min(p_x))
            top = int(np.min(p_y))
    
            right = int(np.max(p_x))
            bottom = int(np.max(p_y))   
            
            label = shape[0] 
            
            # load mark files,获取图像中box的列表
            imgs = io.imread(photoname,as_grey=False)
            
            # 图像要满足宽高比要求
            if imgs.shape[1]/imgs.shape[0]<=min_img_w_h_ratio or imgs.shape[1]/imgs.shape[0]>=max_img_w_h_ratio:
                continue    

            box=[0,0,left,top,right-left+1,bottom-top+1,label]
            
            # 标记是否满足宽高比要求
            if float(box[4])*float(box[5])/(imgs.shape[0]*imgs.shape[1])<too_big_overlap and float(box[4])/float(box[5])>min_w_h_ratio and float(box[4])/float(box[5])<max_w_h_ratio:           
                boxs.append(box)              
        
        if len(boxs)<1:
            continue
        


        json_dict_str=make_json_dict(imgs,obj_filename[:-4], imgs.shape[1],imgs.shape[0],boxs)
        xml_str=pythonJsonToXml(json_dict_str)

        #write to file
        xml_filename= mark_folder+obj_filename[:-8]+u'.xml'
        xml_file_object = codecs.open(xml_filename, 'w','utf-8')    

        xml_file_object.write(xml_str)

        xml_file_object.close()        
    
# 去掉xml中product的标签，仅剩下brand的标签
def process_xmls(xml_folder):
     
    

    # 保存到xml文件
    def make_json_dict(photofilename,imw,imh,boxs):
        source_dict=dict(database='The VOC2007 Database',
                 annotation='PASCAL VOC2007',
                 image='manual_collect',
                 flickrid='0000000')
        owner_dict=dict(flickrid='0000000',
                name='hwang')
        size_dict=dict(width=imw,
               height=imh,
               depth=3)   

        def format_object(box,imw,imh):          
        
            #计算窗口的长和宽
            left = float(box[2])
            top = float(box[3])

            right = float(box[2])+float(box[4])
            bottom = float(box[3])+float(box[5])
            
            label=box[6]
           
            
            
#            im_crop = imgs[int(top):int(bottom)+1,int(left):int(right)+1]
#            skimage.io.imsave(photofilename+'_crop'+box[0]+'.jpg',im_crop)

            bndbox_dict=dict(xmin=int(left),
                             ymin=int(top),
                             xmax=int(right),
                             ymax=int(bottom))


            return dict(name=unicode(label),
#            return dict(name=unicode(u'text'),                        
                        pose='left',
                        truncated=1,
                        difficult=0,
                        bndbox=bndbox_dict)

        objects = [format_object(box, imw, imh) for box in boxs]
                    
        annotation_dict=dict(annotation=dict(folder='VOC2007',
                                     filename=photofilename,
                                     source=source_dict,
                                     owner=owner_dict,
                                     size=size_dict,
                                     segmented=0,
                                     object=objects
                                     )
                            )          

        return annotation_dict

    # all files
    all_mark_files=os.listdir(xml_folder)  
    
    for obj_filename in all_mark_files:
    
        if os.path.splitext(obj_filename)[1] != '.xml' :
            continue
        
        print obj_filename
        
        xml_file_object = codecs.open(xml_folder+obj_filename, 'r','utf-8')    
			
        xml_str=xml_file_object.read()
			
        xml_file_object.close()
        
        convertedDict = xmltodict.parse(xml_str)					
        
        boxs = []
        
        #transform1: random set obj in background , random obj size in (80,240)      
        # process only one shape
        
   
        for magnet in convertedDict["annotation"]["object"]:                            
       
            if len(convertedDict["annotation"]["object"])>2:               
               magnet= convertedDict["annotation"]["object"]
            
            if magnet["name"][0:5] == "brand":         
          
                # coordination of out box
                left = int(magnet["bndbox"]["xmin"])
                top = int(magnet["bndbox"]["ymin"])
        
                right = int(magnet["bndbox"]["xmax"])
                bottom = int(magnet["bndbox"]["ymax"])
                
                label = magnet["name"] 
    
                box=[0,0,left,top,right-left+1,bottom-top+1,label]
                
                boxs.append(box)  
                
            if len(convertedDict["annotation"]["object"])>2: 
                break
            
        if len(boxs)<1:
            continue
        
        json_dict_str=make_json_dict(obj_filename[:-4]+'.jpg', 1280,720,boxs)
        xml_str=pythonJsonToXml(json_dict_str)

        #write to file
        xml_filename= xml_folder+'/new/'+obj_filename
        xml_file_object = codecs.open(xml_filename, 'w','utf-8')    

        xml_file_object.write(xml_str)

        xml_file_object.close()    

# transfered oem lif files to put in categorized folders       
def generate_oem_categorize_folders(marks_folder, photo_folder):
    
    
         
    categorized_folder=marks_folder+'categorized_folder/'
    if False==os.path.exists(categorized_folder):
         os.makedirs(categorized_folder)  
         
     # all files
    all_mark_files=os.listdir(marks_folder)  
    
    for obj_filename in all_mark_files:
    
        if os.path.splitext(obj_filename)[1] != '.lif' :
            continue
        
        #obj lif file by labelme
        labelFile = LabelFile(marks_folder+obj_filename)
        
        #transform1: random set obj in background , random obj size in (80,240)      
        # process only one shape
        index=0
        for shape in labelFile.shapes:
            
            photoname = photo_folder+obj_filename[:-4]
             
            print photoname
       
       
            
            p_x=[p[0] for p in shape[1]]
            p_y=[p[1] for p in shape[1]]  
          
            # coordination of out box
            left = int(np.min(p_x))
            top = int(np.min(p_y))
    
            right = int(np.max(p_x))
            bottom = int(np.max(p_y))   
            
            label = shape[0]
            
        
            categorize_folder_path=categorized_folder+label

              
            if False==os.path.exists(categorize_folder_path):
                os.makedirs(categorize_folder_path)                
            
            save_labelimg_name=categorize_folder_path+'/'+obj_filename[:-8]+'_'+str(index)+'.jpg'
            
            imgs=io.imread(photoname)
            im_crop_s = imgs[top:bottom,left:right]
            
            io.imsave(save_labelimg_name, im_crop_s)
            
          
                
            index=index+1

# modify nth label of lif file in folder     
def lif_batch_modification(marks_folder, labels_folder, modified_label):
     # all files
    all_mark_files=os.listdir(labels_folder)  
    
    for obj_filename in all_mark_files:
    
        if os.path.splitext(obj_filename)[1] != '.jpg' :
            continue
        
        #obj lif file by labelme
        index=os.path.splitext(obj_filename)[0].split('_')[-1]
        
        labelfilename=marks_folder+os.path.splitext(obj_filename)[0][:-(len(index)+1)]+'.jpg.lif'
        
        print labelfilename
        
        labelFile = LabelFile(labelfilename)
        
        index=int(index)
        
        
        def format_shape(lbl,pts):
        	return dict(label=lbl,
        				line_color=None,
        				fill_color=None,
        				points=pts)
        
        counti=0
        shapes=[]
        for shape in labelFile.shapes:  
            if counti==index:
                shapes.append(format_shape(modified_label,shape[1]))
            else:
                shapes.append(format_shape(shape[0],shape[1]))
                
            counti=counti+1
            
            
        
       
          
                
        try:
        	labelFile.save(labelfilename, shapes, labelFile.imagePath, labelFile.imageData,labelFile.lineColor, labelFile.fillColor)        	
        except :
        	continue

# 列出所有rcnn的类别     
def list_rcnn_labels_dir(rcnn_labels_folder): 
 
    labels=[]
    nclasses_for_rcnn=0
    for dirName, subdirList, fileList in os.walk(rcnn_labels_folder):
        #print unicode('%s' % dirName,'GB2312')
        
        labels_in_dir=dirName.split('\\')

        bFindDisable=False        
        # 如果labels_in_dir中disable项目就挑果
        for label_item in labels_in_dir:
            if label_item == 'disable':
                bFindDisable=True
                break

        if True==bFindDisable:
            continue
        
        label=[]
        # 例如rcnn_labels_folder\果汁\冰糖系列\冰糖金橘_统一_绿_a
        if len(labels_in_dir)>3:
            label.append(labels_in_dir[-1])
            label.append(labels_in_dir[2]+'~'+str(nclasses_for_rcnn))
            label.append(labels_in_dir[1])
            labels.append(label)
        elif  len(labels_in_dir)>2:
            nclasses_for_rcnn=nclasses_for_rcnn+1
            
            # 例如rcnn_labels_folder\果汁\冰糖系列
            label.append(labels_in_dir[-1])
            label.append(labels_in_dir[2]+'~'+str(nclasses_for_rcnn))
            label.append(labels_in_dir[1])
            labels.append(label)
            
            
    
    for label in labels:
        print unicode(label[0]+','+label[1]+','+label[2],'GB2312')
      
            
        
#        for fname in fileList:
#            abspath=dirName+'\\'+fname
#            print unicode(abspath,'GB2312') 
        
    return labels


# 找出text-mark-lif-total所有不属于rcnn_lables的标记  
def find_not_in_labels(marks_folder, rcnn_labels_folder):
    
    rcnn_labels=list_rcnn_labels_dir(rcnn_labels_folder)
         
    not_find_labels=[]
         
     # all files
    all_mark_files=os.listdir(marks_folder)  
    
    for obj_filename in all_mark_files:
    
        if os.path.splitext(obj_filename)[1] != '.lif' :
            continue
        
        #obj lif file by labelme
        labelFile = LabelFile(marks_folder+obj_filename)
        

        for shape in labelFile.shapes:
          
            label = shape[0]
            
            bFindLable = False
            
            for rcnn_label in rcnn_labels:   
                if unicode(rcnn_label[0],'GB2312') == label:
                    bFindLable=True
                    break
            
            if bFindLable==False:                
                for no_label in not_find_labels:   
                    if no_label == label:
                        bFindLable=True
                        break                  
                
                if bFindLable==False: 
                    not_find_labels.append(label)
            
    for no_label in not_find_labels:
        print no_label
        
def detail_label_2_rcnn_labels(label,rcnn_labels):
   
            
    for rcnn_label in rcnn_labels:   
        if unicode(rcnn_label[0],'GB2312') == label:
            index=rcnn_label[1].split('~')[-1]         
            return 'textmark'+index
            
    return 'None'
       
# 转换text-mark-lif-total中的labels到 rcnn-classes文件夹中的各个类别上
def detailed_text_mark_lif_data_2xmls(photo_folder, mark_folder, rcnn_labels_folder):
    
    xmls_folder=mark_folder+'rcnn_xmls/'
    if False==os.path.exists(xmls_folder):
         os.makedirs(xmls_folder)  
         
    rcnn_labels=list_rcnn_labels_dir(rcnn_labels_folder)
     
    min_w_h_ratio=0.117
    max_w_h_ratio=15.500

    min_img_w_h_ratio=0.25
    max_img_w_h_ratio=4.0
    
    too_big_overlap=0.8

    # 保存到xml文件
    def make_json_dict(imgs,photofilename,imw,imh,boxs,rcnn_labels):
        source_dict=dict(database='The VOC2007 Database',
                 annotation='PASCAL VOC2007',
                 image='manual_collect',
                 flickrid='0000000')
        owner_dict=dict(flickrid='0000000',
                name='hwang')
        size_dict=dict(width=imw,
               height=imh,
               depth=3)   

        def format_object(box,imw,imh,rcnn_labels):          
        
            #计算窗口的长和宽
            left = float(box[2])
            top = float(box[3])

            right = float(box[2])+float(box[4])
            bottom = float(box[3])+float(box[5])
            
            label=box[6]
           
            
            
            im_crop = imgs[int(top):int(bottom)+1,int(left):int(right)+1]
            io.imsave(photofilename+'_crop'+str(box[0])+'.jpg',im_crop)

            bndbox_dict=dict(xmin=int(left),
                             ymin=int(top),
                             xmax=int(right),
                             ymax=int(bottom))

            class_name=unicode(detail_label_2_rcnn_labels(label,rcnn_labels))
            
            if 'None'==class_name:
                return 'None'
                
            return dict(name=class_name,
#            return dict(name=unicode(u'text'),                        
                        pose='left',
                        truncated=1,
                        difficult=0,
                        bndbox=bndbox_dict)

        objects=[]
        for box in boxs:
            fmt_object=format_object(box, imw, imh,rcnn_labels) 
            if 'None'!=fmt_object:
                objects.append(fmt_object)
                    
        annotation_dict=dict(annotation=dict(folder='VOC2007',
                                     filename=photofilename,
                                     source=source_dict,
                                     owner=owner_dict,
                                     size=size_dict,
                                     segmented=0,
                                     object=objects
                                     )
                            )          

        return annotation_dict

    # all files
    all_mark_files=os.listdir(mark_folder)  
    
    for obj_filename in all_mark_files:
    
        if os.path.splitext(obj_filename)[1] != '.lif' :
            continue

        print mark_folder+obj_filename
        
        #obj lif file by labelme
        labelFile = LabelFile(mark_folder+obj_filename)
        
        photoname = photo_folder+obj_filename[:-4]

        
        boxs = []
        #transform1: random set obj in background , random obj size in (80,240)      
        # process only one shape
        for shape in labelFile.shapes:       
       
            
            p_x=[p[0] for p in shape[1]]
            p_y=[p[1] for p in shape[1]]  
          
            # coordination of out box
            left = int(np.min(p_x))
            top = int(np.min(p_y))
    
            right = int(np.max(p_x))
            bottom = int(np.max(p_y))   
            
            label = shape[0] 
            
            # load mark files,获取图像中box的列表
            imgs = io.imread(photoname,as_grey=False)
            
            # 图像要满足宽高比要求
            if imgs.shape[1]/imgs.shape[0]<=min_img_w_h_ratio or imgs.shape[1]/imgs.shape[0]>=max_img_w_h_ratio:
                continue    

            box=[0,0,left,top,right-left+1,bottom-top+1,label]
            
            # 标记是否满足宽高比要求
            if float(box[4])*float(box[5])/(imgs.shape[0]*imgs.shape[1])<too_big_overlap and float(box[4])/float(box[5])>min_w_h_ratio and float(box[4])/float(box[5])<max_w_h_ratio:           
                boxs.append(box)              
        
        if len(boxs)<1:
            continue
        


        json_dict_str=make_json_dict(imgs,obj_filename[:-4], imgs.shape[1],imgs.shape[0],boxs,rcnn_labels)
        xml_str=pythonJsonToXml(json_dict_str)

        #write to file         
        xml_filename= xmls_folder+obj_filename[:-8]+u'.xml'
        xml_file_object = codecs.open(xml_filename, 'w','utf-8')    

        xml_file_object.write(xml_str)

        xml_file_object.close()        

  
# 转换lif转换为mask文件
def lif_data_2mask(photo_folder, mark_folder):
  

    # all files
    all_mark_files=os.listdir(mark_folder)  
    
    for obj_filename in all_mark_files:
    
        if os.path.splitext(obj_filename)[1] != '.lif' :
            continue
        
        #obj lif file by labelme
        labelFile = LabelFile(mark_folder+obj_filename)
        
        photoname = photo_folder+obj_filename[:-4]
             
        print photoname

        #处理保存到mask文件
        # load mark files,获取图像中box的列表
        imgs = io.imread(photoname,as_grey=False)
        mask_im = Image.new("RGB",(imgs.shape[1],imgs.shape[0]),"black")  
        draw_mask = ImageDraw.Draw(mask_im)  
            
        #transform1: random set obj in background , random obj size in (80,240)      
        # process only one shape
        for shape in labelFile.shapes:
            
            shape_points=[(p[0],p[1]) for p in shape[1]] 
            
            draw_mask.polygon(shape_points, fill = (255, 255, 255))
                
       
        mask_filename=photo_folder+obj_filename[:-8]+'_mask.jpg'
        mask_im.save(mask_filename)

# 转换lif转换为mask文件, 缩放到500*500尺寸
def lif_data_2mask_scale(photo_folder, mark_folder):

    newimsize=500

    # 转换到lif文件

    def format_shape(shape_points):
        return dict(label='text',
                    line_color=None,
                    fill_color=None,
                    points=shape_points)

    #写入训练图像文件的列表，文件名 类别号
    save_directory = mark_folder+u'gen_mask_samples/'
    if False==os.path.exists(save_directory):
        os.makedirs(save_directory)

    # 挑选过的文件中选择
    chosen_files = os.listdir(mark_folder+'chosed_samples_fcn')

    # all files
    all_mark_files=os.listdir(mark_folder)

    for obj_filename in all_mark_files:



        if os.path.splitext(obj_filename)[1] != '.lif' :
            continue

        if obj_filename[:-4] not in chosen_files:
            continue

        #obj lif file by labelme
        labelFile = LabelFile(mark_folder+obj_filename)

        photoname = photo_folder+obj_filename[:-4]

        print photoname

        #处理保存到mask文件
        # load mark files,获取图像中box的列表
        imgs = io.imread(photoname,as_grey=False)

        xscale=newimsize*1.0/imgs.shape[1]
        yscale=newimsize*1.0/imgs.shape[0]

        im_resize = tf.resize(imgs,(newimsize,newimsize))

        io.imsave(save_directory+obj_filename[:-4],im_resize)

        mask_im = Image.new("P",(newimsize,newimsize),0)
        #mask_im = Image.new("RGB",(newimsize,newimsize),"black")
        draw_mask = ImageDraw.Draw(mask_im)

        #transform1: random set obj in background , random obj size in (80,240)
        # process only one shape
        new_shapes=[]
        for shape in labelFile.shapes:

            shape_points=[(int(p[0]*xscale),int(p[1]*yscale)) for p in shape[1]]

            draw_mask.polygon(shape_points, fill = 1)
            #draw_mask.polygon(shape_points, fill = (255,255,255))

            new_shapes.append(format_shape(shape_points))

        #lf = LabelFile()
        #lf.save(save_directory+obj_filename, new_shapes, unicode(save_directory+obj_filename[:-4]), read(save_directory+obj_filename[:-4], None), [0,255,0,128], [0,128,255,155])

        mask_filename=save_directory+obj_filename[:-8]+'.png'
        mask_im.save(mask_filename)

# 转换单个lif转换为mask文件
# 对于target box缩放到长边500的情况下，scale为缩放倍率
# 需要验证是否有character包含在target_box中，否则返回无效
# # 保存3幅321*321的mask，分别是top，middle，bottom，或者left，middle，right
def lif2mask(lablefilename, mark_folder, photo_folder, scale, target_box):  

 
    #obj lif file by labelme
    labelFile = LabelFile(mark_folder+lablefilename[:-4]+'.lif') 

    #处理保存到mask文件
    # load mark files,获取图像中box的列表
    imgs = io.imread(photo_folder+lablefilename[:-4],as_grey=False)
    mask_im = Image.new('P',(int(imgs.shape[1]*scale),int(imgs.shape[0]*scale)),0)#"black")  
    #mask_im = Image.new('RGB',(int(imgs.shape[1]*scale),int(imgs.shape[0]*scale)),"black")  
    draw_mask = ImageDraw.Draw(mask_im)  
    
    left=np.array([min([p[0] for p in shape[1]]) for  shape in labelFile.shapes])
    
    labelFile = LabelFile(mark_folder+lablefilename[:-4]+'.lif') 
    top=np.array([min([p[1] for p in shape[1]]) for  shape in labelFile.shapes])
    
    labelFile = LabelFile(mark_folder+lablefilename[:-4]+'.lif') 
    right=np.array([max([p[0] for p in shape[1]]) for  shape in labelFile.shapes])
    
    labelFile = LabelFile(mark_folder+lablefilename[:-4]+'.lif') 
    bottom=np.array([max([p[1] for p in shape[1]]) for  shape in labelFile.shapes])

           
    area = (right - left + 1) * (bottom - top + 1)
    
    # 计算和gt里面char box的重合区域，如果完全重合就认为是这个out box里的字
    #找出窗口的最大的坐标位置，以及最小的坐标位置。
    xx1 = np.maximum(target_box[0], left)
    yy1 = np.maximum(target_box[1], top)
    xx2 = np.minimum(target_box[2], right)
    yy2 = np.minimum(target_box[3], bottom)
         
    #计算窗口的长和宽
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)                     

    #计算重合率，占detected_char_boxe的面积比率，
    overlap = (w * h*1.0) / (area)
    overlap_idxs = np.where(overlap > 0.8)[0] 
    
    
    # 如果是2个以下的文字在target_box中就不考虑
    if len(overlap_idxs)<2 :
        return False                        
                 

    
    #transform1: random set obj in background , random obj size in (80,240)      
    # process only one shape
    labelFile = LabelFile(mark_folder+lablefilename[:-4]+'.lif') 
    
    shapeidx=0
    for shape in labelFile.shapes:
        shapeidx=shapeidx+1
        # 仅绘制target_box以内的mask
        #if overlap[shapeidx-1]<=0.9:            
        #    continue
        
        shape_points=[(int(p[0]*scale),int(p[1]*scale)) for p in shape[1]] 
        
        draw_mask.polygon(shape_points, fill = 1)#(128, 0, 0))  
        #draw_mask.polygon(shape_points, fill = (128, 0, 0))   
            
    # 保存3幅mask，分别是top，middle，bottom，或者left，middle，right
    mask_filename=photo_folder+lablefilename[:-8]+'_mask.png'
    mask_im.save(mask_filename)   
    
    return True
        

# 生成ground truth box对应的图片和mask文件
def deeplab_fcn_samples_gen(photo_folder, obj_mark_folder, char_mark_folder):
    
    crop_size=321#108 #321

    #写入训练图像文件的列表，文件名 类别号
    save_directory = obj_mark_folder+u'gen_mask_samples/'
    if False==os.path.exists(save_directory):
        os.makedirs(save_directory) 
        
    # load目标类别列表
    type_filelist = obj_mark_folder+u'filelist.txt'
    try:
       type_names = np.loadtxt(type_filelist, str, delimiter='\t')
    except:   
       type_names = np.loadtxt(type_filelist, str, delimiter='\t')


    # load 每一类的图像列表
    for type_name in type_names:
        mark_filelist = obj_mark_folder+unicode(type_name,"utf-8")+u'/filelist.txt'
        try:
           mark_filenames = np.loadtxt(mark_filelist, str, delimiter='\t')
        except:   
           mark_filenames = np.loadtxt(mark_filelist, str, delimiter='\t')

        for mark_filename in mark_filenames:
            obj_mark_filename=mark_filename 
            
            
            photo_filename = photo_folder+mark_filename[:-4]
            print photo_filename
            
           
            mark_filename = obj_mark_folder+unicode(type_name,"utf-8")+u'/'+mark_filename
            
            # load mark files,获取图像中box的列表
            obj_box=[]
            with open(mark_filename) as f:    
                array = [[int(x) for x in line.split()] for line in f]
                
            # mark is empty line
            if len(array)<1:
                continue
            
            if len(array[0])<4:
                continue
                
            for i in range(len(array[0])/4):
                box=[]
                for j in range(4):
                   box.append(array[0][i*4+j])
                   
                left=min(box[0],box[2])   
                right=max(box[0],box[2]) 
                top=min(box[1],box[3]) 
                bottom=max(box[1],box[3]) 

                box[0]=left
                box[1]=top
                box[2]=right
                box[3]=bottom        			
        			
                obj_box.append(box)
             
            # photo not exist
            if False==os.path.exists(photo_filename):
                continue
            
            #if False==os.path.exists(photo_filename[:-4]+'_mask.jpg'):
            #    continue
            
            if False==os.path.exists(char_mark_folder+obj_mark_filename[:-4]+'.lif'):
                continue
            
            imgs = io.imread(photo_filename,as_grey=False) 
            
           

            for i in range(0,len(obj_box)):
                 box=obj_box[i]      

                 # right小于left或者bottom小于top
                 if box[2] <box[0] or box[3] <box[1]:
                     continue
                 
                 if  np.maximum(box[2]-box[0], box[3]-box[1])<240:
                     continue
                 
                 boxes=[]
                 # 缩放到短边是321
                 if (box[3]-box[1])<(box[2]-box[0]):
                     scale=321*1.0/(box[3]-box[1])
                     # left
                     child_box1=[box[0], box[1],box[0]+int(box[3]-box[1]),box[3]]
                     #middle
                     child_box2=[int((box[0]+box[2])/2.0-(box[3]-box[1])/2.0), box[1],int((box[0]+box[2])/2.0+(box[3]-box[1])/2.0),box[3]]
                     #right
                     child_box3=[box[2]-int(box[3]-box[1]), box[1],box[2],box[3]]

                     # left - real_crop_size*real_crop_size
                     #real_crop_size=crop_size*1.0/scale
                     #child_box1=[box[0], int((box[1]+box[3])/2.0-real_crop_size/2.0),box[0]+int(real_crop_size),int((box[1]+box[3])/2.0+real_crop_size/2.0)]
                     #middle
                     #child_box2=[int((box[0]+box[2])/2.0-real_crop_size/2.0), int((box[1]+box[3])/2.0-real_crop_size/2.0),int((box[0]+box[2])/2.0+real_crop_size/2.0),int((box[1]+box[3])/2.0+real_crop_size/2.0)]
                     #right
                     #child_box3=[box[2]-int(real_crop_size),  int((box[1]+box[3])/2.0-real_crop_size/2.0),box[2],int((box[1]+box[3])/2.0+real_crop_size/2.0)]
                     
                     boxes.append(child_box1)
                     boxes.append(child_box2)
                     boxes.append(child_box3)
                 else:                    
                     scale=321*1.0/(box[2]-box[0])      
                     # top
                     child_box1=[box[0], box[1],box[2],box[1]+int(box[2]-box[0])]
                     #middle
                     child_box2=[box[0],int((box[1]+box[3])/2.0-(box[2]-box[0])/2.0),box[2],int((box[1]+box[3])/2.0+(box[2]-box[0])/2.0)]
                     #bottom
                     child_box3=[box[0],box[3]-int(box[2]-box[0]), box[2],box[3]]
                     
                     # top  - real_crop_size*real_crop_size
                     #real_crop_size=crop_size*1.0/scale
                     #child_box1=[int((box[0]+box[2])/2.0-real_crop_size/2.0), box[1],int((box[0]+box[2])/2.0+real_crop_size/2.0),box[1]+int(real_crop_size)]
                     #middle
                     #child_box2=[int((box[0]+box[2])/2.0-real_crop_size/2.0),int((box[1]+box[3])/2.0-(box[2]-box[0])/2.0),int((box[0]+box[2])/2.0+real_crop_size/2.0),int((box[1]+box[3])/2.0+(box[2]-box[0])/2.0)]
                     #bottom
                     #child_box3=[int((box[0]+box[2])/2.0-real_crop_size/2.0),box[3]-int(real_crop_size), int((box[0]+box[2])/2.0+real_crop_size/2.0),box[3]]
                     
                     boxes.append(child_box1)
                     boxes.append(child_box2)
                     boxes.append(child_box3)
                 
                 for boxid in range(3):
                     box=boxes[boxid]
                     # 缩放mask到scale尺寸
                     b_mask_in_box=lif2mask(obj_mark_filename, char_mark_folder, photo_folder, scale,box)
    
                     if b_mask_in_box!=True:
                         continue
                     
                     im_crop = imgs[box[1]:box[3],box[0]:box[2]]
                     #im_crop = tf.resize(im_crop,(int((box[3]-box[1])*scale),int((box[2]-box[0])*scale)))  
                     im_crop = tf.resize(im_crop,(crop_size,crop_size))  
                     save_filename = save_directory+unicode(type_name,"utf-8")+u'_'+photo_filename.split('/')[-1][:-4]+u'_crop'+str(i)+'_'+str(boxid)+u'.jpg'
    
                     io.imsave(save_filename,im_crop)     
                     
                     imgs_mask = io.imread(photo_filename[:-4]+'_mask.jpg',as_grey=False) 
                     
                     
                     im_mask_crop = imgs_mask[int(box[1]*scale):int(box[3]*scale),int(box[0]*scale):int(box[2]*scale)]
                     #im_mask_crop = tf.resize(im_mask_crop,(int((box[3]-box[1])*scale),int((box[2]-box[0])*scale)))  
                     im_mask_crop = tf.resize(im_mask_crop,(crop_size,crop_size))  
                     
                     save_filename = save_directory+unicode(type_name,"utf-8")+u'_'+photo_filename.split('/')[-1][:-4]+u'_crop'+str(i)+'_'+str(boxid)+u'.png'
    
                     io.imsave(save_filename,im_mask_crop)     
                 
                     new_imgs_mask = io.imread(save_filename,as_grey=True) 
#                             
#                     kernel = np.ones((20,20),np.uint8)
#                     erosion = cv2.erode(new_imgs_mask,kernel,iterations = 1)
#
                     new_imgs_mask=(new_imgs_mask>0)
        
                     io.imsave(save_filename,new_imgs_mask)  


                   
    
    return 0
    

# 直接生成ground truth box对应的图片和mask文件，不用缩放
def deeplab_fcn_samples_gen_simple(photo_folder, obj_mark_folder,char_mark_folder):
    
   

    #写入训练图像文件的列表，文件名 类别号
    save_directory = obj_mark_folder+u'gen_mask_samples/'
    if False==os.path.exists(save_directory):
        os.makedirs(save_directory) 
        
    # load目标类别列表
    type_filelist = obj_mark_folder+u'filelist.txt'
    try:
       type_names = np.loadtxt(type_filelist, str, delimiter='\t')
    except:   
       type_names = np.loadtxt(type_filelist, str, delimiter='\t')


    # load 每一类的图像列表
    for type_name in type_names:
        mark_filelist = obj_mark_folder+unicode(type_name,"utf-8")+u'/filelist.txt'
        try:
           mark_filenames = np.loadtxt(mark_filelist, str, delimiter='\t')
        except:   
           mark_filenames = np.loadtxt(mark_filelist, str, delimiter='\t')

        for mark_filename in mark_filenames:
            obj_mark_filename=mark_filename 
            
            photo_filename = photo_folder+mark_filename[:-4]
            print photo_filename
            
           
            mark_filename = obj_mark_folder+unicode(type_name,"utf-8")+u'/'+mark_filename
            
            # load mark files,获取图像中box的列表
            obj_box=[]
            with open(mark_filename) as f:    
                array = [[int(x) for x in line.split()] for line in f]
                
            # mark is empty line
            if len(array)<1:
                continue
            
            if len(array[0])<4:
                continue
                
            for i in range(len(array[0])/4):
                box=[]
                for j in range(4):
                   box.append(array[0][i*4+j])
                   
                left=min(box[0],box[2])   
                right=max(box[0],box[2]) 
                top=min(box[1],box[3]) 
                bottom=max(box[1],box[3]) 

                box[0]=left
                box[1]=top
                box[2]=right
                box[3]=bottom        			
        			
                obj_box.append(box)
             
            # photo not exist
            if False==os.path.exists(photo_filename):
                continue
            
            if False==os.path.exists(char_mark_folder+obj_mark_filename[:-4]+'.lif'):
                continue
            
           
            
            imgs = io.imread(photo_filename,as_grey=False) 
            
                     
           

            for box in obj_box:
    

                 # right小于left或者bottom小于top
                 if box[2] <box[0] or box[3] <box[1]:
                     continue
                 
                 if  np.maximum(box[2]-box[0], box[3]-box[1])<240:
                     continue
                 
                 b_mask_in_box=lif2mask(obj_mark_filename, char_mark_folder, photo_folder, 1.0,box)
    
                 if b_mask_in_box!=True:
                     continue
                     
                 im_crop = imgs[box[1]:box[3],box[0]:box[2]]
                     
                 save_filename = save_directory+unicode(type_name,"utf-8")+u'_'+photo_filename.split('/')[-1][:-4]+u'_crop'+str(i)+u'.jpg'
    
                 io.imsave(save_filename,im_crop)     
                     
                     
                 imgs_mask = io.imread(photo_filename[:-4]+'_mask.png',as_grey=False)     
                 im_mask_crop = imgs_mask[box[1]:box[3],box[0]:box[2]]                     
                 save_filename = save_directory+unicode(type_name,"utf-8")+u'_'+photo_filename.split('/')[-1][:-4]+u'_crop'+str(i)+u'.png'
    
                 io.imsave(save_filename,im_mask_crop)     
                 
#                 new_imgs_mask = io.imread(save_filename,as_grey=True)      


#                 new_imgs_mask=(new_imgs_mask>0)
        
#                 io.imsave(save_filename,new_imgs_mask)     

                   
    
    return 0
    

# 直接生成ground truth box对应的图片和mask文件，不用缩放
# 将textbox放在样本图片的中央，样本图片是500*500
def deeplab_fcn_samples_gen_textbox_center(photo_folder, obj_mark_folder,char_mark_folder):
 
    #写入训练图像文件的列表，文件名 类别号
    save_directory = obj_mark_folder+u'gen_mask_samples/'
    if False==os.path.exists(save_directory):
        os.makedirs(save_directory) 
        
    # load目标类别列表
    type_filelist = obj_mark_folder+u'filelist.txt'
    try:
       type_names = np.loadtxt(type_filelist, str, delimiter='\t')
    except:   
       type_names = np.loadtxt(type_filelist, str, delimiter='\t')


    # load 每一类的图像列表
    for type_name in type_names:
        mark_filelist = obj_mark_folder+unicode(type_name,"utf-8")+u'/filelist.txt'
        try:
           mark_filenames = np.loadtxt(mark_filelist, str, delimiter='\t')
        except:   
           mark_filenames = np.loadtxt(mark_filelist, str, delimiter='\t')

        for mark_filename in mark_filenames:
            obj_mark_filename=mark_filename 
            
            photo_filename = photo_folder+mark_filename[:-4]
            print photo_filename
            
           
            mark_filename = obj_mark_folder+unicode(type_name,"utf-8")+u'/'+mark_filename
            
            # load mark files,获取图像中box的列表
            obj_box=[]
            with open(mark_filename) as f:    
                array = [[int(x) for x in line.split()] for line in f]
                
            # mark is empty line
            if len(array)<1:
                continue
            
            if len(array[0])<4:
                continue
                
            for i in range(len(array[0])/4):
                box=[]
                for j in range(4):
                   box.append(array[0][i*4+j])
                   
                left=min(box[0],box[2])   
                right=max(box[0],box[2]) 
                top=min(box[1],box[3]) 
                bottom=max(box[1],box[3]) 

                box[0]=left
                box[1]=top
                box[2]=right
                box[3]=bottom        			
        			
                obj_box.append(box)
             
            # photo not exist
            if False==os.path.exists(photo_filename):
                continue
            
            if False==os.path.exists(char_mark_folder+obj_mark_filename[:-4]+'.lif'):
                continue
            
           
            
            imgs = io.imread(photo_filename,as_grey=False) 
            
                     
            samplesize=108

            for box in obj_box:
    
                
                 # right小于left或者bottom小于top
                 if box[2] <box[0] or box[3] <box[1]:
                     continue
                 
                 # 放在500*500的中央
                 xmiddle=(box[2]+box[0])*1.0/2
                 ymiddle=(box[3]+box[1])*1.0/2
                 
                 xx1 = np.maximum(0, xmiddle-samplesize/2)
                 yy1 = np.maximum(0, ymiddle-samplesize/2)
                 xx2 = np.minimum(imgs.shape[1]-1, xx1+samplesize)
                 yy2 = np.minimum(imgs.shape[0]-1, yy1+samplesize)
                 
                 xx1 = np.maximum(0, xx2-samplesize)
                 yy1 = np.maximum(0, yy2-samplesize)
                 
                 box[0]=xx1
                 box[1]=yy1
                 box[2]=xx2
                 box[3]=yy2
                 
                 
                 b_mask_in_box=lif2mask(obj_mark_filename, char_mark_folder, photo_folder, 1.0,box)
    
                 if b_mask_in_box!=True:
                     continue
                     
                 im_crop = imgs[box[1]:box[3],box[0]:box[2]]
                     
                 save_filename = save_directory+unicode(type_name,"utf-8")+u'_'+photo_filename.split('/')[-1][:-4]+u'_crop'+str(i)+u'.jpg'
    
                 io.imsave(save_filename,im_crop)     
                     
                     
                 imgs_mask = io.imread(photo_filename[:-4]+'_mask.png',as_grey=False)     
                 im_mask_crop = imgs_mask[box[1]:box[3],box[0]:box[2]]                     
                 save_filename = save_directory+unicode(type_name,"utf-8")+u'_'+photo_filename.split('/')[-1][:-4]+u'_crop'+str(i)+u'.png'
    
                 io.imsave(save_filename,im_mask_crop)     
                  
    
    return 0

# fix deeplab mask label - fix lable 2, only 0 and 1
def deeplab_fcn_samples_fix(mask_folder):
     # all files
    all_mask_files=os.listdir(mask_folder)  
    
    for mask_file in all_mask_files:
    
        if os.path.splitext(mask_file)[1] != '.png' :
            continue
        
        print mask_folder+mask_file
        
        imgs_mask = io.imread(mask_folder+mask_file,as_grey=False) 
        
        imgs_mask=(imgs_mask>0)
        
        io.imsave(mask_folder+mask_file,imgs_mask)     
   
#还需要增加1/N份作为testing样本的功能    
def deeplab_fcn_list_prepare(photo_folder):
     
    index_in_all=0
    
    #写入训练图像文件的列表，文件名 类别号
    save_directory = photo_folder+u'list/'
    if False==os.path.exists(save_directory):
        os.makedirs(save_directory)         
        
    file_object_train = codecs.open(save_directory+'train_list.txt', 'w','utf-8')
    file_object_val = codecs.open(save_directory+'val_list.txt', 'w','utf-8')
    file_object_test = codecs.open(save_directory+'test_list.txt', 'w','utf-8')
    
    
    # all files
    all_mark_files=os.listdir(photo_folder)  
    
    for obj_filename in all_mark_files:
    
        if os.path.splitext(obj_filename)[1] != '.png' :
            continue

   
            
        index_in_all=index_in_all+1
                    
        if 0==index_in_all%(int(1/testing_samples_ratio)):
            #file_object_test.write('/JPEGImages/'+obj_filename[:-4]+'.jpg '+'/SegmentationClassAug/'+obj_filename+u'\r\n')
            file_object_test.write(obj_filename[:-4]+u'\r\n')
        if 0==(index_in_all+5)%(int(1/testing_samples_ratio)):
            #file_object_val.write('/JPEGImages/'+obj_filename[:-4]+'.jpg '+'/SegmentationClassAug/'+obj_filename+u'\r\n')
            file_object_val.write(obj_filename[:-4]+u'\r\n')
        else:
            #file_object_train.write('/JPEGImages/'+obj_filename[:-4]+'.jpg '+'/SegmentationClassAug/'+obj_filename+u'\r\n')
            file_object_train.write(obj_filename[:-4]+u'\r\n')

                 
                    
                 
    #关闭文件
    file_object_train.close()     
    file_object_test.close()    
    file_object_val.close()    
                 


    return 0 
    
       
# 根据标记的结果自动填充汉字标记框里的汉字
def autofill_character_with_rcnnlabels(rcnn_mark_folder, char_mark_folder):
    
    #写入训练图像文件的列表，文件名 类别号
    save_directory = char_mark_folder+u'autofillchars/'
    if False==os.path.exists(save_directory):
        os.makedirs(save_directory)
        
    def format_shape(lbl,pts):
        return dict(label=lbl,
        				line_color=None,
        				fill_color=None,
        				points=pts)
         
    # all files
    all_char_mark_files=os.listdir(char_mark_folder)  
    
    for char_mark_filename in all_char_mark_files:
    
        if os.path.splitext(char_mark_filename)[1] != '.lif' :
            continue    
       

        print char_mark_folder+char_mark_filename
        
        #obj lif file
        obj_labelFile = LabelFile(rcnn_mark_folder+char_mark_filename)
        
        
                

        shapes=[]
        for shape in obj_labelFile.shapes:       
       
            
            p_x=[p[0] for p in shape[1]]
            p_y=[p[1] for p in shape[1]]  
          
            # coordination of out box
            left = int(np.min(p_x))
            top = int(np.min(p_y))
    
            right = int(np.max(p_x))
            bottom = int(np.max(p_y))   
            
            label = shape[0].split('_')[0] 
            
            #print label
            
            counti=0
            
            #char lif file
            char_labelFile = LabelFile(char_mark_folder+char_mark_filename)
            for char_shape in char_labelFile.shapes: 
                
                p_x=[p[0] for p in char_shape[1]]
                p_y=[p[1] for p in char_shape[1]]  
          
                # coordination of character
                left1 = int(np.min(p_x))
                top1 = int(np.min(p_y))
        
                right1 = int(np.max(p_x))
                bottom1 = int(np.max(p_y))   
           
                area = (right1 - left1 + 1) * (bottom1 - top1 + 1)
                    
                # 计算和gt里面char box的重合区域，如果完全重合就认为是这个out box里的字
                #找出窗口的最大的坐标位置，以及最小的坐标位置。
                xx1 = np.maximum(left, left1)
                yy1 = np.maximum(top, top1)
                xx2 = np.minimum(right, right1)
                yy2 = np.minimum(bottom, bottom1)
                     
                #计算窗口的长和宽
                w = np.maximum(0, xx2 - xx1 + 1)
                h = np.maximum(0, yy2 - yy1 + 1)
                     
                #计算重合率
                overlap = (w * h*1.0) / (area)
                
                
                #print overlap        
                # 完全重合就认为是这个out box里的字
                if overlap > 0.6 :
                   
                    if counti<len(label):
                        #print counti,label[counti],overlap
                        #time.sleep(3)
                        shapes.append(format_shape(label[counti],char_shape[1]))
                        counti=counti+1  

                        
        
        
        
       
          
                
        try:
        	char_labelFile.save(save_directory+char_mark_filename, shapes, char_labelFile.imagePath, char_labelFile.imageData,char_labelFile.lineColor, char_labelFile.fillColor)        	
        except :
        	continue                        
            
                
def textline_correction(photo_folder, obj_mark_folder,char_mark_folder):
     
    #写入训练图像文件的列表，文件名 类别号
    save_directory = obj_mark_folder+u'gen_mask_samples/'
    if False==os.path.exists(save_directory):
        os.makedirs(save_directory) 
        
    # load目标类别列表
    type_filelist = obj_mark_folder+u'filelist.txt'
    try:
       type_names = np.loadtxt(type_filelist, str, delimiter='\t')
    except:   
       type_names = np.loadtxt(type_filelist, str, delimiter='\t')


    # load 每一类的图像列表
    for type_name in type_names:
        mark_filelist = obj_mark_folder+unicode(type_name,"utf-8")+u'/filelist.txt'
        try:
           mark_filenames = np.loadtxt(mark_filelist, str, delimiter='\t')
        except:   
           mark_filenames = np.loadtxt(mark_filelist, str, delimiter='\t')

        for mark_filename in mark_filenames:
            obj_mark_filename=mark_filename 
            
            photo_filename = photo_folder+mark_filename[:-4]
            print photo_filename
            
           
            mark_filename = obj_mark_folder+unicode(type_name,"utf-8")+u'/'+mark_filename
            
            # load mark files,获取图像中box的列表
            obj_box=[]
            with open(mark_filename) as f:    
                array = [[int(x) for x in line.split()] for line in f]
                
            # mark is empty line
            if len(array)<1:
                continue
            
            if len(array[0])<4:
                continue
                
            for i in range(len(array[0])/4):
                box=[]
                for j in range(4):
                   box.append(array[0][i*4+j])
                   
                left=min(box[0],box[2])   
                right=max(box[0],box[2]) 
                top=min(box[1],box[3]) 
                bottom=max(box[1],box[3]) 

                box[0]=left
                box[1]=top
                box[2]=right
                box[3]=bottom        			
        			
                obj_box.append(box)
             
            # photo not exist
            if False==os.path.exists(photo_filename):
                continue
            
            if False==os.path.exists(char_mark_folder+obj_mark_filename[:-4]+'.lif'):
                continue
            
           
            
            imgs = io.imread(photo_filename,as_grey=False) 
            
                     
           

            for box in obj_box:
    

                 # right小于left或者bottom小于top
                 if box[2] <box[0] or box[3] <box[1]:
                     continue
                 
                 if  np.maximum(box[2]-box[0], box[3]-box[1])<240:
                     continue
                 
                 b_mask_in_box=lif2mask(obj_mark_filename, char_mark_folder, photo_folder, 1.0,box)
    
                 if b_mask_in_box!=True:
                     continue
                     
                 im_crop = imgs[box[1]:box[3],box[0]:box[2]]
                     
                 save_filename = save_directory+unicode(type_name,"utf-8")+u'_'+photo_filename.split('/')[-1][:-4]+u'_crop'+str(i)+u'.jpg'
    
                 io.imsave(save_filename,im_crop)     
                     
                     
                 imgs_mask = io.imread(photo_filename[:-4]+'_mask.png',as_grey=False)     
                 im_mask_crop = imgs_mask[box[1]:box[3],box[0]:box[2]]                     
                 save_filename = save_directory+unicode(type_name,"utf-8")+u'_'+photo_filename.split('/')[-1][:-4]+u'_crop'+str(i)+u'.png'
    
                 io.imsave(save_filename,im_mask_crop)  
                 
                 # 处理每个obj_box中文字mask区域最大的一块
                 
 
def char_label_lif2folders(photo_folder, char_mark_folder):
        
    
     #写入训练图像文件的列表，文件名 类别号
    save_directory = char_mark_folder+u'recog_samples/'
    if False==os.path.exists(save_directory):
        os.makedirs(save_directory) 
#    file_object_train = codecs.open(save_directory+'classifier_train_list.txt', 'w','utf-8')
#    file_object_test = codecs.open(save_directory+'classifier_test_list.txt', 'w','utf-8')
    
         
    # all files
    all_char_mark_files=os.listdir(char_mark_folder)  
    
    index_in_all=0
    for char_mark_filename in all_char_mark_files:
    
        if os.path.splitext(char_mark_filename)[1] != '.lif' :
            continue    
       
        photo_filename = photo_folder+char_mark_filename[:-4]
        print photo_filename

        imgs = io.imread(photo_filename,as_grey=False)      
        
        #obj lif file
        obj_labelFile = LabelFile(char_mark_folder+char_mark_filename)                       

       
        for shape in obj_labelFile.shapes:       
       
            
            p_x=[p[0] for p in shape[1]]
            p_y=[p[1] for p in shape[1]]  
          
            # coordination of out box
            left = int(np.min(p_x))
            top = int(np.min(p_y))
    
            right = int(np.max(p_x))
            bottom = int(np.max(p_y))   
            
            label = shape[0]
            
            
            # 新建字符对应目录，保存所有检测到的图像块
            save_directory1 = save_directory+label+u'/'
            if False==os.path.exists(save_directory1):
                os.makedirs(save_directory1)     
            
            recog_ext_offset_h = int((bottom-top)*recog_char_ext_ratio)
            recog_ext_offset_w = int((right-left)*recog_char_ext_ratio)
            
            if recog_ext_offset_h < 1:
                recog_ext_offset_h = 1
            if recog_ext_offset_w < 1:
                recog_ext_offset_w = 1
                
            recog_ext_stride_h = int(int(bottom-top)/recog_char_w)
            recog_ext_stride_w = int(int(right-left)/recog_char_w)
            
            if recog_ext_stride_h < 1:
                recog_ext_stride_h = 1
            if recog_ext_stride_w < 1:
                recog_ext_stride_w = 1
            
            im_crop = imgs[top:bottom,left:right]
            
            for y_slice in range(0,recog_ext_offset_h*2+1,recog_ext_stride_h):
                for x_slice in range(0,recog_ext_offset_w*2+1,recog_ext_stride_w):
                    # 裁剪图片
                    blob_img=im_crop[y_slice:(y_slice+(bottom-top)), x_slice:(x_slice+(right-left))]
                    
#                   scaled_blob_img= tf.resize(blob_img,(recog_char_w,recog_char_w))     
                
                  
            
                    save_filename = save_directory1+photo_filename.split('/')[-1][:-4]+u'_crop'+str(get_time_rand_num())+u'.jpg'           
            
#                   io.imsave(save_filename,scaled_blob_img) 
                    # 保留原始尺寸，在create——data，生成图像时统一resize
                    io.imsave(save_filename,blob_img) 
                    
                   
                    
                    index_in_all=index_in_all+1
                    
#                    if 0==index_in_all%(int(1/testing_samples_ratio)):
#                        file_object_test.write(save_filename[3:]+u' '+str(char2label(unicode(gt_line[6].strip(),"utf-8")))+u'\r\n')
#                    else:
#                        file_object_train.write(save_filename[3:]+u' '+str(char2label(unicode(gt_line[6].strip(),"utf-8")))+u'\r\n')
                 
    #关闭文件
#    file_object_train.close()     
#    file_object_test.close()     


    return 0
    

        
if __name__ == "__main__":    
    
#    img_url = photo_base_dir+u'fridge/image/image/BMT2/'
#    gt_lable_url = photo_base_dir+u'batchtran-utf8-nobom/batchtran/bmt_chn_labels/BMT2_chn_labels/'
    
#    img_url = photo_base_dir+u'fridge/image/image/BMT3/'
#    gt_lable_url = photo_base_dir+u'batchtran-utf8-nobom/batchtran/bmt_chn_labels/BMT3_chn_labels/'
    
#    img_url = photo_base_dir+u'test/train/test/'
#    gt_lable_url = photo_base_dir+u'batchtran-utf8-nobom/batchtran/test_chn_labels/test_chn_labels/'
    
#    img_url = photo_base_dir+u'unlabel/photo/'
##    img_url = photo_base_dir+u'ocr/gt_mark/det_samples~/'
#    gt_lable_url = photo_base_dir+u'unlabel/gt_mark/'
#    obj_mark_url = photo_base_dir+u'unlabel/mark/'
   
   # 输出多类标签
#    label_str=''
#    for n in range(len(chars)+1):
#        
#        label_str=label_str+'\''+'char_'+str(n-1)+'\','
#        
#        if 0==n%10:
#            print label_str
#            label_str=''
#    print label_str         
    
    img_url = photo_base_dir+u'ocr/photo/'
#    img_url = photo_base_dir+u'ocr/gt_mark/det_samples~/'
    gt_lable_url = photo_base_dir+u'ocr/gt_mark/'
    obj_mark_url = photo_base_dir+u'unlabel/box_gt_mark/'
    
        
#    det_data_prep_dir(img_url, gt_lable_url)
#    recog_data_prep_dir(img_url, gt_lable_url)

#    neg_trainimg_prep_dir_only_objmark(img_url,obj_mark_url)
#    neg_trainimg_prep_dir_objmark(img_url,gt_lable_url,obj_mark_url,0.1)

#    rcnn_recog_data_2xmls(img_url, obj_mark_url)

# 暂时不用neg_trainimg_prep_dir_no_objmark，全图检索太消耗
#    img_url = photo_base_dir+u'fridge/image/image/BMT2/'
#    lable_url = photo_base_dir+u'批量转换-utf8-nobom/批量转换/bmt汉字标记/BMT2汉字标记/'
#    neg_trainimg_prep_dir_no_objmark(img_url, lable_url,0.5)


#    mat_img_url = photo_base_dir+u'matlab-svt-data/detector_training_images/'
#    listfile_filename = photo_base_dir+u'matlab-svt-data/detector_train_list.txt'
#    mat_img_url = photo_base_dir+u'matlab-svt-data/detector_testing_images/'
#    listfile_filename = photo_base_dir+u'matlab-svt-data/detector_test_list.txt'
#    mat_det_trainimg_data_prep_dir(mat_img_url, listfile_filename)

#    mat_img_url = photo_base_dir+u'matlab-svt-data/classifier_testing_images/'
#    listfile_filename = photo_base_dir+u'matlab-svt-data/classifier_test_list.txt'
#    mat_img_url = photo_base_dir+u'matlab-svt-data/classifier_training_images/'
#    listfile_filename = photo_base_dir+u'matlab-svt-data/classifier_train_list.txt'
#    mat_recog_trainimg_data_prep_dir(mat_img_url, listfile_filename)


    
    img_url = 'D:/hwang/fridge-photos/Data2_training/training/photo/'
#    img_url = photo_base_dir+u'ocr/gt_mark/det_samples~/'
    gt_lable_url = 'D:/hwang/fridge-photos/Data2_training/training/photo/mark/'
    obj_mark_url = 'D:/hwang/fridge-photos/Data2_training/training/photo/mark/'
    
    img_url = 'D:/hwang/fridge-photos/Data2_training/training/photo/'
#    img_url = photo_base_dir+u'ocr/gt_mark/det_samples~/'
    gt_lable_url = 'D:/hwang/fridge-photos/Data2_training/training/photo/mark/'
    obj_mark_url = 'D:/hwang/fridge-photos/Data2_training/training/photo/mark/'
    
    img_url = 'D:/hwang/fridge-photos/Data2_training/training/photo/'
#    img_url = photo_base_dir+u'ocr/gt_mark/det_samples~/'
    gt_lable_url = 'D:/hwang/fridge-photos/Data2_training/training/photo/char-mark-lif-total/'
    obj_mark_url = 'D:/hwang/fridge-photos/Data2_training/training/photo/groundtruth/'
    
    #draw_box_with_objmarks(img_url,gt_lable_url,obj_mark_url)

#    out_char_list=find_all_chinese_chars(chars, gt_lable_url)
#    find_all_chinese_words(gt_lable_url, 'lexword__.txt')

#    listfile_filename=photo_base_dir+'ocr/gt_mark/neg_det_samples/1-neg_samples_selected/filelist.txt'
#    dir_to_det_trainimg_datas(listfile_filename, 1)

#    img_url = photo_base_dir+u'MSRA-TD500/train/'
#    gt_lable_url = photo_base_dir+u'MSRA-TD500/train/'

#    generate_MTR500_objs_photo(img_url,gt_lable_url)
    
    
#    img_url = photo_base_dir+u'MSRA-TD500/test/ext_samples_gt/'
#    gt_lable_url = photo_base_dir+u'MSRA-TD500/test/ext_samples_gt/'
#    img_url = photo_base_dir+u'MSRA-TD500/test/'
#    gt_lable_url = photo_base_dir+u'MSRA-TD500/test/'
#    generat_MTR500_objxmls(img_url,gt_lable_url)
    photo_folder=u'D:/fridge-photos/MSRA-TD500/total/'
    mark_folder=u'D:/fridge-photos/MSRA-TD500/total/'
    lif_folder=u'D:/fridge-photos/MSRA-TD500/total/ext_samples_gt/'
#    generate_MTR500_objs_2lif(photo_folder,mark_folder)
    #lif_data_2mask_scale(photo_folder, lif_folder)
	
    obj_mark_url = 'D:/hwang/fridge-photos/Data1_textRG_training_GroundTruth/'
#    objmark_wh_stat(obj_mark_url)
    
    format_marks_folder = 'E:/hwang/renderer/text_marks/'
    photo_folder='E:/hwang/renderer/photos/'
#    generate_brandname_productname_folders(format_marks_folder, photo_folder)
        
    img_url = u'E:/hwang/renderer/new_photos/'
    lif_lable_url=u'E:/hwang/renderer/new_text_marks/'
#    rcnn_lif_data_2xmls(img_url, lif_lable_url)
    
    xml_folder=u'E:/hwang/py-faster-rcnn-brandrecog/data/VOCdevkit2007/VOC2007/Annotations/'
#    process_xmls(xml_folder)
    a=1

    #marks_folder = 'D:/hwang/fridge-photos/Data2_test/test/photo/'
    #photo_folder='D:/hwang/fridge-photos/Data2_test/test/photo/'
    #marks_folder = u'D:/hwang/fridge-photos/Data2_training/training/photo/text-mark-lif/20160524-morning/'
    #photo_folder='D:/hwang/fridge-photos/Data2_training/training/photo/'
    marks_folder = 'D:/hwang/fridge-photos/Data2_test/test/photo/text-mark-lif/'
    photo_folder='D:/hwang/fridge-photos/Data2_test/test/photo/'
    #marks_folder = u'D:/hwang/fridge-photos/Data2_training/training/photo/text-mark-lif-total/'
    #generate_oem_categorize_folders(marks_folder, photo_folder)
    
    #marks_folder = u'D:/hwang/fridge-photos/Data2_training/training/photo/text-mark-lif-total/'
    #labels_folder = u'D:/hwang/fridge-photos/Data2_training/training/photo/text-mark-lif-total/categorized_folder/modify/'
    labels_folder = u'D:/hwang/fridge-photos/Data2_test/test/photo/text-mark-lif/categorized_folder/modify/'
    #lif_batch_modification(marks_folder,labels_folder, u'风味酸牛奶_光明')
    
    #marks_folder = u'D:/hwang/fridge-photos/Data2_training/training/photo/text-mark-lif-total/'
    marks_folder = u'D:/hwang/fridge-photos/Data2_test/test/photo/text-mark-lif/'
    rcnn_labels_folder='E:/hwang/pylabelme/rcnn-classes4train-20160526'
    #list_rcnn_labels_dir(rcnn_labels_folder)
    #find_not_in_labels(marks_folder,rcnn_labels_folder)
    
    #detailed_text_mark_lif_data_2xmls(photo_folder, marks_folder, rcnn_labels_folder)
    
    img_url = u'D:/hwang/fridge-photos/Data2_training/training/photo/'
    lif_lable_url=u'D:/hwang/fridge-photos/Data2_training/training/photo/char-mark-lif-total/'
    #lif_data_2mask(img_url, lif_lable_url)

    photo_folder = u'D:/hwang/fridge-photos/Data2_training/training/photo/'
    obj_mark_folder=u'D:/hwang/fridge-photos/Data2_training/training/photo/groundtruth/'
    textbox_mark_folder=u'D:/hwang/fridge-photos/Data2_training/training/photo/mark/'
    char_mark_folder=u'D:/hwang/fridge-photos/Data2_training/training/photo/char-mark-lif-total/'
    #deeplab_fcn_samples_gen_simple(photo_folder, obj_mark_folder,char_mark_folder)  
    #deeplab_fcn_samples_gen_textbox_center(photo_folder, textbox_mark_folder,char_mark_folder)  
    
    #photo_folder = u'E:/hwang/deeplab-fcn2-textnotext/data/VOCdevkit/VOC2012/SegmentationClassAug/'
    #photo_folder = u'E:/hwang/caffe-fcn/data/ocr/SegmentationClass/'
    #photo_folder = u'D:/deeplab-public-ver2_gitlocal/data/VOCdevkit/VOC2012/SegmentationClassAug/'
    photo_folder = u'D:/caffe_fcn_gitlocal/data/msra500/SegmentationClass/'

    #deeplab_fcn_list_prepare(photo_folder)
    #deeplab_fcn_samples_fix(photo_folder)

    rcnn_mark_folder = u'D:/hwang/fridge-photos/Data2_test/test/photo/text-mark-lif-total/'
    char_mark_folder=u'D:/hwang/fridge-photos/Data2_test/test/photo/char-mark-lif-total/'
    #autofill_character_with_rcnnlabels(rcnn_mark_folder, char_mark_folder)
    
    photo_folder=u'D:/hwang/fridge-photos/Data2_test_training/'
    char_mark_folder=u'D:/hwang/fridge-photos/Data2_training_test_character/'
    char_label_lif2folders(photo_folder, char_mark_folder)