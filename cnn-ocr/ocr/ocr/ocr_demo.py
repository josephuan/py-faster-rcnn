# -*- coding: utf-8 -*-
# 图像都是行优先的，访问图像都按照，(y,x)坐标来访问
import cv2
import sys
import os
import skimage.io

from cnn_ocr import text_detection_image, text_recog_image

if __name__ == "__main__":       
    
    if len(sys.argv) < 2:
        print 'please input photo full path'

    image_name = sys.argv[1]

#    image_name = 'D_20151221_1063.jpg'
    if False==os.path.isfile(image_name):  
        print "photo file not exist!!!!"  
        sys.exit(0) ##正常退出  
    
    lable_name = image_name+u'.txt'

    if False==os.path.isfile(lable_name):  
        print "photo mark file not exist!!!!"
        sys.exit(0) ##正常退出    
    
    if len(sys.argv) > 2:
        det_use_gpu=int(sys.argv[2])
        recog_use_gpu=int(sys.argv[3])
    else:
        det_use_gpu=False
        recog_use_gpu=False

#    image_name = 'C_20160108_1921.jpg'
#    lable_name = image_name+u'.txt'
#
#    det_use_gpu=False
#    recog_use_gpu=False

    # load mark files,获取图像中box的列表
    obj_box=[]
    with open(lable_name) as f:    
        array = [[int(x) for x in line.split()] for line in f]
                
                
    # mark is empty line
    if len(array)<1:
        sys.exit(0) 
            
    if len(array[0])<4:
        sys.exit(0) 
                
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
       
        imgs = skimage.io.imread(image_name,as_grey=False)
        im_crop_s = imgs[box[1]:box[3],box[0]:box[2]]
        if im_crop_s.shape[0] == 0 or im_crop_s.shape[1] == 0:
           continue
        skimage.io.imsave('crop.jpg',im_crop_s)
        
#        img1 = cv2.imread('crop.jpg')
#        replicate = cv2.copyMakeBorder(img1,16,16,16,16,cv2.BORDER_REPLICATE)
#        cv2.imwrite('crop_rep.jpg',replicate)
        
        
        detected_char_boxes = text_detection_image('crop.jpg',det_use_gpu)
                
        pred_word = text_recog_image(detected_char_boxes,'crop.jpg',recog_use_gpu)

        print pred_word
       