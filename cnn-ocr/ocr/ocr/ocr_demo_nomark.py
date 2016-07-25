# -*- coding: utf-8 -*-
# 图像都是行优先的，访问图像都按照，(y,x)坐标来访问
import cv2
import sys
import os
import skimage.io

from cnn_ocr import text_detection_image, text_recog_image

from time import clock


if __name__ == "__main__":       
    
    if len(sys.argv) < 2:
        print 'please input photo full path'  

    image_name = sys.argv[1]    
    
#    image_name = 'D_20151221_1063.jpg'
    if False==os.path.isfile(unicode(image_name,'gb2312')):  
        print "photo file not exist!!!!"  
        sys.exit(0) ##正常退出  
    
    
    if len(sys.argv) > 2:
        det_use_gpu=int(sys.argv[2])
        recog_use_gpu=int(sys.argv[3])
    else:
        det_use_gpu=False
        recog_use_gpu=False   
        
        
    start=clock()
    
 
    detected_char_boxes = text_detection_image(unicode(image_name,'gb2312'),det_use_gpu)
    
    finish1=clock()
   
            
    pred_word = text_recog_image(detected_char_boxes,unicode(image_name,'gb2312'),recog_use_gpu)

    print pred_word
    
    finish2=clock()
       
    print 'det cost:'+str(finish1)
    print 'recog cost:'+str(finish2-finish1)