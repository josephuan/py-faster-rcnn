# -*- coding: utf-8 -*-
# ͼ���������ȵģ�����ͼ�񶼰��գ�(y,x)����������
import numpy as np

import cv2


img = cv2.imread('star.jpg',0)
ret,thresh = cv2.threshold(img,127,255,0)
contours,hierarchy = cv2.findContours(thresh, 1, 2)

cnt = contours[0]