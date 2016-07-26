#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__', # always index 0
           'textmark1',
                         'textmark2',
                         'textmark3',
                         'textmark4',
                         'textmark5',
                         'textmark6',
                         'textmark7',
                         'textmark8',
                         'textmark9',
                         'textmark10',
                         'textmark11',
                         'textmark12',
                         'textmark13',
                         'textmark14',
                         'textmark15',
                         'textmark16',
                         'textmark17',
                         'textmark18',
                         'textmark19',
                         'textmark20',
                         'textmark21',
                         'textmark22',
                         'textmark23',
                         'textmark24',
                         'textmark25',
                         'textmark26',
                         'textmark27',
                         'textmark28',
                         'textmark29',
                         'textmark30',
                         'textmark31',
                         'textmark32',
                         'textmark33',
                         'textmark34',
                         'textmark35',
                         'textmark36',
                         'textmark37',
                         'textmark38',
                         'textmark39',
                         'textmark40',
                         'textmark41',
                         'textmark42',
                         'textmark43',
                         'textmark44',
                         'textmark45',
                         'textmark46',
                         'textmark47',
                         'textmark48',
                         'textmark49',
                         'textmark50',
                         'textmark51',
                         'textmark52',
                         'textmark53',
                         'textmark54',
                         'textmark55',
                         'textmark56',
                         'textmark57',
                         'textmark58',
                         'textmark59',
                         'textmark60',
                         'textmark61',
                         'textmark62',
                         'textmark63',
                         'textmark64',
                         'textmark65',
                         'textmark66',
                         'textmark67',
                         'textmark68',
                         'textmark69',
                         'textmark70',
                         'textmark71',
                         'textmark72',
                         'textmark73',
                         'textmark74',
                         'textmark75',
                         'textmark76',
                         'textmark77',
                         'textmark78',
                         'textmark79',
                         'textmark80',
                         'textmark81',
                         'textmark82',
                         'textmark83',
                         'textmark84',
                         'textmark85',
                         'textmark86',
                         'textmark87',
                         'textmark88',
                         'textmark89',
                         'textmark90',
                         'textmark91',
                         'textmark92',
                         'textmark93',
                         'textmark94',
                         'textmark95',
                         'textmark96',
                         'textmark97',
                         'textmark98',
                         'textmark99',
                         'textmark100',
                         'textmark101',
                         'textmark102',
                         'textmark103',
                         'textmark104',
                         'textmark105',
                         'textmark106',
                         'textmark107',
                         'textmark108',
                         'textmark109',
                         'textmark110',
                         'textmark111',
                         'textmark112',
                         'textmark113',
                         'textmark114',
                         'textmark115',
                         'textmark116',
                         'textmark117',
                         'textmark118'
						)

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
		'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                  'vgg_cnn_m_1024_faster_rcnn_iter_70000.caffemodel'),
        'zf': ('ZF',
                  'zf_faster_rcnn_iter_10000.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]

    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


 #   plt.savefig(image_name.split('/')[-1][:-4]+'_'+cls+'_res.jpg')
    plt.savefig('_res.jpg')
    

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
       
        vis_detections(im, cls, dets, thresh=CONF_THRESH)

        

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_end2end', 'test.prototxt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    im_names = ['K_20160510_CanBeer_t186.jpg']
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name)

    plt.show()
