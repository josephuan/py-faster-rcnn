# -*- coding: utf-8 -*-
# 图像都是行优先的，访问图像都按照，(y,x)坐标来访问
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os
from  math import pow
import skimage.io
from skimage import transform as tf

import caffe

from nms import nms_average,nms_max

#============
#Model related:

model_path = '../../'
#model_path = './'
#model_define_det= model_path+'ocr_detector_deploy4eval.prototxt'
#model_weight_det =model_path+'ocr_detector/ocr_detector_iter_5000.caffemodel.h5'
#model_define_fullcon_det =model_path+'ocr_detector_deploy4eval_fc.prototxt'
#model_weight_fullcon_det =model_path+'ocr_detector/ocr_detector_iter_5000_fc.caffemodel'

#model_define_det= model_path+'ocr_detector_deploy4eval_p3s1.prototxt'
#model_weight_det =model_path+'ocr_detector/ocr_detector_iter_5000_p3s1.caffemodel.h5'
#model_define_fullcon_det =model_path+'ocr_detector_deploy4eval_p3s1_fc.prototxt'
#model_weight_fullcon_det =model_path+'ocr_detector/ocr_detector_iter_5000_p3s1_fc.caffemodel'

#model_define_det= model_path+'ocr_detector_deploy4eval_p3s1_p2s1.prototxt'
#model_weight_det =model_path+'ocr_detector/ocr_detector_iter_5000_p3s1_p2s1.caffemodel.h5'
#model_define_fullcon_det =model_path+'ocr_detector_deploy4eval_p3s1_p2s1_fc.prototxt'
#model_weight_fullcon_det =model_path+'ocr_detector/ocr_detector_iter_5000_p3s1_p2s1_fc.caffemodel'

model_define_det= model_path+'ocr_detector_full_deploy4eval_p3s1_p2s1.prototxt'
model_weight_det =model_path+'ocr_detector/ocr_detector_full_iter_70000_p3s1_p2s1.caffemodel.h5'
model_define_fullcon_det =model_path+'ocr_detector_full_deploy4eval_p3s1_p2s1_fc.prototxt'
model_weight_fullcon_det =model_path+'ocr_detector/ocr_detector_full_iter_70000_p3s1_p2s1_fc.caffemodel'

#model_define_det= model_path+'ocr_detector_deploy4eval_p3s1_p2s1_p1s1.prototxt'
#model_weight_det =model_path+'ocr_detector/ocr_detector_iter_5000_p3s1_p2s1_p1s1.caffemodel.h5'
#model_define_fullcon_det =model_path+'ocr_detector_deploy4eval_p3s1_p2s1_p1s1_fc.prototxt'
#model_weight_fullcon_det =model_path+'ocr_detector/ocr_detector_iter_5000_p3s1_p2s1_p1s1_fc.caffemodel'

#model_define_recog= model_path+'ocr_recog_deploy.prototxt'
#model_weight_recog =model_path+'ocr_recog/ocr_recog_iter_5000.caffemodel.h5'

#model_define_recog= model_path+'ocr_recog_full_deploy.prototxt'
#model_weight_recog =model_path+'ocr_recog/ocr_recog_full_iter_70000.caffemodel.h5'

model_define_recog= model_path+'ocr_recog_casia_googleNet_deploy.prototxt'
model_weight_recog =model_path+'ocr_recog/ocr_recog_casia_googleNet_iter_350000.caffemodel'

ocr_det_mean_npy = model_path+'ocr_detector_mean.npy'
#ocr_recog_mean_npy = model_path+'ocr_recog_mean.npy'
ocr_recog_mean_npy = model_path+'ocr_recog_casia_googlenet_mean.npy'

lexicon_filename = model_path+'lexword.txt'
labels_filename = model_path+'ocr_recog_synset_words.txt'

#输出的字符分类
#chars = u'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
#chars = u'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789味全优酪乳李錦記山楂枸杞风酸牛奶新希望蔓越莓蓝蒙大果粒蚝鲜油鸿光浪花杭州超值嫩火中華豆腐西湖老现代牧场卡夫蛋黄选芦荟天喜温馨汤锅亨氏番茄沙司茵腿片乌江高钙香榨菜稥蜜桃多畅缤纷纯爽丝明小龙女四季美汁源青岛啤酒饮品乐统一';
#chars = u'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789味全优酪乳李錦記山楂枸杞风酸牛奶新希望蔓越莓蓝蒙大果粒蚝鲜油鸿光浪花杭州超值嫩火中華豆腐西湖老现代牧场卡夫蛋黄选芦荟天喜温馨汤锅亨氏番茄沙司茵腿片静思微软亚洲研究院车库入口水表间讨论室办公电话工作台实验卫生消栓文印会议可使用警疏散图如遇情请按此处紧急打破玻璃开门擎许伟捷保持常关统一茶提示把瓶子送回房谢注意理箱密件柜日式沈为菊演厅丹棱街号停空闲位禁防止通道购物广地下剩余乐活出收费编海国机集团心东淀安勿留家自主创试范区核钢际银行民三善缘方教育科技械业满星早村麻辣诱惑眉坡酒楼剧南路铁便利交北京妇幼健儿童期发展免货巴比伦宾馆药店租驶信建投证券彩和坊惠寺书万典当苏售宝姿造型龙学校小时助服务二首都人才厦化爱者四环随分享快博管告牌最运吸烟津汇百步福食堆放品卷帘即将幕恒记甜加菲猫派克兰帝酷旗舰威迩明朗眼镜属于你我的刻装亮相滑凡蒂诺而森格冰淇淋霍顿美联亲仙踪林洗手奥特婴床佰草锦益高贝订针本传真码节约纸张就绪数据纯净商部储藏践踏青枯萎垃圾不触动未来华合差脾气禹单必有师鑫搬内石珠秋远悦莱寓座层侧燃危险航站线邮政筒埠外河深您六泉想走金凤成祥鄂尔多斯质推荐友灵感志愿盛招募类长城计划每点吃进钮受好营盖浇饭酱餐查询巾盒界农拉是说弱配梯屠卓普在腾飞哪里让英语要踩江赋专婚礼吴欧盈居玛娜前籍芙蓉价上共汽饮客屈臣诗碰庆周身年边识确鱼榨菜宠们供面给只具领取夏令脑十世纪算术斌非授权热姚聪遛狗护应避难所紫园精修看系市干鸭头奠基害嘉陵无障碍垂直设置往米压靠近维像采域严喷局铜雕社畅春雪芹画名住户绿邂逅艺再沸刷乌钙香稥蜜桃缤纷爽丝女季汁源岛啤';
chars = u'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789味全优酪乳李錦記山楂枸杞风酸牛奶新希望蔓越莓蓝蒙大果粒蚝鲜油鸿光浪花杭州超值嫩火中華豆腐西湖老现代牧场卡夫蛋黄选芦荟天喜温馨汤锅亨氏番茄沙司腿片乌江高钙香榨菜蜜桃多畅明缤纷纯爽丝小龙女四季美汁源青岛啤酒饮品乐统一挂面椰树荞麦酷儿口脆苹醋料不王吉尔士卤无雪碧悦肉精炖冰红茶坛包合五敢承仑春昆打菇加宝烧怡飘泉农辣正宗绍兴字号厚德载物上善若水升级这才梦梅酱歪桂井肠鸡古道双汇档泡铁观音拍枣咸康华年达日每橙原蒋去皮陈克梨博威百成吃佬长记溶力阿萨姆来易方午恒排冷哈娃黑椒餐罐头芬发酵趣恋胎菊勇闯涯招牌拌饭好滋养滨巨麻白释放夏零度可事冠益旺晶玉我爱仔更筋银倍伊利十运信赖见质先轻千葡萄生活如想'

channel = 1
raw_scale = 255.0
det_char_w = 32 #检测网络的input dim
recog_char_w = 114 #114 #识别网络的input dim
stride = 2 #检测网络转换为fcn后的步长,根据fcn网络或者w,win,new转换得到(w-win)/stride+1=neww
threshold = 0.99 #检测后proposal为char box的置信度门限

recog_ext_offset = 0 #12 #4 #图像扩展的边缘宽度，针对于recog_char_w窗口的大小来扩展
recog_ext_stride = 1 #6 #2

enable_show = True


map_idx = 0 #0，表示char，1表示non-char
#params = ['ip1', 'ip2']
#params_fc =  ['ip1-conv', 'ip2-conv']

params = ['ip1']
params_fc =  ['ip1-conv']

# 检测用的原图resize尺度
det_scales=[0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]
#det_scales=[0.4,0.5,0.6,0.7,0.8,0.9,1.0]
#det_scales=[0.5,0.6]
#det_scales=[2.0]

def generateBoundingBox(featureMap, scale):
    '''
    @brief: 生成窗口
    @param: featureMap,特征图，scale：尺度
    '''
    cols = featureMap.shape[1]
    rows = featureMap.shape[0]
    
    boundingBox = []
    for (y,x), prob in np.ndenumerate(featureMap):
       if(prob >= threshold):
           #映射到原始的图像中的大小
            x=x-1
            y=y-1   
            
            if x<0:
                x=0
            if y<0:
                y=0
            
            
            # 避免在边界的位置，这样不能拓展方框
#            if float(stride * x)<recog_ext_offset:
#                continue
#            if float(stride * y)<recog_ext_offset:
#                continue
#            
#            if float(stride * x)>cols-recog_ext_offset:
#                continue
#            if float(stride * y)>rows-recog_ext_offset:
#                continue
            
            boundingBox.append([float(stride * y)/scale, float(stride *x )/scale, 
                              float(stride * y + det_char_w - 1)/scale, float(stride * x + det_char_w - 1)/scale, prob, scale])
            
    return boundingBox

def convert_full_conv(model_define,model_weight,model_define_fc,model_weight_fc):
    '''
    @breif : 将原始网络转换为全卷积模型
    @param: model_define,二分类网络定义文件
    @param: model_weight，二分类网络训练好的参数
    @param: model_define_fc,生成的全卷积网络定义文件
    @param: model_weight_fc，转化好的全卷积网络的参数
    '''
    net = caffe.Net(model_define, model_weight, caffe.TEST)
    fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}
    net_fc = caffe.Net(model_define_fc, model_weight, caffe.TEST)
    conv_params = {pr: (net_fc.params[pr][0].data, net_fc.params[pr][1].data) for pr in params_fc}
    for pr, pr_conv in zip(params, params_fc):
       conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
       conv_params[pr_conv][1][...] = fc_params[pr][1]
    net_fc.save(model_weight_fc)
    print 'convert done!'
    return net_fc
#
def re_verify(net_vf, img):
    '''
    @breif: 对检测到的目标框进行重新的验证
    '''
    grayimg= tf.resize(img,(det_char_w,det_char_w))
    scale_img = np.reshape(grayimg, (det_char_w,det_char_w,channel))
    
    #更改网络输入data图像的大小
    net_vf.blobs['data'].reshape(1,channel,det_char_w,det_char_w)
        
    transformer = caffe.io.Transformer({'data': net_vf.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(model_path+'ocr_detector_mean.npy').mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
#    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', raw_scale)
    out = net_vf.forward_all(data=np.asarray([transformer.preprocess('data', scale_img)]))
    #print out['prob']
    if out['prob'][0,map_idx] > 0.8:
        return True
    else:
        return False
      
def prob_large_than_opt(prob,thre):
    idx = []
    for kx, item in np.ndenumerate(prob):
        if item > thre:
            idx.append(kx)

    return idx

def prob_small_than_opt(prob,thre):
    idx = []
    for kx, item in np.ndenumerate(prob):
        if item < thre:
            idx.append(kx)

    return idx

def char2label(in_char):

    label = chars.find(in_char)    

    return label

# input scale to make box-size as 32/scale back to original image
def score2word(scores, boxes, imh, imw, scale):

    # 转换为大小写不敏感
    itemcomp=scores[0:26,:]>scores[26:52,:]
    newscore1=scores[0:26,:]*itemcomp+scores[26:52,:]*(1-itemcomp)

    char_num=scores.shape[0]

    case_insen_scores=[]
    newscore2=scores[52:char_num+1,:];
    case_insen_scores=np.vstack((newscore1,newscore1,newscore2))   
    
    # load lexicon_word   
#    try:
#       lexicon_words = np.loadtxt(lexicon_filename, str, delimiter='\t')
#    except:   
#       lexicon_words = np.loadtxt(lexicon_filename, str, delimiter='\t')
#
#    lexicon_words_num = lexicon_words.shape[0]
    
    with open(lexicon_filename) as f:
        lexword_lines = [[unicode(str(x),'utf-8') for x in line.strip().split('\t')] for line in f]
        
    lexicon_words=[]
    
    for word in lexword_lines:           
        lexicon_words.append(word[0]) 
                
    lexicon_words_num = len(lexicon_words)
    
    #given sliding window classifier scores, predict the word label using
    # a Viterbi - style alignment algorithm.
    predict_word = u''
    max_matchscore = -100.0
    matchscore_thresh = 0.7 # 一个词里的至少3/4的字要相同

    for wordi in range(0,lexicon_words_num):
#        lexicon_word=unicode(lexicon_words[wordi], "utf-8")
        lexicon_word=lexicon_words[wordi]
        lexicon_word_labels=[]
        for i in range(0,len(lexicon_word)):
            lexicon_word_labels.append(char2label(lexicon_word[i]))

        w=len(lexicon_word_labels)
        s=case_insen_scores.shape[1] 
        
        if s<w :
            continue      

        scoreMat=np.zeros((w,s))
        scoreIdx = np.zeros((w,s))

        scoreMat[0,:]  = case_insen_scores[lexicon_word_labels[0],:] # initialize first row


        # Viterbi dynamic programming
        for i in range(1,w):
            for j in range(i,s):
                maxPrev = np.max(scoreMat[i-1, i-1:j],0)
                maxPrevIdx = np.argmax(scoreMat[i-1, i-1:j],0);
                scoreMat[i,j] = case_insen_scores[lexicon_word_labels[i], j] + maxPrev;

                scoreIdx[i,j] = maxPrevIdx;    

        matchscore = np.max(scoreMat[scoreMat.shape[0]-1,w-1:scoreMat.shape[1]],0);
        lastidx = np.argmax(scoreMat[scoreMat.shape[0]-1,w-1:scoreMat.shape[1]],0);
   
        real_good_idx = np.zeros((1,w));
        real_good_idx[0,w-1] = lastidx+w-1;

        i = w-1;
        # backtrace to find correspondence between peaks and chars.
        while i>0:
            real_good_idx[0,i-1] = scoreIdx[i, real_good_idx[0,i]]+i-1;
            i = i-1;   

        # 修改为根据2维坐标的实际gap
        gaps = np.zeros((1, w + 1))
        
        #计算block distance作为gap
        tmp1_2d = np.zeros((2, w + 1))
        tmp2_2d = np.zeros((2, w + 1));
       
        for i in range(0,w):
            tmp1_2d[0,i] = boxes[np.long(real_good_idx[0,i])][0];
            tmp1_2d[1,i] = boxes[np.long(real_good_idx[0,i])][1];

            tmp2_2d[0,i+1] = boxes[np.long(real_good_idx[0,i])][0];
            tmp2_2d[1,i+1] = boxes[np.long(real_good_idx[0,i])][1];

	    # 置为图片最右下角的点
        tmp1_2d[0, w] = imh-1;
        tmp1_2d[1, w] = imw-1;

        tmp2_2d[0, 0] = 0;
        tmp2_2d[1, 0] = 0;

	    # gaps = tmp1 - tmp2;
        gaps1 = np.absolute(tmp1_2d[0,:]-tmp2_2d[0,:])
        gaps2 = np.absolute(tmp1_2d[1,:]-tmp2_2d[1,:])

        gaps = gaps1+gaps2
        
	    # 计算字符间的距离的方差，如果大就会惩罚大
	    # penalize geometric inconsistency
        c_std = 0.08;
       
        # inconsistent character spacing
        if gaps.shape[0] >= 4 :
            tmp = gaps[1:w]
            std_loss = c_std*tmp.std()
        elif gaps.shape[0] >= 3:
            #处理两个字的词，认为理想的gap距离（block距离）为1个字符宽度,即32
            tmp = np.zeros((1, 2))
            tmp[0, 0] = gaps[1]
            tmp[0, 1] = 32/scale
            
            std_loss = c_std*tmp.std()
        else :
            std_loss = 100;
            


        #由于是各个零散的box串起来进行匹配，不是在一个定位好的text box中进行
	    #所以不计算边界距离的惩罚值在内
#        matchscore1 = matchscore - std_loss;
        matchscore1 = matchscore;

        if matchscore1>max_matchscore and matchscore/len(lexicon_word)>matchscore_thresh :
            max_matchscore = matchscore1;
            predict_word = lexicon_word;

    return predict_word+str(max_matchscore)



#输入img为所有原始box区域图像
#oimgs 原始图像
#net_char_recog 为recog网络
# detected_char_boxes为对应检测得到proposals位置boxes
def text_recog_image(detected_char_boxes,image_name,use_gpu):

    if True==use_gpu:
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_device(0)
        caffe.set_mode_cpu()

    # 所有proposal box个数
    nboxes=len(detected_char_boxes)

    if nboxes<1:
        return ''


    '''
    @breif: 检测所有proposal位置的文字，并score2word得到词组  
    '''
    net_recog = caffe.Net(model_define_recog, model_weight_recog, caffe.TEST)

    net_char_recog=net_recog

    oimg = skimage.io.imread(image_name,as_grey=True)
    if oimg.ndim==3:
        rows,cols,ch = oimg.shape
    else:
        rows,cols = oimg.shape

    # load labels    
#    try:
#       labels = np.loadtxt(labels_filename, str, delimiter='\t')
#    except:   
#       labels = np.loadtxt(labels_filename, str, delimiter='\t')
       
    labels=np.asarray(list(chars))

    #更改网络输入data图像的大小
    net_char_recog.blobs['data'].reshape(1,channel,recog_char_w,recog_char_w)
    transformer = caffe.io.Transformer({'data': net_char_recog.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(ocr_recog_mean_npy).mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
#    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', raw_scale)
  

     # 输出字符个数
    out_char_num=net_char_recog.blobs['prob'].data.shape[1] 

    # 扩展后得到的recog boxes所有recog blobs的计数器
    blob_it=0
    
    # 记录原始图像上的坐标点，输入score2word
    anchor_boxes=[]    
  

    net_char_recog.blobs['data'].reshape(nboxes,channel,recog_char_w,recog_char_w)
    for box in detected_char_boxes:
                       
        pos_box=box[0:4]
               
        top0 = pos_box[0]
        left0 = pos_box[1]
        
        bottom0 = pos_box[2]
        right0 = pos_box[3]      

        im_crop = oimg[int(top0):int(bottom0),int(left0):int(right0)] 
    
        # 记录原始图像上的坐标点，输入score2word
        anchor_boxes.append(box[0:4])

       
        blob_img= tf.resize(im_crop,(recog_char_w,recog_char_w)) 
        blob_img = np.reshape(blob_img, (recog_char_w,recog_char_w,channel)) 

        # 保存每张图片
#        skimage.io.imsave(model_path+'result/'+image_name.split('/')[-1][:-4]+'_'+str(blob_it)+'.jpg',blob_img)

        # 数据输入，构成多个batches，一次性做recog forward
        net_char_recog.blobs['data'].data[blob_it] = transformer.preprocess('data', blob_img)                    
        blob_it=blob_it+1               
     
    #一次性做recog forward                                                  
    out = net_char_recog.forward() 

    # 记录原始图像上的坐标点对应的scores，输入score2word
    scores = np.zeros((out_char_num,nboxes));    
    
    # 取出每个proposal所有的ext_boxes的prob
    iscore=0
    for i in range(nboxes):

        outp_per_proposal = out['prob'][i]
        predicts = outp_per_proposal
    
          
        # 取最大的概率分布为最终结果
#        predict = predicts.argmax()        
    
        top_k = predicts.flatten().argsort()[-1:-2:-1]
    
        # 对于该区域所有位置的检测结果均值，要判断大于0.5门限
        idx = prob_large_than_opt(predicts[top_k],0.5)
        if len(idx) > 0:
#            c = unicode(labels[top_k][idx], "gb2312")   
#            print c
            # 记录每个proposal的结果到scores
            scores[:,iscore] = predicts
            iscore=iscore+1
            print labels[top_k][0], predicts[top_k][0]

    scores_crop=np.array(scores[:,0:iscore])

    # 字间距std_loss暂时不用，scale任意，=1.0
    predword=''
    predword=score2word(scores_crop, anchor_boxes, rows, cols, 1.0)
    
    return predword

#输入img为所有proposal区域扩展图进行hstack后的大图
#oimgs 原始图像
#net_char_recog 为recog网络
# detected_char_boxes为对应检测得到proposals位置boxes
def text_recog_image_ext(detected_char_boxes,image_name,use_gpu):

    if True==use_gpu:
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_device(0)
        caffe.set_mode_cpu()

    # 所有proposal box个数
    nboxes=len(detected_char_boxes)

    if nboxes<1:
        return ''


    '''
    @breif: 检测所有proposal位置的文字，并score2word得到词组  
    '''
    net_recog = caffe.Net(model_define_recog, model_weight_recog, caffe.TEST)

    net_char_recog=net_recog

    oimg = skimage.io.imread(image_name,as_grey=True)
    if oimg.ndim==3:
        rows,cols,ch = oimg.shape
    else:
        rows,cols = oimg.shape

    # load labels    
#    try:
#       labels = np.loadtxt(labels_filename, str, delimiter='\t')
#    except:   
#       labels = np.loadtxt(labels_filename, str, delimiter='\t')
       
    labels=np.asarray(list(chars))

    #更改网络输入data图像的大小
    net_char_recog.blobs['data'].reshape(1,channel,recog_char_w,recog_char_w)
    transformer = caffe.io.Transformer({'data': net_char_recog.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(ocr_recog_mean_npy).mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
#    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', raw_scale)
  

     # 输出字符个数
    out_char_num=net_char_recog.blobs['prob'].data.shape[1] 

    # 扩展后得到的recog boxes所有recog blobs的计数器
    blob_it=0
    
    # 记录原始图像上的坐标点，输入score2word
    anchor_boxes=[]    

    # 一个proposal box扩展后得到的recog boxes
    n_extblobs_of_box=len(range(0,recog_ext_offset*2+1,recog_ext_stride))
    n_extblobs_of_box=n_extblobs_of_box*n_extblobs_of_box
    
   

    net_char_recog.blobs['data'].reshape(nboxes*n_extblobs_of_box,channel,recog_char_w114,recog_char_w114)
        
    for box in detected_char_boxes:
        scale = box[5]*recog_char_w/det_char_w

        #对于recog任务，计算原图应该resize的比率,相对于det任务依据窗口大小同等resize
        w,h = int(rows* scale),int(cols* scale)
        if w<recog_char_w or h<recog_char_w:
            continue
                  
        scaled_gray_oimg= tf.resize(oimg,(w,h))

        # 在scale后的原图上裁剪出recog_char_w+2*recog_ext_offset的待识别扩展图
        pos_box=box[0:4]*scale
               
        top0 = pos_box[0] - recog_ext_offset
        left0 = pos_box[1] - recog_ext_offset
        if top0 < 0:
            top0 = 0
        if left0 < 0:
            left0 = 0
            
        bottom0 = top0 + recog_char_w+recog_ext_offset
        right0 = left0 + recog_char_w+recog_ext_offset
       
        if bottom0 > w:
            bottom0 = w
        if right0 > h:
            right0 = h

        im_crop = scaled_gray_oimg[top0:bottom0,left0:right0]   
    
        # 记录原始图像上的坐标点，输入score2word
        anchor_boxes.append(box[0:4])

        for y_slice in range(0,recog_ext_offset*2+1,recog_ext_stride):
            for x_slice in range(0,recog_ext_offset*2+1,recog_ext_stride):
                
                # 裁剪图片
                blob_img=im_crop[y_slice:(y_slice+recog_char_w), x_slice:(x_slice+recog_char_w)]
                blob_img= tf.resize(blob_img,(recog_char_w114,recog_char_w114))
                blob_img = np.reshape(blob_img, (recog_char_w114,recog_char_w114,channel))  
                
                # 保存每张图片
#                skimage.io.imsave(model_path+'result/'+image_name.split('/')[-1][:-4]+'_'+str(blob_it)+'.jpg',blob_img)

                # 数据输入，构成多个batches，一次性做recog forward
                net_char_recog.blobs['data'].data[blob_it] = transformer.preprocess('data', blob_img)                    
                blob_it=blob_it+1
                
                
     
    #一次性做recog forward                                                  
    out = net_char_recog.forward() 

    # 记录原始图像上的坐标点对应的scores，输入score2word
    scores = np.zeros((out_char_num,nboxes));    
    
    # 取出每个proposal所有的ext_boxes的prob
    iscore=0
    for i in range(nboxes):
        outp_per_proposal = out['prob'][i*n_extblobs_of_box:(i+1)*n_extblobs_of_box]
               
        predicts = np.mean(outp_per_proposal, axis=0)
    
        # 记录每个proposal的结果到scores
#        scores[:,i] = np.mean(outp_per_proposal, axis=0)
           
        # 取最大的概率分布为最终结果
#        predict = predicts.argmax()        
    
        top_k = predicts.flatten().argsort()[-1:-2:-1]
    
        # 对于该区域所有位置的检测结果均值，要判断大于0.8门限
        idx = prob_large_than_opt(predicts[top_k],0.9)
        if 1:#len(idx) > 0:
#            c = unicode(labels[top_k][idx], "gb2312")   
#            print c
            # 记录每个proposal的结果到scores
            scores[:,iscore] = np.mean(outp_per_proposal, axis=0)
            iscore=iscore+1
            print labels[top_k][0]

        scores_crop=np.array(scores[:,0:iscore])
    
#       c = unicode(labels[top_k], "gb2312")  
#       print c
        
        
        '''
#       每一次分类的概率分布叠加
        for j in range(n_extblobs_of_box):
            idx = prob_small_than_opt(outp_per_proposal[j],0.8)
            outp_per_proposal[j,idx]=0.0    

        #对每个proposal扩展的区域的每一个blob的输出topk
        print "next batch"
        for j in range(n_extblobs_of_box):            
            top_k = outp_per_proposal[j].flatten().argsort()[-1:-6:-1]
            idx = prob_large_than_opt(outp_per_proposal[j,top_k],0.8)
            c = unicode(labels[top_k][idx], "gb2312")
            if len(idx) > 0:
                print c
        '''

    # 字间距std_loss暂时不用，scale任意，=1.0
    predword=''
    predword=score2word(scores_crop, anchor_boxes, rows, cols, 1.0)
    
    return predword


def text_detection_image(image_name,use_gpu):

    if True==use_gpu:
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_device(0)
        caffe.set_mode_cpu()

    '''
    @检测单张文字图像
    '''
    if not os.path.isfile(model_weight_fullcon_det):
        net_fullcon_det = convert_full_conv(model_define_det,model_weight_det,model_define_fullcon_det,model_weight_fullcon_det)
    else:
        net_fullcon_det = caffe.Net(model_define_fullcon_det, model_weight_fullcon_det, caffe.TEST)
    net_verify_det = caffe.Net(model_define_det, model_weight_det, caffe.TEST)

    net=net_fullcon_det
    net_vf=net_verify_det

    imgs = skimage.io.imread(image_name,as_grey=True)
    if imgs.ndim==3:
            rows,cols,ch = imgs.shape
    else:
            rows,cols = imgs.shape      
            
    #=========================
    if enable_show:
#   ## 显示热图用
        num_scale = len(det_scales)
        s1=int(np.sqrt(num_scale))+1
        tt=1
        plt.subplot(s1, s1+1, tt)
        plt.axis('off')
        plt.title("Input Image")
        im=caffe.io.load_image(image_name)
        plt.imshow(im)
    #============
    total_boxes = []
    for scale in det_scales:

        w,h = int(rows* scale),int(cols* scale)
        
        if w<det_char_w or h<det_char_w:
            continue
        
        scaled_gray_img= tf.resize(imgs,(w,h))        
        scaled_gray_img = np.reshape(scaled_gray_img, (w,h,channel))
    
        
        #更改网络输入data图像的大小
        net.blobs['data'].reshape(1,channel,w,h)
        #转换结构
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_mean('data', np.load(ocr_det_mean_npy).mean(1).mean(1))
        transformer.set_transpose('data', (2,0,1))
#       transformer.set_channel_swap('data', (2,1,0))
        transformer.set_raw_scale('data', raw_scale)
#       前馈一次
        out = net.forward_all(data=np.asarray([transformer.preprocess('data', scaled_gray_img)]))
        
        if enable_show:
#       ## 显示热图用
           tt=tt+1
           plt.subplot(s1, s1+1, tt)
           plt.axis('off')
           plt.title("sacle: "+ "%.2f" %scale)
           plt.imshow(out['prob'][0,map_idx])
        #===========
        boxes = generateBoundingBox(out['prob'][0,map_idx], scale)
        if(boxes):
            total_boxes.extend(boxes)

#   非极大值抑制    
    boxes_nms = np.array(total_boxes)
    boxes_nms_max = nms_max(boxes_nms, overlapThresh=0.5)
    boxes_nms_average = nms_average(np.array(boxes_nms_max), overlapThresh=0.07)
    
    true_boxes=boxes_nms_max;
    #===================
    if enable_show:
#   ## 显示结果图用
        plt.savefig(model_path+'heatmap/'+image_name.split('/')[-1])
#   在图像中画出检测到的文字框
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        ax.imshow(im)
  
    for i in range(0,len(true_boxes)):

        box=true_boxes[i]

        if box[0] <0 or box[1] <0 or box[2] >=rows  or box[3] >=cols:
            continue
        im_crop = imgs[box[0]:box[2],box[1]:box[3]]
        if im_crop.shape[0] == 0 or im_crop.shape[1] == 0:
            continue
#       if re_verify(net_vf, im_crop) == True:
        if enable_show:
#       ##显示结果图用   
           rect = mpatches.Rectangle((box[1], box[0]), box[3]-box[1], box[2]-box[0],
               fill=False, edgecolor='red', linewidth=1)
           ax.text(box[1], box[0]+20,"{0:.3f}".format(box[4]),color='white', fontsize=6)
           ax.add_patch(rect)     

    if enable_show:   
#   ##显示结果图用      
        plt.savefig(model_path+'result/'+image_name.split('/')[-1])
        plt.close()
#   return out['prob'][0,map_idx]
    return true_boxes
    

if __name__ == "__main__":

    det_use_gpu=False
    recog_use_gpu=True
       
    for i in range(1,2):
        image_name = model_path+'database/'+str(i+1)+'.jpg'
        print i
        detected_char_boxes = text_detection_image(image_name,det_use_gpu)
                
        pred_word = text_recog_image(detected_char_boxes,image_name,recog_use_gpu)

        print pred_word

        if enable_show:
            plt.close('all')
   