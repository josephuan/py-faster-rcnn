# -*- coding: utf-8 -*-
# 图像都是行优先的，访问图像都按照，(y,x)坐标来访问
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
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

#model_define_det= model_path+'ocr_recog_casia_googleNet_deploy.prototxt'
#model_weight_det =model_path+'ocr_recog/ocr_recog_casia_googleNet_iter_350000.caffemodel'
#model_define_fullcon_det =model_path+'ocr_recog_casia_googleNet_deploy_fc.prototxt'
#model_weight_fullcon_det =model_path+'ocr_recog/ocr_recog_casia_googleNet_iter_350000_fc.caffemodel'

#model_define_det= model_path+'ocr_recog_casia_googleNet_cls3_pool_s1_deploy.prototxt'
#model_weight_det =model_path+'ocr_recog/ocr_recog_casia_googleNet_cls3_pool_s1_iter_155000.caffemodel'
#model_define_fullcon_det =model_path+'ocr_recog_casia_googleNet_cls3_pool_s1_deploy_fc.prototxt'
#model_weight_fullcon_det =model_path+'ocr_recog/ocr_recog_casia_googleNet_cls3_pool_s1_iter_155000_fc.caffemodel'

#model_define_det= model_path+'ocr_recog_casia_googleNet_64in_cls3_pool_s1_deploy.prototxt'
#model_weight_det =model_path+'ocr_recog/ocr_recog_casia_googleNet_64in_cls3_pool_s1_iter_45000.caffemodel'
#model_define_fullcon_det =model_path+'ocr_recog_casia_googleNet_64in_cls3_pool_s1_deploy_fc.prototxt'
#model_weight_fullcon_det =model_path+'ocr_recog/ocr_recog_casia_googleNet_64in_cls3_pool_s1_iter_45000_fc.caffemodel'

model_define_det= model_path+'ocr_recog_casia_googleNet_cls3_pool_s1_p2s1_deploy.prototxt'
model_weight_det =model_path+'ocr_recog/ocr_recog_casia_googleNet_cls3_pool_s1_p2s1_iter_155000.caffemodel'
model_define_fullcon_det =model_path+'ocr_recog_casia_googleNet_cls3_pool_s1_p2s1_deploy_fc.prototxt'
model_weight_fullcon_det =model_path+'ocr_recog/ocr_recog_casia_googleNet_cls3_pool_s1_p2s1_iter_155000_fc.caffemodel'

ocr_recog_mean_npy = model_path+'ocr_recog_casia_googlenet_mean.npy'
#ocr_recog_mean_npy = model_path+'ocr_recog_casia_googlenet_64in_mean.npy'

lexicon_filename = model_path+'lexword.txt'
labels_filename = model_path+'ocr_recog_synset_words.txt'

#输出的字符分类
#chars = u'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
#chars = u'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789味全优酪乳李錦記山楂枸杞风酸牛奶新希望蔓越莓蓝蒙大果粒蚝鲜油鸿光浪花杭州超值嫩火中華豆腐西湖老现代牧场卡夫蛋黄选芦荟天喜温馨汤锅亨氏番茄沙司茵腿片乌江高钙香榨菜稥蜜桃多畅缤纷纯爽丝明小龙女四季美汁源青岛啤酒饮品乐统一';
#chars = u'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789味全优酪乳李錦記山楂枸杞风酸牛奶新希望蔓越莓蓝蒙大果粒蚝鲜油鸿光浪花杭州超值嫩火中華豆腐西湖老现代牧场卡夫蛋黄选芦荟天喜温馨汤锅亨氏番茄沙司茵腿片静思微软亚洲研究院车库入口水表间讨论室办公电话工作台实验卫生消栓文印会议可使用警疏散图如遇情请按此处紧急打破玻璃开门擎许伟捷保持常关统一茶提示把瓶子送回房谢注意理箱密件柜日式沈为菊演厅丹棱街号停空闲位禁防止通道购物广地下剩余乐活出收费编海国机集团心东淀安勿留家自主创试范区核钢际银行民三善缘方教育科技械业满星早村麻辣诱惑眉坡酒楼剧南路铁便利交北京妇幼健儿童期发展免货巴比伦宾馆药店租驶信建投证券彩和坊惠寺书万典当苏售宝姿造型龙学校小时助服务二首都人才厦化爱者四环随分享快博管告牌最运吸烟津汇百步福食堆放品卷帘即将幕恒记甜加菲猫派克兰帝酷旗舰威迩明朗眼镜属于你我的刻装亮相滑凡蒂诺而森格冰淇淋霍顿美联亲仙踪林洗手奥特婴床佰草锦益高贝订针本传真码节约纸张就绪数据纯净商部储藏践踏青枯萎垃圾不触动未来华合差脾气禹单必有师鑫搬内石珠秋远悦莱寓座层侧燃危险航站线邮政筒埠外河深您六泉想走金凤成祥鄂尔多斯质推荐友灵感志愿盛招募类长城计划每点吃进钮受好营盖浇饭酱餐查询巾盒界农拉是说弱配梯屠卓普在腾飞哪里让英语要踩江赋专婚礼吴欧盈居玛娜前籍芙蓉价上共汽饮客屈臣诗碰庆周身年边识确鱼榨菜宠们供面给只具领取夏令脑十世纪算术斌非授权热姚聪遛狗护应避难所紫园精修看系市干鸭头奠基害嘉陵无障碍垂直设置往米压靠近维像采域严喷局铜雕社畅春雪芹画名住户绿邂逅艺再沸刷乌钙香稥蜜桃缤纷爽丝女季汁源岛啤';
chars = u'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789味全优酪乳李錦記山楂枸杞风酸牛奶新希望蔓越莓蓝蒙大果粒蚝鲜油鸿光浪花杭州超值嫩火中華豆腐西湖老现代牧场卡夫蛋黄选芦荟天喜温馨汤锅亨氏番茄沙司腿片乌江高钙香榨菜蜜桃多畅明缤纷纯爽丝小龙女四季美汁源青岛啤酒饮品乐统一挂面椰树荞麦酷儿口脆苹醋料不王吉尔士卤无雪碧悦肉精炖冰红茶坛包合五敢承仑春昆打菇加宝烧怡飘泉农辣正宗绍兴字号厚德载物上善若水升级这才梦梅酱歪桂井肠鸡古道双汇档泡铁观音拍枣咸康华年达日每橙原蒋去皮陈克梨博威百成吃佬长记溶力阿萨姆来易方午恒排冷哈娃黑椒餐罐头芬发酵趣恋胎菊勇闯涯招牌拌饭好滋养滨巨麻白释放夏零度可事冠益旺晶玉我爱仔更筋银倍伊利十运信赖见质先轻千葡萄生活如想'

channel = 1
raw_scale = 255.0
baseline_char_w = 32 #文字参考大小
det_char_w = 114 #114 #检测网络的input dim

stride = 7.981981981981981981981981981982 #检测网络转换为fcn后的步长,根据fcn网络或者w,win,new转换得到(w-win)/stride+1=neww
threshold = 0.99 #检测后proposal为char box的置信度门限

enable_show = True


map_idx = 0 #0，表示char，1表示non-char
params = ['cls3_fc1', 'cls3_fc2']
params_fc =  ['cls3_fc1-conv', 'cls3_fc2-conv']

# 检测用的原图resize尺度
#det_scales=[0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]
#det_scales=[0.4,0.5,0.6,0.7,0.8,0.9,1.0]
#det_scales=[0.5,0.6]
#det_scales=[2.0]
det_scales=[2.25] #0.5*114/32

def generateBoundingBox(featureMap, scale, label):
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
            x=x
            y=y   
            
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
                              float(stride * y + det_char_w - 1)/scale, float(stride * x + det_char_w - 1)/scale, prob, scale, label])
            
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
    
    net=net_fullcon_det


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
   
    true_boxes = []
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
        transformer.set_mean('data', np.load(ocr_recog_mean_npy).mean(1).mean(1))
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
        for id_char in range(len(chars)):
            total_boxes = []
            boxes = generateBoundingBox(out['prob'][0,id_char], scale, id_char)
            if(boxes):
                total_boxes.extend(boxes)

            if(total_boxes):
            #   非极大值抑制    
                boxes_nms = np.array(total_boxes)
                boxes_nms_max = nms_max(boxes_nms, overlapThresh=0.5)
                boxes_nms_average = nms_average(np.array(boxes_nms_max), overlapThresh=0.07)
                
                true_boxes.extend(boxes_nms_average);
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
        
        zhfont1 = matplotlib.font_manager.FontProperties(fname='C:/Anaconda/Lib/site-packages/matplotlib/mpl-data/fonts/ttf/msyh.ttf')

        if enable_show:
#       ##显示结果图用   
           rect = mpatches.Rectangle((box[1], box[0]), box[3]-box[1], box[2]-box[0],
               fill=False, edgecolor='red', linewidth=1)
#           ax.text(box[1], box[0]+20,"{0:.3f}".format(box[6]),color='white', fontsize=6)
#           aaa=ax.text(box[1], box[0]+20,chars[int(box[6])].decode('ascii'),color='white', fontsize=6,fontproperties=zhfont1)
           aaa=ax.text(box[1], box[0]+20,chars[int(box[6])],color='white', fontsize=10,fontproperties=zhfont1)
           aaa.get_fontname()
          
           ax.add_patch(rect)     

    if enable_show:   
#   ##显示结果图用      
        plt.savefig(model_path+'result/'+image_name.split('/')[-1])
        plt.close()
#   return out['prob'][0,map_idx]
    return true_boxes
    

if __name__ == "__main__":

    det_use_gpu=False   
       
    for i in range(1,8):
        image_name = model_path+'database/'+str(i+1)+'.jpg'
        print i
        detected_char_boxes = text_detection_image(image_name,det_use_gpu)                
        
        if enable_show:
            plt.close('all')
   