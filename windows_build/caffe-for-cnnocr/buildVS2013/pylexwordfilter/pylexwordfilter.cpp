#include <Python.h>  // NOLINT(build/include_alpha)

// Produce deprecation warnings (needs to come before arrayobject.h inclusion).
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <boost/make_shared.hpp>
#include <boost/python.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <numpy/arrayobject.h>

// these need to be included after boost on OS X
#include <string>  // NOLINT(build/include_order)
#include <vector>  // NOLINT(build/include_order)
#include <fstream>  // NOLINT

#include "caffe/caffe.hpp"
#include "caffe/layers/memory_data_layer.hpp"
#include "caffe/layers/python_layer.hpp"
#include "caffe/sgd_solvers.hpp"

#include "opencv2/text.hpp"
#include "opencv2/text/erfilter.hpp"
#include "opencv2/text/ocr.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/mat.inl.hpp"

#include <time.h>

#include <mat.h>


#include <string>
#include <vector>
#include <iostream>


#include <functional>
#include <algorithm>
#include <cstdlib>
#include <numeric>

using namespace cv;
using namespace std;
using namespace cv::text;
using namespace cv::ml;

// Temporary solution for numpy < 1.7 versions: old macro, no promises.
// You're strongly advised to upgrade to >= 1.7.
#ifndef NPY_ARRAY_C_CONTIGUOUS
#define NPY_ARRAY_C_CONTIGUOUS NPY_C_CONTIGUOUS
#define PyArray_SetBaseObject(arr, x) (PyArray_BASE(arr) = (x))
#endif

#define CHAR_NUMBER 719 // 识别的目标字符的个数

void filterScores(Mat &s, int NMS_RADIUS, double marginThresh,
	std::vector<double> &maxvalvector, std::vector<int> maxidxvector, std::vector<int> &max_idx);
void wtnms(Mat &s, int rad, std::vector<int> &output_I);
void ascii2label(wstring y, vector<double> &lable);

wstring Score2Word_of_Boxes(Mat &score, Mat &boxpos, int boxnum, int w, int h, std::vector<wstring> &Lex);
void getMatchScore_of_Boxes(Mat &origscores, Mat &boxpos, int w, int h, wstring origword, vector<int> &good_idx,
	double &matchscore, Mat &real_good_idx);

void ascii2label(wstring y, vector<double> &lable)
{
#if 0
	for (int i = 0; i < y.length(); i++)
	{
		if (y[i] >= 97) // lowercase 27 - 52
			lable.push_back(y[i] - 70 - 1);
		if (y[i] >= 65 && y[i] <= 90) // upper case 1 - 26
			lable.push_back(y[i] - 64 - 1);
		if (y[i] >= 48 && y[i] <= 57) // numbers 53 - 62
			lable.push_back(y[i] + 5 - 1);

	}
#endif

	wchar_t* char_list = L"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789味全优酪乳李錦記山楂枸杞风酸牛奶新希望蔓越莓蓝蒙大果粒蚝鲜油鸿光浪花杭州超值嫩火中華豆腐西湖老现代牧场卡夫蛋黄选芦荟天喜温馨汤锅亨氏番茄沙司茵腿片静思微软亚洲研究院车库入口水表间讨论室办公电话工作台实验卫生消栓文印会议可使用警疏散图如遇情请按此处紧急打破玻璃开门擎许伟捷保持常关统一茶提示把瓶子送回房谢注意理箱密件柜日式沈为菊演厅丹棱街号停空闲位禁防止通道购物广地下剩余乐活出收费编海国机集团心东淀安勿留家自主创试范区核钢际银行民三善缘方教育科技械业满星早村麻辣诱惑眉坡酒楼剧南路铁便利交北京妇幼健儿童期发展免货巴比伦宾馆药店租驶信建投证券彩和坊惠寺书万典当苏售宝姿造型龙学校小时助服务二首都人才厦化爱者四环随分享快博管告牌最运吸烟津汇百步福食堆放品卷帘即将幕恒记甜加菲猫派克兰帝酷旗舰威迩明朗眼镜属于你我的刻装亮相滑凡蒂诺而森格冰淇淋霍顿美联亲仙踪林洗手奥特婴床佰草锦益高贝订针本传真码节约纸张就绪数据纯净商部储藏践踏青枯萎垃圾不触动未来华合差脾气禹单必有师鑫搬内石珠秋远悦莱寓座层侧燃危险航站线邮政筒埠外河深您六泉想走金凤成祥鄂尔多斯质推荐友灵感志愿盛招募类长城计划每点吃进钮受好营盖浇饭酱餐查询巾盒界农拉是说弱配梯屠卓普在腾飞哪里让英语要踩江赋专婚礼吴欧盈居玛娜前籍芙蓉价上共汽饮客屈臣诗碰庆周身年边识确鱼榨菜宠们供面给只具领取夏令脑十世纪算术斌非授权热姚聪遛狗护应避难所紫园精修看系市干鸭头奠基害嘉陵无障碍垂直设置往米压靠近维像采域严喷局铜雕社畅春雪芹画名住户绿邂逅艺再沸刷乌钙香稥蜜桃缤纷爽丝女季汁源岛啤";


	for (int i = 0; i < y.length(); i++)
	{
		for (int j = 0; j < CHAR_NUMBER; j++)
		{
			/*
			wchar_t str1 = L"0\0\0";;
			wchar_t *str2 = L"0\0\0";;
			str1[0] = *(y.c_str()+i);
			str2[0] = char_list[j];*/
			if (y[i] == char_list[j])
			{
				lable.push_back(j);
				break;
			}
		}
	}
}

void filterScores(Mat &s, int NMS_RADIUS, double marginThresh,
	std::vector<double> &maxvalvector, std::vector<int> maxidxvector, std::vector<int> &max_idx)
{


	Mat sorted_s1(s.rows, s.cols, CV_64FC1);



	cv::sort(s, sorted_s1, CV_SORT_EVERY_COLUMN);


	// compute difference between best score and second best.
	Mat margin(1, s.cols, CV_64FC1);

	margin = sorted_s1.row(s.rows - 1) - sorted_s1.row(s.rows - 2);

	// do NMS on the margin ("confidence")
	std::vector<int> I_max;
	wtnms(margin, NMS_RADIUS, I_max);

	//
	for (int i = 0; i < I_max.size(); i++)
	if (margin.at<float>(0, I_max[i])>marginThresh)
		max_idx.push_back(I_max[i]);



#if 0
	MATFile *pmatFile = NULL;
	mxArray *pMxArray = NULL;
	unsigned char *initA;
	long M, N;


	M = sorted_s1.rows;
	N = sorted_s1.cols;


	// 生成.mat文件  
	double *outA = (double*)mxMalloc(M*N*sizeof(double));
	for (int ii = 0; ii<M; ii++)
	for (int jj = 0; jj<N; jj++)
		outA[M*jj + ii] = sorted_s1.at<double>(ii, jj);
	pmatFile = matOpen("bigMat.mat", "w");
	pMxArray = mxCreateDoubleMatrix(M, N, mxREAL);
	mxSetData(pMxArray, (void *)outA);
	matPutVariable(pmatFile, "bigMat", pMxArray);

	mxFree(outA);
	matClose(pmatFile);
#endif

}


void wtnms(Mat &s, int rad, std::vector<int> &output_I)
{
	// pading
	Mat pad_s = Mat::zeros(1, s.cols + 2 * rad, CV_64FC1);

	s.copyTo(pad_s(Rect(2, 0, s.cols, 1)));

	// im2col, 2*rad+1的window依次滑动pad_s组成列，相当于每个s的元素是每列的中心元素
	Mat im2col_mat(2 * rad + 1, s.cols, CV_64FC1);
	for (int i = 0; i < s.cols; i++)
	{
		Mat tmp = pad_s(Rect(i, 0, 2 * rad + 1, 1));
		tmp = tmp.t();
		tmp.copyTo(im2col_mat.col(i));
	}

	// s与img2col_mat比较，如果大于或者等于就记录idx到output_I当中
	double minval, maxval;
	double eps = 2.2204e-16;

	for (int i = 0; i < im2col_mat.cols; i++)
	{
		cv::minMaxIdx(im2col_mat.col(i), &minval, &maxval);

		if (s.at<float>(0, i) >= maxval - 2 * eps)
			output_I.push_back(i);

	}

}



// given sliding window classifier scores, predict the word label using
// a Viterbi - style alignment algorithm.
// 增加w,h，图片的宽高，用来getMatchScore里计算每个字符在文字块的2维坐标和到边界距离
wstring Score2Word_of_Boxes(Mat &score, Mat &boxpos, int boxnum, int w, int h, std::vector<wstring> &Lex)
{
	double matchscore_thresh = -4.0;
	double minval, maxval;
	int minidx[2], maxidx[2];

	std::vector<double> maxvalvector;
	std::vector<int> maxidxvector;


	// 求取每一个位置的所有CHAR_NUMBER字符的概率最大值和位置
	for (int i = 0; i < score.cols; i++)
	{
		cv::minMaxIdx(score.col(i), &minval, &maxval, minidx, maxidx);
		maxvalvector.push_back(maxval);
		maxidxvector.push_back(maxidx[0]);
	}

	// 

	//	char* chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
	wchar_t* chars = L"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789味全优酪乳李錦記山楂枸杞风酸牛奶新希望蔓越莓蓝蒙大果粒蚝鲜油鸿光浪花杭州超值嫩火中華豆腐西湖老现代牧场卡夫蛋黄选芦荟天喜温馨汤锅亨氏番茄沙司茵腿片静思微软亚洲研究院车库入口水表间讨论室办公电话工作台实验卫生消栓文印会议可使用警疏散图如遇情请按此处紧急打破玻璃开门擎许伟捷保持常关统一茶提示把瓶子送回房谢注意理箱密件柜日式沈为菊演厅丹棱街号停空闲位禁防止通道购物广地下剩余乐活出收费编海国机集团心东淀安勿留家自主创试范区核钢际银行民三善缘方教育科技械业满星早村麻辣诱惑眉坡酒楼剧南路铁便利交北京妇幼健儿童期发展免货巴比伦宾馆药店租驶信建投证券彩和坊惠寺书万典当苏售宝姿造型龙学校小时助服务二首都人才厦化爱者四环随分享快博管告牌最运吸烟津汇百步福食堆放品卷帘即将幕恒记甜加菲猫派克兰帝酷旗舰威迩明朗眼镜属于你我的刻装亮相滑凡蒂诺而森格冰淇淋霍顿美联亲仙踪林洗手奥特婴床佰草锦益高贝订针本传真码节约纸张就绪数据纯净商部储藏践踏青枯萎垃圾不触动未来华合差脾气禹单必有师鑫搬内石珠秋远悦莱寓座层侧燃危险航站线邮政筒埠外河深您六泉想走金凤成祥鄂尔多斯质推荐友灵感志愿盛招募类长城计划每点吃进钮受好营盖浇饭酱餐查询巾盒界农拉是说弱配梯屠卓普在腾飞哪里让英语要踩江赋专婚礼吴欧盈居玛娜前籍芙蓉价上共汽饮客屈臣诗碰庆周身年边识确鱼榨菜宠们供面给只具领取夏令脑十世纪算术斌非授权热姚聪遛狗护应避难所紫园精修看系市干鸭头奠基害嘉陵无障碍垂直设置往米压靠近维像采域严喷局铜雕社畅春雪芹画名住户绿邂逅艺再沸刷乌钙香稥蜜桃缤纷爽丝女季汁源岛啤";

	// 处理scores，使得和大小写无关
	Mat smallcaseMat = score(Rect(0, 0, boxnum, 26));
	Mat bigcaseMat = score(Rect(0, 26, boxnum, 26));

	Mat diffcaseMat = smallcaseMat - bigcaseMat;
	Mat tmp(26, boxnum, CV_64FC1);

	for (int i = 0; i < smallcaseMat.rows; i++)
	for (int j = 0; j < smallcaseMat.cols; j++)
	{
		if (diffcaseMat.at<float>(i, j) > 0)
			// 小写字符概率比较高，取小写字符的概率
			tmp.at<double>(i, j) = smallcaseMat.at<float>(i, j);
		else
			// 大写字符概率比较高，取大写字符的概率
			tmp.at<double>(i, j) = bigcaseMat.at<float>(i, j);
	}

	Mat case_insense_scores(score.rows, boxnum, CV_64FC1);
	Mat rect_score;

	rect_score = score(Rect(0, 0, boxnum, score.rows));
	rect_score.copyTo(case_insense_scores);
	tmp.copyTo(case_insense_scores(Rect(0, 0, boxnum, 26)));
	tmp.copyTo(case_insense_scores(Rect(0, 26, boxnum, 26)));

	vector<int> max_idx(boxnum);
	for (int i = 0; i < boxnum; i++)
		max_idx[i] = i;

	wstring predword;

#if 0
	wchar_t word2out[100000];
	for (int i = 0; i < max_idx.size(); i++)
	{


		word2out[i] = chars[maxidxvector[max_idx[i]]];

		word2out[i + 1] = '\0';
	}

	predword = word2out;

#else

	//
	Mat matchScoreArray = Mat::ones(1, Lex.size(), CV_64FC1);
	matchScoreArray = -99 * matchScoreArray;

	double max_matchscore = -100.0;
	predword = L"";

	printf("\n");
	for (int i = 0; i < Lex.size(); i++)
	{
		printf("\rlex process word %d/%d", i + 1, Lex.size());

		double matchscore;
		Mat real_good_idx;

		getMatchScore_of_Boxes(case_insense_scores, boxpos, w, h, Lex[i], max_idx, matchscore, real_good_idx);

		if (matchscore>max_matchscore && matchscore>matchscore_thresh)
		{
			max_matchscore = matchscore;
			predword = Lex[i];
		}
	}
#endif

	return predword;


}




// given the char recognition score and a candidate lexicon word,
// computes the matchscore using dynamic programming
// origscores: the score matrix M
// origword : one single lexicon word
// good_idx : peak positions obtained after NMS on 'confidence margin'
// 增加w,h，图片的宽高，用来getMatchScore里计算每个字符在文字块的2维坐标和到边界距离
void getMatchScore_of_Boxes(Mat &origscores, Mat &boxpos, int w, int h, wstring origword, vector<int> &good_idx,
	double &matchscore, Mat &real_good_idx)
{
	// 只留下fliterscores得到的备选列
	Mat scores(origscores.rows, good_idx.size(), CV_64FC1);

	for (int i = 0; i<good_idx.size(); i++)
	{
		Mat tmp = origscores.col(good_idx[i]);
		tmp.copyTo(scores.col(i));
	}

	// 
	vector<double> word_lable;
	ascii2label(origword, word_lable);




	//
	// scoreMat(i, j) contains the maximum score you can get so far by matching
	// the ith character with the jth score location

	// if word is longer than number of sliding windows, something's wrong

	if (origword.length()<1 || origword.length()>good_idx.size())
	{
		matchscore = -100;
		return;
	}


	// dynamic programming window
	Mat scoreMat = Mat::zeros(origword.length(), good_idx.size(), CV_64FC1);
	Mat scoreIdx = Mat::zeros(origword.length(), good_idx.size(), CV_64FC1);

	// initialize first row
	scores.row(word_lable[0]).copyTo(scoreMat.row(0));

	// Viterbi dynamic programming
	for (int i = 1; i < origword.length(); i++)
	for (int j = i; j < good_idx.size(); j++)
	{
		Mat tmp = scoreMat(Rect(i - 1, i - 1, j - i + 1, 1));
		double minval, maxPrev;
		int minidx[2], maxPrevIdx[2];

		cv::minMaxIdx(tmp.t(), &minval, &maxPrev, minidx, maxPrevIdx);

		scoreMat.at<double>(i, j) = scores.at<double>((int)(word_lable[i]), j) + maxPrev;

		scoreIdx.at<double>(i, j) = maxPrevIdx[0];
	}

	double minval;
	int minidx[2], lastidx[2];

	Mat tmp = scoreMat(Rect(origword.length() - 1, origword.length() - 1, good_idx.size() - origword.length() + 1, 1));


	cv::minMaxIdx(tmp.t(), &minval, &matchscore, minidx, lastidx);

	//

	real_good_idx = Mat::zeros(1, origword.length(), CV_64FC1);

	real_good_idx.at<double>(0, origword.length() - 1) = lastidx[0] + origword.length() - 1;

	int i = origword.length() - 1;
	// backtrace to find correspondence between peaks and chars.
	while (i > 0)
	{
		real_good_idx.at<double>(0, i - 1) = (int)(scoreIdx.at<double>(i, (int)(real_good_idx.at<double>(0, i)))) + i - 1;
		i = i - 1;
	}


	for (int i = 0; i < real_good_idx.cols; i++)
		real_good_idx.at<double>(0, i) = good_idx[(int)(real_good_idx.at<double>(0, i))];

	// 修改为根据2维坐标的实际gap
	Mat gaps(1, real_good_idx.cols + 1, CV_64FC1);

	Mat tmp1(1, real_good_idx.cols + 1, CV_64FC1);
	Mat tmp2(1, real_good_idx.cols + 1, CV_64FC1);
	real_good_idx.copyTo(tmp1(Rect(0, 0, real_good_idx.cols, 1)));
	real_good_idx.copyTo(tmp2(Rect(1, 0, real_good_idx.cols, 1)));


	// 计算block distance作为gap
	Mat tmp1_2d(1, real_good_idx.cols + 1, CV_64FC2);
	Mat tmp2_2d(1, real_good_idx.cols + 1, CV_64FC2);

	for (int i = 0; i < tmp1.cols - 1; i++)
		tmp1_2d.at<Vec2d>(0, i) = boxpos.at<Vec2d>(0, (int)(tmp1.at<double>(0, i)));

	for (int i = 1; i < tmp2.cols; i++)
		tmp2_2d.at<Vec2d>(0, i) = boxpos.at<Vec2d>(0, (int)(tmp2.at<double>(0, i)));


	// 置为图片最右下角的点
	tmp1_2d.at<Vec2d>(0, tmp1.cols - 1)[0] = h - 1;
	tmp1_2d.at<Vec2d>(0, tmp1.cols - 1)[1] = w - 1;

	tmp2_2d.at<Vec2d>(0, 0)[0] = 0;
	tmp2_2d.at<Vec2d>(0, 0)[1] = 0;

	// gaps = tmp1 - tmp2;

	for (int i = 0; i < tmp1.cols; i++)
	{
		gaps.at<double>(i) = abs(tmp1_2d.at<Vec2d>(0, i)[0] - tmp2_2d.at<Vec2d>(0, i)[0]) +
			abs(tmp1_2d.at<Vec2d>(0, i)[1] - tmp2_2d.at<Vec2d>(0, i)[1]);
	}

	// 计算字符间的距离的方差，如果大就会惩罚大
	// penalize geometric inconsistency
	double c_std = 0.08;
	double  c_narrow = 0.6;
	double std_loss;
	// inconsistent character spacing
	if (gaps.cols >= 4)
	{
		Mat tmp = gaps(Rect(1, 0, gaps.cols - 2, 1));

		Scalar row_mean, row_std;
		meanStdDev(tmp, row_mean, row_std);

#if 0
		for (int i = 0; i < tmp.cols; i++)
			printf("%lf ", tmp.at<double>(0, i));

		printf("\n%lf ", row_std[0]);
#endif

		std_loss = c_std*row_std[0];
	}
	else if (gaps.cols >= 3)
	{
		// 处理两个字的词，认为理想的gap距离（block距离）为1个字符宽度,即32
		Mat tmp(1, 2, CV_64FC1);
		tmp.at<double>(0, 0) = gaps.at<double>(0, 1);
		tmp.at<double>(0, 1) = 32;

		Scalar row_mean, row_std;
		meanStdDev(tmp, row_mean, row_std);

		std_loss = c_std*row_std[0];
	}
	else
		std_loss = 0;


	//very narrow characters
	double narrow_loss = 0;

	// 不等于'I'和'l'
	if (wcscmp(origword.c_str(), L"I") && wcscmp(origword.c_str(), L"l"))
	{
		if (origscores.cols*1.0 / origword.length() < 8)
			narrow_loss = (8 - origscores.cols*1.0 / origword.length())*c_narrow;
	}



	//penalize excessive extra space on both sides
	/*
	double rgi_minval, rgi_maxval;

	cv::minMaxIdx(real_good_idx.t(), &rgi_minval, &rgi_maxval);
	*/

	// 计算左右上下距离边界的距离，如果比较远就惩罚比较多
	double rgi_minval, rgi_maxval;  // 左右边界距离
	double tgi_minval, tgi_maxval;  // 上下边界距离
	Mat tmp1_2d_spit[2];

	for (int i = 0; i < real_good_idx.cols; i++)
	{
		// 行坐标
		tmp1_2d.at<Vec2d>(0, i)[0] = (int)(real_good_idx.at<double>(0, i)) / w;

		// 列坐标
		tmp1_2d.at<Vec2d>(0, i)[1] = (int)(real_good_idx.at<double>(0, i)) % w;
	}

	split(tmp1_2d, tmp1_2d_spit);

	cv::minMaxIdx(tmp1_2d_spit[1].t(), &rgi_minval, &rgi_maxval);
	cv::minMaxIdx(tmp1_2d_spit[0].t(), &tgi_minval, &tgi_maxval);


	//	matchscore = matchscore - std_loss - narrow_loss - ((rgi_minval - 1) / origscores.cols + (origscores.cols - rgi_maxval) / origscores.cols);

	// 由于是纯中文，所以不用考虑narrow_loss,另外惩罚项也按照块的方式来计算
	//matchscore = matchscore - std_loss - ((rgi_minval - 1) / w + (w - rgi_maxval) / w) -
	//	((tgi_minval - 1) / h + (h - tgi_maxval) / h);

	// 由于是各个零散的box串起来进行匹配，不是在一个定位好的text box中进行
	// 所以不计算边界距离的惩罚值在内
	matchscore = matchscore - std_loss;

}