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

#define CHAR_NUMBER 719 // ʶ���Ŀ���ַ��ĸ���

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

	wchar_t* char_list = L"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789ζȫ���������\ӛɽ���轷���ţ����ϣ����Խݮ���ɴ��������ͺ���˻����ݳ�ֵ�ۻ����A�����������ִ��������򵰻�ѡ«����ϲ��ܰ�������Ϸ���ɳ˾����Ƭ��˼΢�������о�Ժ�������ˮ��������Ұ칫�绰����̨ʵ��������˨��ӡ�����ʹ�þ���ɢͼ�������밴�˴��������Ʋ�����������ΰ�ݱ��ֳ���ͳһ����ʾ��ƿ���ͻط�лע�������ܼ�����ʽ��Ϊ����������ֺ�ͣ����λ����ֹͨ����������ʣ���ֻ���շѱຣ���������Ķ����������������Է����˸ּ�����������Ե�������Ƽ�еҵ������������ջ�ü�¾�¥����·���������������׽���ͯ�ڷ�չ����ͱ��ױ���ҩ����ʻ�Ž�Ͷ֤ȯ�ʺͷ���������䵱���۱���������ѧУСʱ��������׶��˲��û������Ļ������첩�ܸ����������̽��ٲ���ʳ�ѷ�Ʒ��������Ļ�����ӷ�è�ɿ����ۿ��콢���������۾��������ҵĿ�װ���໬����ŵ��ɭ�����ܻ���������������ϴ�ְ���Ӥ���۲ݽ���߱����뱾�������Լֽ�ž������ݴ����̲����ؼ�̤���ή����������δ�����ϲ�Ƣ��������ʦ�ΰ���ʯ����Զ����Ԣ�����ȼΣ�պ�վ������Ͳ�����������Ȫ���߽����������˹���Ƽ������־Ըʢ��ļ�೤�Ǽƻ�ÿ��Խ�ť�ܺ�Ӫ�ǽ������Ͳ�ѯ��н�ũ����˵��������׿�����ڷ�������Ӣ��Ҫ�Ƚ���ר������ŷӯ������ǰ��ܽ�ؼ��Ϲ�����������ʫ�����������ʶȷ��ե�˳��ǹ����ֻ����ȡ������ʮ�������������Ȩ��Ҧ���޹���Ӧ��������԰���޿�ϵ�и�Ѽͷ������������ϰ���ֱ��������ѹ����ά����������ͭ���糩��ѩ�ۻ���ס�����������ٷ�ˢ�ڸ���E�����ͷ�ˬ˿Ů��֭Դ��ơ";


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


	// ����.mat�ļ�  
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

	// im2col, 2*rad+1��window���λ���pad_s����У��൱��ÿ��s��Ԫ����ÿ�е�����Ԫ��
	Mat im2col_mat(2 * rad + 1, s.cols, CV_64FC1);
	for (int i = 0; i < s.cols; i++)
	{
		Mat tmp = pad_s(Rect(i, 0, 2 * rad + 1, 1));
		tmp = tmp.t();
		tmp.copyTo(im2col_mat.col(i));
	}

	// s��img2col_mat�Ƚϣ�������ڻ��ߵ��ھͼ�¼idx��output_I����
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
// ����w,h��ͼƬ�Ŀ�ߣ�����getMatchScore�����ÿ���ַ������ֿ��2ά����͵��߽����
wstring Score2Word_of_Boxes(Mat &score, Mat &boxpos, int boxnum, int w, int h, std::vector<wstring> &Lex)
{
	double matchscore_thresh = -4.0;
	double minval, maxval;
	int minidx[2], maxidx[2];

	std::vector<double> maxvalvector;
	std::vector<int> maxidxvector;


	// ��ȡÿһ��λ�õ�����CHAR_NUMBER�ַ��ĸ������ֵ��λ��
	for (int i = 0; i < score.cols; i++)
	{
		cv::minMaxIdx(score.col(i), &minval, &maxval, minidx, maxidx);
		maxvalvector.push_back(maxval);
		maxidxvector.push_back(maxidx[0]);
	}

	// 

	//	char* chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
	wchar_t* chars = L"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789ζȫ���������\ӛɽ���轷���ţ����ϣ����Խݮ���ɴ��������ͺ���˻����ݳ�ֵ�ۻ����A�����������ִ��������򵰻�ѡ«����ϲ��ܰ�������Ϸ���ɳ˾����Ƭ��˼΢�������о�Ժ�������ˮ��������Ұ칫�绰����̨ʵ��������˨��ӡ�����ʹ�þ���ɢͼ�������밴�˴��������Ʋ�����������ΰ�ݱ��ֳ���ͳһ����ʾ��ƿ���ͻط�лע�������ܼ�����ʽ��Ϊ����������ֺ�ͣ����λ����ֹͨ����������ʣ���ֻ���շѱຣ���������Ķ����������������Է����˸ּ�����������Ե�������Ƽ�еҵ������������ջ�ü�¾�¥����·���������������׽���ͯ�ڷ�չ����ͱ��ױ���ҩ����ʻ�Ž�Ͷ֤ȯ�ʺͷ���������䵱���۱���������ѧУСʱ��������׶��˲��û������Ļ������첩�ܸ����������̽��ٲ���ʳ�ѷ�Ʒ��������Ļ�����ӷ�è�ɿ����ۿ��콢���������۾��������ҵĿ�װ���໬����ŵ��ɭ�����ܻ���������������ϴ�ְ���Ӥ���۲ݽ���߱����뱾�������Լֽ�ž������ݴ����̲����ؼ�̤���ή����������δ�����ϲ�Ƣ��������ʦ�ΰ���ʯ����Զ����Ԣ�����ȼΣ�պ�վ������Ͳ�����������Ȫ���߽����������˹���Ƽ������־Ըʢ��ļ�೤�Ǽƻ�ÿ��Խ�ť�ܺ�Ӫ�ǽ������Ͳ�ѯ��н�ũ����˵��������׿�����ڷ�������Ӣ��Ҫ�Ƚ���ר������ŷӯ������ǰ��ܽ�ؼ��Ϲ�����������ʫ�����������ʶȷ��ե�˳��ǹ����ֻ����ȡ������ʮ�������������Ȩ��Ҧ���޹���Ӧ��������԰���޿�ϵ�и�Ѽͷ������������ϰ���ֱ��������ѹ����ά����������ͭ���糩��ѩ�ۻ���ס�����������ٷ�ˢ�ڸ���E�����ͷ�ˬ˿Ů��֭Դ��ơ";

	// ����scores��ʹ�úʹ�Сд�޹�
	Mat smallcaseMat = score(Rect(0, 0, boxnum, 26));
	Mat bigcaseMat = score(Rect(0, 26, boxnum, 26));

	Mat diffcaseMat = smallcaseMat - bigcaseMat;
	Mat tmp(26, boxnum, CV_64FC1);

	for (int i = 0; i < smallcaseMat.rows; i++)
	for (int j = 0; j < smallcaseMat.cols; j++)
	{
		if (diffcaseMat.at<float>(i, j) > 0)
			// Сд�ַ����ʱȽϸߣ�ȡСд�ַ��ĸ���
			tmp.at<double>(i, j) = smallcaseMat.at<float>(i, j);
		else
			// ��д�ַ����ʱȽϸߣ�ȡ��д�ַ��ĸ���
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
// ����w,h��ͼƬ�Ŀ�ߣ�����getMatchScore�����ÿ���ַ������ֿ��2ά����͵��߽����
void getMatchScore_of_Boxes(Mat &origscores, Mat &boxpos, int w, int h, wstring origword, vector<int> &good_idx,
	double &matchscore, Mat &real_good_idx)
{
	// ֻ����fliterscores�õ��ı�ѡ��
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

	// �޸�Ϊ����2ά�����ʵ��gap
	Mat gaps(1, real_good_idx.cols + 1, CV_64FC1);

	Mat tmp1(1, real_good_idx.cols + 1, CV_64FC1);
	Mat tmp2(1, real_good_idx.cols + 1, CV_64FC1);
	real_good_idx.copyTo(tmp1(Rect(0, 0, real_good_idx.cols, 1)));
	real_good_idx.copyTo(tmp2(Rect(1, 0, real_good_idx.cols, 1)));


	// ����block distance��Ϊgap
	Mat tmp1_2d(1, real_good_idx.cols + 1, CV_64FC2);
	Mat tmp2_2d(1, real_good_idx.cols + 1, CV_64FC2);

	for (int i = 0; i < tmp1.cols - 1; i++)
		tmp1_2d.at<Vec2d>(0, i) = boxpos.at<Vec2d>(0, (int)(tmp1.at<double>(0, i)));

	for (int i = 1; i < tmp2.cols; i++)
		tmp2_2d.at<Vec2d>(0, i) = boxpos.at<Vec2d>(0, (int)(tmp2.at<double>(0, i)));


	// ��ΪͼƬ�����½ǵĵ�
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

	// �����ַ���ľ���ķ�������ͻ�ͷ���
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
		// ���������ֵĴʣ���Ϊ�����gap���루block���룩Ϊ1���ַ����,��32
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

	// ������'I'��'l'
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

	// �����������¾���߽�ľ��룬����Ƚ�Զ�ͳͷ��Ƚ϶�
	double rgi_minval, rgi_maxval;  // ���ұ߽����
	double tgi_minval, tgi_maxval;  // ���±߽����
	Mat tmp1_2d_spit[2];

	for (int i = 0; i < real_good_idx.cols; i++)
	{
		// ������
		tmp1_2d.at<Vec2d>(0, i)[0] = (int)(real_good_idx.at<double>(0, i)) / w;

		// ������
		tmp1_2d.at<Vec2d>(0, i)[1] = (int)(real_good_idx.at<double>(0, i)) % w;
	}

	split(tmp1_2d, tmp1_2d_spit);

	cv::minMaxIdx(tmp1_2d_spit[1].t(), &rgi_minval, &rgi_maxval);
	cv::minMaxIdx(tmp1_2d_spit[0].t(), &tgi_minval, &tgi_maxval);


	//	matchscore = matchscore - std_loss - narrow_loss - ((rgi_minval - 1) / origscores.cols + (origscores.cols - rgi_maxval) / origscores.cols);

	// �����Ǵ����ģ����Բ��ÿ���narrow_loss,����ͷ���Ҳ���տ�ķ�ʽ������
	//matchscore = matchscore - std_loss - ((rgi_minval - 1) / w + (w - rgi_maxval) / w) -
	//	((tgi_minval - 1) / h + (h - tgi_maxval) / h);

	// �����Ǹ�����ɢ��box����������ƥ�䣬������һ����λ�õ�text box�н���
	// ���Բ�����߽����ĳͷ�ֵ����
	matchscore = matchscore - std_loss;

}