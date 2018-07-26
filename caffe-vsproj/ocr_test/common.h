#pragma once

#define  _CRT_SECURE_NO_WARNINGS
#define OPENCV
#include <caffe/caffe.hpp>




#include "parser.h"


#include"opencv/cv.h"
#include"opencv/highgui.h"
#include"opencv/cxcore.h"



#include"math.h"

#include <stdio.h>
#include <fstream>
#include <iostream>
#include<string>
#include <vector>
#include <windows.h>
#include <process.h>
#include <direct.h> 
#include <cmath>
#include <ctime>
#include <chrono>
#include <random>
#include <memory>


using namespace caffe;  // NOLINT(build/namespaces)
using std::string;



using namespace cv;
using namespace std;




#define showSteps  0
#define showdemo 1
#define USE_OPENCV




int notfound = 2;
int readtype = 0;//1为车牌2为字符

CascadeClassifier car_cascade;
CascadeClassifier char_cascade;
vector<string> file_name, img_path;
vector<string> imgfileNames;



void FastFilter(IplImage *img, double sigma)
{
	int filter_size;

	// Reject unreasonable demands
	if (sigma > 200) sigma = 200;

	// get needed filter size (enforce oddness)
	filter_size = (int)floor(sigma * 6) / 2;
	filter_size = filter_size * 2 + 1;

	// If 3 sigma is less than a pixel, why bother (ie sigma < 2/3)
	if (filter_size < 3) return;

	// Filter, or downsample and recurse
	/*if (filter_size < 10) {

	#ifdef USE_EXACT_SIGMA
	FilterGaussian(img, sigma)
	#else
	cvSmooth( img, img, CV_GAUSSIAN, filter_size, filter_size );
	#endif

	}
	else*/ {
		if (img->width < 2 || img->height < 2) return;

		IplImage* sub_img = cvCreateImage(cvSize(img->width / 2, img->height / 2), img->depth, img->nChannels);

		cvPyrDown(img, sub_img);

		FastFilter(sub_img, sigma / 2.0);

		cvResize(sub_img, img, CV_INTER_LINEAR);

		cvReleaseImage(&sub_img);
	}

}

void MultiScaleRetinex(IplImage *img, int scales, double *weights, double *sigmas, int gain, int offset)
{
	int i;
	double weight;
	IplImage *A, *fA, *fB, *fC;

	// Initialize temp images
	fA = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, img->nChannels);
	fB = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, img->nChannels);
	fC = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, img->nChannels);


	// Compute log image
	cvConvert(img, fA);
	cvLog(fA, fB);

	// Normalize according to given weights
	for (i = 0, weight = 0; i < scales; i++)
		weight += weights[i];

	if (weight != 1.0) cvScale(fB, fB, weight);

	// Filter at each scale
	for (i = 0; i < scales; i++) {
		A = cvCloneImage(img);
		FastFilter(A, sigmas[i]);

		cvConvert(A, fA);
		cvLog(fA, fC);
		cvReleaseImage(&A);

		// Compute weighted difference
		cvScale(fC, fC, weights[i]);
		cvSub(fB, fC, fB);
	}

	// Restore
	cvConvertScale(fB, img, gain, offset);

	// Release temp images
	cvReleaseImage(&fA);
	cvReleaseImage(&fB);
	cvReleaseImage(&fC);
}




int init_detect(CascadeClassifier &car_cascade, CascadeClassifier &char_cascade) {

	
	readtype = 2;
	if (!char_cascade.load("cascade12.xml"))
	{
		cerr << "ERROR: Could not load classifier cascade" << endl;
		return -1;
	}
	
	readtype = 1;
	if (!car_cascade.load("cascade_11_plane_card_20160912.xml"))
	{
		cerr << "ERROR: Could not load classifier cascade" << endl;
		return -1;
	}
}

void InsertSort(int a[], int count)
{
	int i, j, temp;
	for (i = 1; i<count; i++)
	{
		temp = a[i];
		j = i - 1;
		while (a[j]>temp && j >= 0)
		{
			a[j + 1] = a[j];
			j--;
		}
		if (j != (i - 1))
			a[j + 1] = temp;
	}
}

int SizeOfRect(const CvRect& rect)   //面积
{
	return rect.height*rect.width;
}

CvRect IntersectRect(CvRect result_i, CvRect resmax_j) {    //交集
	CvRect rectInter;


	rectInter.x = max(result_i.x, resmax_j.x);
	rectInter.y = max(result_i.y, resmax_j.y);

	int xxx = min((result_i.x + result_i.width), (resmax_j.x + resmax_j.width));
	int yyy = min((result_i.y + result_i.height), (resmax_j.y + resmax_j.height));

	rectInter.width = xxx - rectInter.x;
	rectInter.height = yyy - rectInter.y;

	return rectInter;
}

CvRect UnionRect(CvRect resmax_j, CvRect result_i) {   //并集
	CvRect resmax_jj;

	resmax_jj.x = min(result_i.x, resmax_j.x);
	resmax_jj.y = min(result_i.y, resmax_j.y);

	int xxx = max((result_i.x + result_i.width), (resmax_j.x + resmax_j.width));
	int yyy = max((result_i.y + result_i.height), (resmax_j.y + resmax_j.height));

	resmax_jj.width = xxx - resmax_j.x;
	resmax_jj.height = yyy - resmax_j.y;

	return resmax_jj;

}

vector<CvRect> roichoose(vector<CvRect>detectROI, Mat choose_detect_obj) {
	/***************框框宽度未进行筛选*******************/
	int image_width = choose_detect_obj.cols;
	int image_height = choose_detect_obj.rows;
	int judgeheight = image_height*0.3;
	vector<vector<CvRect>> b;//所有疑似等间距框的存储位置
	vector<int> d;//所有疑似间距存储位置
				  //	vector<int> shanchu;//记录需要删除的点
	for (int i = detectROI.size() - 1; i > -1; i--)
	{
		if (detectROI[i].height < judgeheight)
			//			shanchu.push_back(i);
			detectROI.erase(detectROI.begin() + i);
	}

	/***************测试****************/
	if (showSteps) {
		for (int i = 0; i < detectROI.size(); i++) {
			printf("x: %d \n", detectROI[i].x);
		}
	}
	for (int i = 0; i < detectROI.size(); i++)//循环中i指向起始点，j代表搜索宽度
	{
		if (detectROI[i].height < judgeheight)
			continue;
		for (int j = 10; j < (image_width / 7); j++)//j为在一定范围内的宽度搜索
		{
			/******测试*******/



			/******测试*******/

			int k = 1;//k代表点的数量
			int startpoint = detectROI[i].x;
			int startpoint_y = detectROI[i].y;
			int min_distance = 2 * image_width;       //记录点与最佳位置的间距
			int bestpoint = detectROI.size() + 1; //记录最佳点
			vector<CvRect> grouppoint;//每一组最佳点的存储位置
			grouppoint.push_back(detectROI[i]);

			for (int ii = i + 1; ii < detectROI.size(); ii++)//寻找搜索宽度内的最优点
			{
				if (detectROI[ii].height < judgeheight)
					continue;
				int distance_x = abs(j + startpoint - detectROI[ii].x);
				if (distance_x < 0.5*j)                         //选取x值合适的点，比较它们与最佳位置点的距离选最小的
				{
					int distance = (distance_x)*(distance_x)+0.7*(detectROI[ii].y - startpoint_y)*(detectROI[ii].y - startpoint_y);//我们更看重x的坐标，故y的差值影响的权重降低
					distance = sqrt(distance);
					if (distance < min_distance)
					{
						min_distance = distance;
						bestpoint = ii;

					}
				}
			}
			/******测试*******/
			//		printf("min_distance=%d", min_distance);

			/*******************************************4-10修改 在后续的框中增加y值的预测位置bestpoint_y*************************************/
			while (bestpoint < (detectROI.size() + 1) && min_distance < 2 * j)
			{
				k++;
				grouppoint.push_back(detectROI[bestpoint]);
				startpoint = detectROI[bestpoint].x;//更新起始位置
				int startpoint_y2 = detectROI[bestpoint].y;//更新y2
				min_distance = image_width;
				bestpoint = detectROI.size() + 1;

				for (int ii = i + 1; ii < detectROI.size(); ii++)//寻找搜索宽度内的最优点
				{
					int distance_x = abs(j + startpoint - detectROI[ii].x);
					int best_y = 2 * startpoint_y2 - startpoint_y;//理想中的y坐标
					int distance_y = abs(detectROI[ii].y - best_y);

					if (distance_x < 0.5*j)                         //选取x值合适的点，比较它们与最佳位置点的距离选最小的
					{
						int distance = (distance_x)*(distance_x)+0.7*distance_y*distance_y;//我们更看重x的坐标，故y的差值影响的权重降低
						distance = sqrt(distance);
						if (distance < min_distance)
						{
							min_distance = distance;
							bestpoint = ii;


						}
					}
				}
				startpoint_y = startpoint_y2;//更新y1

			}




			if (grouppoint.size() > 3)
			{
				/*if (showSteps)
				{
				for (int testshow = 0; testshow < grouppoint.size(); testshow++){
				printf("x=%d y=%d width=%d height=%d \n", grouppoint[testshow].x, grouppoint[testshow].y, grouppoint[testshow].width, grouppoint[testshow].height);
				}
				getchar();
				}*/

				b.push_back(grouppoint);
				d.push_back(j);
			}

		}
	}


	//到这里位置我们找完了所有类似等间距框，接下来加以比较选出最优解

	double sum_min = 100000;
	int bestgroup = 0;
	if (b.size() > 2)
	{
		for (int i = 0; i < b.size(); i++)              //计算每一组框的sum值,选出bestgroup
		{
			double sum = 0;

			for (int j = 1; j < b[i].size(); j++)
			{
				int dis = d[i];
				sum = sum + (b[i][j].x - b[i][j - 1].x - dis)*(b[i][j].x - b[i][j - 1].x - dis) + 4 * (b[i][j].y - b[i][j - 1].y)*(b[i][j].y - b[i][j - 1].y) + 0.5*(b[i][j].height - b[i][j - 1].height)*(b[i][j].height - b[i][j - 1].height) + 0.5*(b[i][j].width - b[i][j - 1].width)*(b[i][j].width - b[i][j - 1].width);
			}
			sum = sum / b[i].size();
			sum = sum / (1 + pow(b[i].size(), 2));            //给数量多的列奖励机制

															  //				printf("b[i].size=%d sum=%f \n", b[i].size(),sum);
			if (sum < sum_min)
			{
				sum_min = sum;
				bestgroup = i;
			}
		}
		//		printf("size=%d distacne=%d sum_min=%f \n", b[bestgroup].size(), d[bestgroup], sum_min);
		//		return b[bestgroup];
		/*****************在bestgroup的基础上找到前后的框********/
		int bestnumber = b[bestgroup].size();
		int bestx0 = b[bestgroup][0].x;//定义最佳组起点终点坐标
		int besty0 = b[bestgroup][0].y;
		int bestx1 = b[bestgroup][bestnumber - 1].x;
		int besty1 = b[bestgroup][bestnumber - 1].y;
		int bestheight = 0;
		int bestw = 0;
		for (int i = 0; i < bestnumber; i++) {
			bestheight = bestheight + b[bestgroup][0].height;
			bestw = bestw + b[bestgroup][0].width;
		}

		bestheight = bestheight / bestnumber;
		bestw = bestw / bestnumber;
		int bestwidth = d[bestgroup];
		/********************************增加前框*****************************/
		int search_startpoint = bestx0;
		int search_starty = besty0;
		int searchpoint = detectROI.size() + 1;
		int search_min_distance = image_width;
		int presize = b[bestgroup].size();

		for (int i = 0; i < detectROI.size(); i++)
		{
			int distance_x = search_startpoint - bestwidth - detectROI[i].x;
			int distance_y = abs(search_starty - detectROI[i].y);
			/***************测试****************/

			/*	if (distance_x>-0.8*bestwidth && distance_x < 1.8*bestwidth){
			printf("bestwidth=%d bestheight=%d bestw=%d \n", bestwidth, bestheight, bestw);
			printf("distance_x=%d x=%d \n", distance_x, detectROI[i].x);
			printf("width=%d height=%d \n", detectROI[i].width, detectROI[i].height);
			}*/


			/***************测试****************/
			if (distance_x>-0.6*bestwidth && distance_x < 1.8*bestwidth && detectROI[i].height > 0.7*bestheight &&  detectROI[i].height<1.4*bestheight &&   detectROI[i].width  > 0.7*bestw &&  detectROI[i].width < 1.4*bestw && distance_y<0.4*bestheight)  //选取x值合适的点，比较它们与最佳位置点的距离选最小的，加入框高度宽度的筛选
			{
				int distance = (distance_x)*(distance_x)+0.7*(besty0 - detectROI[i].y)*(besty0 - detectROI[i].y);//我们更看重x的坐标，故y的差值影响的权重降低
				distance = sqrt(distance);
				if (distance < search_min_distance)
				{
					search_min_distance = distance;
					searchpoint = i;

				}
			}
		}


		while (searchpoint<(detectROI.size() + 1) && search_min_distance<2 * bestwidth)
		{

			b[bestgroup].insert(b[bestgroup].begin(), detectROI[searchpoint]);
			search_startpoint = detectROI[searchpoint].x;//更新起始位置
			search_starty = detectROI[searchpoint].y;
			search_min_distance = image_width;
			searchpoint = detectROI.size() + 1;

			for (int i = 0; i < detectROI.size(); i++)//寻找搜索宽度内的最优点
			{
				int distance_x = search_startpoint - bestwidth - detectROI[i].x;
				int distance_y = abs(search_starty - detectROI[i].y);
				/***************测试****************/
				/*if (distance_x>-0.8*bestwidth && distance_x < 1.8*bestwidth && detectROI[i].height > 0.7*bestheight &&  detectROI[i].height<1.6*bestheight){
				printf("bestwidth=%d bestw=%d bestheight=%d  \n", bestwidth, bestw,bestheight);
				printf("x=%d distance_x=%d width=%d height=%d \n", detectROI[i].x, distance_x, detectROI[i].width, detectROI[i].height);
				}*/

				/***************测试****************/
				if (distance_x>-0.6*bestwidth && distance_x < 1.8*bestwidth && detectROI[i].height > 0.7*bestheight &&  detectROI[i].height<1.4*bestheight &&   detectROI[i].width  > 0.7*bestw &&  detectROI[i].width < 1.4*bestw && distance_y<0.4*bestheight)                         //选取x值合适的点，比较它们与最佳位置点的距离选最小的
				{

					int distance = (distance_x)*(distance_x)+0.7*(besty0 - detectROI[i].y)*(besty0 - detectROI[i].y);//我们更看重x的坐标，故y的差值影响的权重降低
					distance = sqrt(distance);
					//					    printf("distance=%d \n", distance);
					if (distance < search_min_distance)
					{
						search_min_distance = distance;
						searchpoint = i;
						if (showSteps)
						{
							if (search_min_distance < 2 * bestwidth)
								printf("search_min_distance=%d \n", search_min_distance);
						}
					}
				}
			}
		}


		/********************************增加后框*****************************/
		search_startpoint = bestx1;
		search_starty = besty1;
		searchpoint = detectROI.size() + 1;
		search_min_distance = image_width;


		for (int i = 0; i < detectROI.size(); i++)
		{
			int distance_x = detectROI[i].x - search_startpoint - bestwidth;
			int distance_y = abs(search_starty - detectROI[i].y);
			/***************测试****************/

			/*	if (distance_x>-0.8*bestwidth && distance_x < 1.8*bestwidth){
			printf("bestwidth=%d bestheight=%d bestw=%d \n", bestwidth, bestheight, bestw);
			printf("distance_x=%d x=%d \n", distance_x, detectROI[i].x);
			printf("width=%d height=%d \n", detectROI[i].width, detectROI[i].height);
			}*/


			/***************测试****************/
			if (distance_x>-0.6*bestwidth && distance_x < 1.8*bestwidth && detectROI[i].height > 0.7*bestheight &&  detectROI[i].height<1.4*bestheight &&   detectROI[i].width  > 0.7*bestw &&  detectROI[i].width < 1.4*bestw && distance_y<0.4*bestheight)  //选取x值合适的点，比较它们与最佳位置点的距离选最小的，加入框高度宽度的筛选
			{
				int distance = (distance_x)*(distance_x)+0.7*(besty1 - detectROI[i].y)*(besty1 - detectROI[i].y);//我们更看重x的坐标，故y的差值影响的权重降低
				distance = sqrt(distance);
				if (distance < search_min_distance)
				{
					search_min_distance = distance;
					searchpoint = i;

				}
			}
		}


		while (searchpoint<(detectROI.size() + 1) && search_min_distance<2 * bestwidth)
		{

			b[bestgroup].push_back(detectROI[searchpoint]);
			search_startpoint = detectROI[searchpoint].x;//更新起始位置
			search_starty = detectROI[searchpoint].y;
			search_min_distance = image_width;
			searchpoint = detectROI.size() + 1;

			for (int i = 0; i < detectROI.size(); i++)//寻找搜索宽度内的最优点
			{
				int distance_x = detectROI[i].x - search_startpoint - bestwidth;
				int distance_y = abs(search_starty - detectROI[i].y);
				/***************测试****************/
				/*if (distance_x < 0.8*bestwidth)
				printf("x=%d width=%d height=%d ", detectROI[i].x, detectROI[i].width, detectROI[i].height);*/
				/***************测试****************/
				if (distance_x>-0.6*bestwidth && distance_x < 1.8*bestwidth && detectROI[i].height > 0.7*bestheight &&  detectROI[i].height<1.4*bestheight &&   detectROI[i].width  > 0.7*bestwidth &&  detectROI[i].width < 1.4*bestwidth && distance_y<0.4*bestheight)                         //选取x值合适的点，比较它们与最佳位置点的距离选最小的
				{
					int distance = (distance_x)*(distance_x)+0.7*(besty1 - detectROI[i].y)*(besty1 - detectROI[i].y);//我们更看重x的坐标，故y的差值影响的权重降低
					distance = sqrt(distance);
					//					printf("distance=%d \n", distance);
					if (distance < search_min_distance)
					{
						search_min_distance = distance;
						searchpoint = i;
						if (showSteps)
						{
							if (search_min_distance < 0.8*bestwidth)
								printf("search_min_distance=%d \n", search_min_distance);
						}
					}
				}
			}
		}
		/*****************在bestgroup的基础上找到前后的框完毕********/

		/********************选出bestgroup中存在包含关系的框*********************/
		vector<int> baohan;
		for (int i = 0; i < b[bestgroup].size(); i++)
			for (int j = 0; j < b[bestgroup].size(); j++)
			{
				CvRect recti = b[bestgroup][i];
				CvRect rectj = b[bestgroup][j];
				/***********vector的移除元素问题*************/
				if (rectj.x >= recti.x && rectj.y >= recti.y && rectj.x + rectj.width < recti.x + recti.width && rectj.y + rectj.height < recti.y + recti.height)
				{
					baohan.push_back(j);
				}
			}
		vector<CvRect> group;//去除包含关系后的bestgroup存在这里
		for (int i = 0; i < b[bestgroup].size(); i++) {
			int kkk = 0;//若kkk为0则说明这个框不存在被包含关系
			for (int j = 0; j < baohan.size(); j++)
			{
				if (i == baohan[j])
					kkk++;
			}
			if (kkk == 0)
				group.push_back(b[bestgroup][i]);
		}







		int finalsize = b[bestgroup].size();
		if (showSteps) {
			printf("bestx:\n");
			for (int i = 0; i < b[bestgroup].size(); i++)
				printf("%d \n", b[bestgroup][i].x);
			printf("add=%d \n", (finalsize - presize));
		}
		//		return b[bestgroup];
		return group;

	}
	else
	{
		printf(" not found \n");
		notfound = 0;

		return detectROI;
	}

}

vector<CvRect> roinormalization(vector<CvRect>chooseROI, Mat normalization_detect_obj)
{
	vector<CvRect> normal;
	int image_width = normalization_detect_obj.cols;
	int image_height = normalization_detect_obj.rows;
	int width_average = 0;
	int height_average = 0;
	if (chooseROI.size() > 1) {
		for (int i = 0; i < chooseROI.size(); i++)
		{
			width_average += chooseROI[i].width;
			height_average += chooseROI[i].height;
		}
		width_average = width_average / chooseROI.size();
		height_average = height_average / chooseROI.size();
		if (width_average * 5>height_average * 3)
			height_average = width_average * 5 / 3;
		else
			width_average = height_average * 3 / 5;
		for (int i = 0; i < chooseROI.size(); i++)
		{
			CvRect roi_normal;
			roi_normal.x = chooseROI[i].x + 0.5*chooseROI[i].width - 0.5*width_average;
			if (roi_normal.x < 0)
				roi_normal.x = 0;
			roi_normal.y = chooseROI[i].y + 0.5*chooseROI[i].height - 0.5*height_average;
			if (roi_normal.y < 0)
				roi_normal.y = 0;
			roi_normal.width = width_average;
			if (roi_normal.width + roi_normal.x>image_width)
				roi_normal.width = image_width - roi_normal.x;
			roi_normal.height = height_average;
			if (roi_normal.height + roi_normal.y>image_height)
				roi_normal.height = image_height - roi_normal.y;
			normal.push_back(roi_normal);
		}
		return normal;
	}
	else
		return chooseROI;
}

vector<CvRect> roicomplete(vector<CvRect>roinormalization, Mat normalization_detect_obj)
{
	int image_width = normalization_detect_obj.cols;
	int image_height = normalization_detect_obj.rows;
	if (roinormalization.size() > 1)
	{
		int roiwidth = roinormalization[0].width;
		int roiheight = roinormalization[0].height;
		vector<CvRect> com;//com为小框暂时的存储
		for (int i = 0; i < roinormalization.size(); i++)
		{
			com.push_back(roinormalization[i]);
		}

		if (showSteps)
			cout << "首尾中间补齐前，检测到的字符个数为：" << roinormalization.size() << endl;
		int sum = 0;
		int avg_distance_of_chars = 0;
		if (roinormalization.size() > 2) {
			int numSum = 0;

			for (int i = 0; i<roinormalization.size() - 1; i++) { //将最后一位排除掉
				if ((roinormalization[i + 1].x - roinormalization[i].x - roinormalization[i].width) < roiwidth * 0.1) {
					if (showSteps)
						cout << "第" << i << "个间距是：" << (roinormalization[i + 1].x - (roinormalization[i].x + roinormalization[i].width)) << endl;
					sum += (roinormalization[i + 1].x - (roinormalization[i].x + roinormalization[i].width));
					numSum++;
				}
			}
			//考虑最后一位（前面的for循环已经包括最后一位了不用考虑了。。。。）
			/*if ((roinormalization[roinormalization.size() - 1].x - roinormalization[roinormalization.size() - 2].x - roinormalization[roinormalization.size() - 2].width) < roiwidth * 0.1)  {
			if (showSteps)
			cout << "第" << roinormalization.size() - 1 << "个间距是：" << roinormalization[roinormalization.size() - 1].x - roinormalization[roinormalization.size() - 2].x - roinormalization[roinormalization.size() - 2].width << endl;
			sum += (roinormalization[roinormalization.size() - 1].x - roinormalization[roinormalization.size() - 2].x - roinormalization[roinormalization.size() - 2].width);
			numSum++;
			}*/

			avg_distance_of_chars = cvFloor((double)sum / (double)numSum);
		}
		else if (roinormalization.size() == 2) {
			avg_distance_of_chars = roinormalization[1].x - roinormalization[0].x - roinormalization[0].width;
		}
		if (showSteps)
			cout << "平均间距为：" << avg_distance_of_chars << endl;



		for (int i = 1; i < roinormalization.size(); i++)//补中间
		{
			int distance = roinormalization[i].x - roinormalization[i - 1].x - roiwidth;//distance为当前的框框左上角与前一个框框右上角之间的x轴距离，若重叠则为负数
			if (showSteps) {
				cout << "distance " << distance << endl;
				cout << "******************　" << roinormalization[i].x - 2.5*(roiwidth + avg_distance_of_chars) << " " << roinormalization[0].x << " " << 0.1*roiwidth << endl;
			}
			if (((i == 1) || (i == 2)) && (distance > 0.1*roiwidth))
			{
				int j = (distance + 0.2*roiwidth) / (0.8 * roiwidth);//当距离在0.6-1.4时补一位，1.4-2.2补二位
				if (showSteps) {
					cout << roinormalization[i].x - roinormalization[i - 1].x - roiwidth << " " << (0.6 * roiwidth) << " " << j << endl;
					cout << (roinormalization[i].x - roinormalization[i - 1].x) % cvRound(0.6*(roiwidth + 2 * avg_distance_of_chars)) << " "
						<< cvRound(0.6*(roiwidth + 2 * avg_distance_of_chars)) << " " << (roiwidth + 2 * avg_distance_of_chars) << endl;
				}
				if ((distance  > 0.6*(roiwidth + 1.5 * avg_distance_of_chars)) && (j == 0))//一般说来平均间距为-0.2w,当前排框间距离超过0.3w时可以补一位
				{
					j++;
				}
				if (showSteps) {
					cout << "add 1:" << j << " CvRect" << endl;
				}

				for (int n = 0; n < j; n++) {   //假设j = 1，则空隙的中心点求出来

					CvPoint centerP;
					centerP.x = roinormalization[i].x - (2 * (j - n) - 1) * (roinormalization[i].x - (roinormalization[i - 1].x + roiwidth)) / (2 * j);
					centerP.y = (roinormalization[i].y + roinormalization[i - 1].y) / 2;

					CvRect Roi;
					Roi.x = centerP.x - roiwidth / 2;
					Roi.y = centerP.y;
					Roi.width = roiwidth;
					Roi.height = roiheight;

					com.push_back(Roi);
				}

			}

			else if ((roinormalization[i].x - 2.7*(roiwidth + avg_distance_of_chars) > roinormalization[0].x) && (distance > 0.01*roiwidth))//离第一个框比较远，也就是靠后面的4位车牌
			{
				//int j = (roinormalization[i].x - roinormalization[i - 1].x - roiwidth - 2 * avg_distance_of_chars) / (0.6 * roiwidth);//换一个算法
				int j = (roinormalization[i].x - roinormalization[i - 1].x - 0.5*roiwidth) / (roiwidth + avg_distance_of_chars);//换一个算法
																																//int j = (distance + 0.6*roiwidth) / (0.9 * roiwidth);
				if (showSteps) {
					cout << roinormalization[i].x - roinormalization[i - 1].x - roiwidth - 2 * avg_distance_of_chars << " " << (0.6 * roiwidth) << " " << j << endl;
					cout << (roinormalization[i].x - roinormalization[i - 1].x) % cvRound(0.8*(roiwidth + 2 * avg_distance_of_chars)) << " "
						<< cvRound(0.6*(roiwidth + 2 * avg_distance_of_chars)) << " " << (roiwidth + 2 * avg_distance_of_chars) << endl;
				}
				if (((roinormalization[i].x - roinormalization[i - 1].x - roiwidth)  > 0.6*(roiwidth + 2 * avg_distance_of_chars)) && (j == 0)) {
					j++;
				}
				if (showSteps) {
					cout << "add 2:" << j << " CvRect" << endl;
				}

				for (int n = 0; n < j; n++) {   //假设j = 1，则空隙的中心点求出来

					CvPoint centerP;
					centerP.x = roinormalization[i].x - (2 * (j - n) - 1) * (roinormalization[i].x - (roinormalization[i - 1].x + roiwidth)) / (2 * j);
					centerP.y = (roinormalization[i].y + roinormalization[i - 1].y) / 2;

					CvRect Roi;
					Roi.x = centerP.x - roiwidth / 2;
					Roi.y = centerP.y;
					Roi.width = roiwidth;
					Roi.height = roiheight;

					com.push_back(Roi);
				}
			}
		}
		if (com.size() == 6)//补前后
		{
			int size = com.size();
			CvRect front, front2;//front2在front前面（如果有必要添加的话）
			CvRect behind, behind2;//behind2在behind后面（如果有必要添加的话）
			if (roinormalization[1].x - roinormalization[0].x > roiwidth)//前面第一个框和第二个框不相连
			{
				if (roinormalization[0].x - 0.8*roiwidth > 0)
					front.x = roinormalization[0].x - 0.8*roiwidth;
				else
					front.x = 0;
				front.y = 2 * roinormalization[0].y - roinormalization[1].y;
				if (front.y < 0)
					front.y = 0;
				if (front.y + roiheight>image_height)
					front.y = image_height - roiheight;
				front.width = roiwidth;
				front.height = roiheight;
				if (roinormalization[0].x - front.x > 0.6*roiwidth)
				{
					com.insert(com.begin(), front);//满足添加第一个的条件 观察是否添加第二个

				}
			}
			else //相连
			{
				int ddd = roinormalization[1].x - roinormalization[0].x;
				if (ddd > 0.8*roiwidth)
					ddd = 0.8*roiwidth;
				if (roinormalization[0].x - ddd > 0)
					front.x = roinormalization[0].x - ddd;
				else
					front.x = 0;
				front.y = 2 * roinormalization[0].y - roinormalization[1].y;
				if (front.y < 0)
					front.y = 0;
				if (front.y + roiheight>image_height)
					front.y = image_height - roiheight;
				front.width = roiwidth;
				front.height = roiheight;
				if (roinormalization[0].x - front.x>0.6*ddd)
					com.insert(com.begin(), front);
			}
			if (roinormalization[roinormalization.size() - 1].x - roinormalization[roinormalization.size() - 2].x > roiwidth)//倒数第一和倒数第二的框框不相连
			{
				if (roinormalization[roinormalization.size() - 1].x + 2 * roiwidth < image_width)
					behind.x = roinormalization[roinormalization.size() - 1].x + roiwidth;
				else
					behind.x = image_width - roiwidth;
				behind.y = 2 * roinormalization[roinormalization.size() - 1].y - roinormalization[roinormalization.size() - 2].y;
				if (behind.y < 0)
					behind.y = 0;
				if (behind.y + roiheight>image_height)
					behind.y = image_height - roiheight;
				behind.width = roiwidth;
				behind.height = roiheight;
				if (behind.x - roinormalization[roinormalization.size() - 1].x>0.6*roiwidth)
					com.push_back(behind);
			}
			else//倒数第一和倒数第二的框框相连
			{
				int fff = roinormalization[roinormalization.size() - 1].x - roinormalization[roinormalization.size() - 2].x;//框框间距为fff
				if (roinormalization[roinormalization.size() - 1].x + roiwidth + fff < image_width)//未越界
				{
					behind.x = roinormalization[roinormalization.size() - 1].x + fff;
					behind.y = 2 * roinormalization[roinormalization.size() - 1].y - roinormalization[roinormalization.size() - 2].y;
					if (behind.y < 0)
						behind.y = 0;
					if (behind.y + roiheight>image_height)
						behind.y = image_height - roiheight;
					behind.width = roiwidth;
					behind.height = roiheight;
					com.push_back(behind);
				}
				else //越界
				{
					behind.width = roiwidth;
					behind.height = roiheight;
					behind.y = 2 * roinormalization[roinormalization.size() - 1].y - roinormalization[roinormalization.size() - 2].y;
					if (behind.y < 0)
						behind.y = 0;
					if (behind.y + roiheight>image_height)
						behind.y = image_height - roiheight;
					behind.x = image_width - roiwidth;
					com.push_back(behind);
				}
			}

		}
		if (com.size() == 5 || com.size() == 4)
		{
			int size = com.size();
			CvRect front, front2;//front2在front前面（如果有必要添加的话）
			CvRect behind, behind2;//behind2在behind后面（如果有必要添加的话）
			if (roinormalization[1].x - roinormalization[0].x > roiwidth)//前面第一个框和第二个框不相连
			{
				if (roinormalization[0].x - 0.8*roiwidth > 0)
					front.x = roinormalization[0].x - 0.8*roiwidth;
				else
					front.x = 0;
				front.y = 2 * roinormalization[0].y - roinormalization[1].y;
				if (front.y < 0)
					front.y = 0;
				if (front.y + roiheight>image_height)
					front.y = image_height - roiheight;
				front.width = roiwidth;
				front.height = roiheight;
				if (roinormalization[0].x - front.x > 0.7*roiwidth)
				{
					com.insert(com.begin(), front);//满足添加第一个的条件 观察是否添加第二个
					front2.x = front.x - 0.8*roiwidth;
					front2.y = 2 * front.y - roinormalization[0].y;
					front2.width = roiwidth;
					front2.height = roiheight;
					if (front2.x >= 0 && front2.y >= 0 && front2.y + roiheight< image_height)
						com.insert(com.begin(), front2);

				}
			}
			else //相连
			{
				/*int ddd = roinormalization[1].x - roinormalization[0].x;
				if (ddd > 0.8*roiwidth)
				ddd = 0.8*roiwidth;*/
				if (roinormalization[0].x - 1.1*roiwidth > 0)
					front.x = roinormalization[0].x - 1.1*roiwidth;
				else
					front.x = 0;
				front.y = 2 * roinormalization[0].y - roinormalization[1].y;
				if (front.y < 0)
					front.y = 0;
				if (front.y + roiheight>image_height)
					front.y = image_height - roiheight;
				front.width = roiwidth;
				front.height = roiheight;
				if (roinormalization[0].x - front.x > 0.7*roiwidth)
				{
					com.insert(com.begin(), front);
					front2.x = front.x - 0.8*roiwidth;
					front2.y = 2 * front.y - roinormalization[0].y;
					front2.width = roiwidth;
					front2.height = roiheight;
					if (front2.x >= 0 && front2.y >= 0 && front2.y + roiheight< image_height)
						com.insert(com.begin(), front2);
				}
			}
			if (roinormalization[roinormalization.size() - 1].x - roinormalization[roinormalization.size() - 2].x > roiwidth)//倒数第一和倒数第二的框框不相连
			{
				if (roinormalization[roinormalization.size() - 1].x + 2 * roiwidth < image_width)
					behind.x = roinormalization[roinormalization.size() - 1].x + roiwidth;
				else
					behind.x = image_width - roiwidth;
				behind.y = 2 * roinormalization[roinormalization.size() - 1].y - roinormalization[roinormalization.size() - 2].y;
				if (behind.y < 0)
					behind.y = 0;
				if (behind.y + roiheight > image_height)
					behind.y = image_height - roiheight;
				behind.width = roiwidth;
				behind.height = roiheight;
				if (behind.x - roinormalization[roinormalization.size() - 1].x > 0.95*roiwidth) {
					com.push_back(behind);
					behind2.x = behind.x + roiwidth;
					behind2.y = 2 * behind.y - roinormalization[roinormalization.size() - 1].y;
					behind2.width = roiwidth;
					behind2.height = roiheight;
					if (behind2.x + roiwidth<image_width && behind2.y >= 0 && behind2.y + roiheight < image_height)
						com.push_back(behind2);

				}
			}
			else//倒数第一和倒数第二的框框相连
			{
				int fff = roinormalization[roinormalization.size() - 1].x - roinormalization[roinormalization.size() - 2].x;//框框间距为fff
				if (roinormalization[roinormalization.size() - 1].x + roiwidth + fff < image_width)//未越界
				{
					behind.x = roinormalization[roinormalization.size() - 1].x + fff;
					behind.y = 2 * roinormalization[roinormalization.size() - 1].y - roinormalization[roinormalization.size() - 2].y;
					if (behind.y < 0)
						behind.y = 0;
					if (behind.y + roiheight>image_height)
						behind.y = image_height - roiheight;
					behind.width = roiwidth;
					behind.height = roiheight;
					com.push_back(behind);
					behind2.x = behind.x + fff;
					behind2.y = 2 * behind.y - roinormalization[roinormalization.size() - 1].y;
					behind2.width = roiwidth;
					behind2.height = roiheight;
					if (behind2.x + roiwidth<image_width && behind2.y >= 0 && behind2.y + roiheight < image_height)
						com.push_back(behind2);
				}
				else //越界
				{
					behind.width = roiwidth;
					behind.height = roiheight;
					behind.y = 2 * roinormalization[roinormalization.size() - 1].y - roinormalization[roinormalization.size() - 2].y;
					if (behind.y < 0)
						behind.y = 0;
					if (behind.y + roiheight>image_height)
						behind.y = image_height - roiheight;
					behind.x = image_width - roiwidth;
					if (behind.x - roinormalization[roinormalization.size() - 1].x > 0.7*roiwidth)
						com.push_back(behind);
				}
			}

		}

		return com;
	}
	else
		return roinormalization;


}

vector<CvRect> roicomplete2(vector<CvRect>roinormalization, Mat normalization_detect_obj)//使用另一个办法来补全车牌
{
	int image_width = normalization_detect_obj.cols;
	int image_height = normalization_detect_obj.rows;
	if (roinormalization.size() > 1)
	{
		int roiwidth = roinormalization[0].width;
		int roiheight = roinormalization[0].height;
		vector<CvRect> com;//com为小框暂时的存储
		for (int i = 0; i < roinormalization.size(); i++)
		{
			com.push_back(roinormalization[i]);
		}
		for (int i = 1; i < roinormalization.size(); i++)//补中间
		{
			int distance = roinormalization[i].x - roinormalization[i - 1].x - roiwidth;
			if (distance>0.4*roiwidth)
			{
				CvRect buchong;
				buchong.x = (roinormalization[i].x + roinormalization[i - 1].x) / 2;
				buchong.y = (roinormalization[i].y + roinormalization[i - 1].y) / 2;
				buchong.width = roiwidth;
				buchong.height = roiheight;
				com.push_back(buchong);

			}
			else if (i > 2 && distance > 0.1*roiwidth)//后面的间隙允许小一些
			{
				CvRect buchong2;
				buchong2.x = (roinormalization[i].x + roinormalization[i - 1].x) / 2;
				buchong2.y = (roinormalization[i].y + roinormalization[i - 1].y) / 2;
				buchong2.width = roiwidth;
				buchong2.height = roiheight;
				com.push_back(buchong2);
			}
		}
		//if (com.size() == 5)
		//{
		//	CvRect add1;
		//	CvRect add2;
		//	int carplatepoint = -1;//这个变量来判断车牌点的位置
		//	for (int i = 1; i < roinormalization.size(); i++)
		//	{
		//		int pointnumber = 0;

		//	}

		//}
		//if (com.size() == 5 || com.size() == 6)//补前后
		//{
		//	CvRect front;
		//	CvRect behind;
		//	if (roinormalization[1].x - roinormalization[0].x > roiwidth)
		//	{
		//		if (roinormalization[0].x - 0.8*roiwidth > 0)
		//			front.x = roinormalization[0].x - 0.8*roiwidth;
		//		else
		//			front.x = 0;
		//		front.y = 2 * roinormalization[0].y - roinormalization[1].y;
		//		if (front.y < 0)
		//			front.y = 0;
		//		if (front.y + roiheight>image_height)
		//			front.y = image_height - roiheight;
		//		front.width = roiwidth;
		//		front.height = roiheight;
		//		if (roinormalization[0].x - front.x>0.7*roiwidth)
		//			com.insert(com.begin(), front);
		//	}
		//	else
		//	{
		//		int ddd = roinormalization[1].x - roinormalization[0].x;
		//		if (ddd > 0.8*roiwidth)
		//			ddd = 0.8*roiwidth;
		//		if (roinormalization[0].x - ddd > 0)
		//			front.x = roinormalization[0].x - ddd;
		//		else
		//			front.x = 0;
		//		front.y = 2 * roinormalization[0].y - roinormalization[1].y;
		//		if (front.y < 0)
		//			front.y = 0;
		//		if (front.y + roiheight>image_height)
		//			front.y = image_height - roiheight;
		//		front.width = roiwidth;
		//		front.height = roiheight;
		//		if (roinormalization[0].x - front.x>0.8*ddd)
		//			com.insert(com.begin(), front);
		//	}
		//	if (roinormalization[roinormalization.size() - 1].x - roinormalization[roinormalization.size() - 2].x > roiwidth)//倒数第一和倒数第二的框框不相连
		//	{
		//		if (roinormalization[roinormalization.size() - 1].x + 2 * roiwidth < image_width)
		//			behind.x = roinormalization[roinormalization.size() - 1].x + roiwidth;
		//		else
		//			behind.x = image_width - roiwidth;
		//		behind.y = 2 * roinormalization[roinormalization.size() - 1].y - roinormalization[roinormalization.size() - 2].y;
		//		if (behind.y < 0)
		//			behind.y = 0;
		//		if (behind.y + roiheight>image_height)
		//			behind.y = image_height - roiheight;
		//		behind.width = roiwidth;
		//		behind.height = roiheight;
		//		if (behind.x - roinormalization[roinormalization.size() - 1].x>0.95*roiwidth)
		//			com.push_back(behind);
		//	}
		//	else//倒数第一和倒数第二的框框相连
		//	{
		//		int fff = roinormalization[roinormalization.size() - 1].x - roinormalization[roinormalization.size() - 2].x;//框框间距为fff
		//		if (roinormalization[roinormalization.size() - 1].x + roiwidth + fff < image_width)//未越界
		//		{
		//			behind.x = roinormalization[roinormalization.size() - 1].x + fff;
		//			behind.y = 2 * roinormalization[roinormalization.size() - 1].y - roinormalization[roinormalization.size() - 2].y;
		//			if (behind.y < 0)
		//				behind.y = 0;
		//			if (behind.y + roiheight>image_height)
		//				behind.y = image_height - roiheight;
		//			behind.width = roiwidth;
		//			behind.height = roiheight;
		//			com.push_back(behind);
		//		}
		//		else if (com.size() == 6)//越界
		//		{
		//			behind.width = roiwidth;
		//			behind.height = roiheight;
		//			behind.y = 2 * roinormalization[roinormalization.size() - 1].y - roinormalization[roinormalization.size() - 2].y;
		//			if (behind.y < 0)
		//				behind.y = 0;
		//			if (behind.y + roiheight>image_height)
		//				behind.y = image_height - roiheight;
		//			behind.x = image_width - roiwidth;
		//			com.push_back(behind);
		//		}
		//	}

		//}
		return com;
	}
	else
		return roinormalization;


}

vector<CvRect>buchong(vector<CvRect>ROI_choose, Mat normalization_detect_obj)
{
	vector<CvRect> qqq;
	int image_width = normalization_detect_obj.cols;
	int image_height = normalization_detect_obj.rows;
	int srcsize = ROI_choose.size();
	for (int i = 1; i < srcsize; i++) {
		qqq.push_back(ROI_choose[i]);
	}
	for (int i = 1; i < srcsize; i++)
	{
		CvRect left;
		CvRect right;
		CvRect up;
		CvRect down;
		left.x = ROI_choose[i].x - 4;
		left.y = ROI_choose[i].y;
		left.width = ROI_choose[i].width;
		left.height = ROI_choose[i].height;
		if (left.x>0)
			qqq.push_back(left);
		//		printf("x=%d,y=%d", left.x, left.y);


		right.x = ROI_choose[i].x + 4;
		right.y = ROI_choose[i].y;
		right.width = ROI_choose[i].width;
		right.height = ROI_choose[i].height;
		if (right.x + right.width<image_width)
			qqq.push_back(right);

		up.x = ROI_choose[i].x;
		up.y = ROI_choose[i].y - 4;
		up.width = ROI_choose[i].width;
		up.height = ROI_choose[i].height;
		if (up.y > 0)
			qqq.push_back(up);

		down.x = ROI_choose[i].x;
		down.y = ROI_choose[i].y + 4;
		down.width = ROI_choose[i].width;
		down.height = ROI_choose[i].height;
		if (down.y + down.height < image_height)
			qqq.push_back(down);
	}
	return qqq;

}

vector<CvRect>hanzibuchong(vector<CvRect>ROI_choose, Mat normalization_detect_obj) {
	vector<CvRect> qqq;
	int image_width = normalization_detect_obj.cols;
	int image_height = normalization_detect_obj.rows;

	qqq.push_back(ROI_choose[0]);

	for (int i = 0; i < 1; i++)
	{
		CvRect left;
		CvRect right;
		CvRect up;
		CvRect down;
		left.x = ROI_choose[i].x - 4;
		left.y = ROI_choose[i].y;
		left.width = ROI_choose[i].width;
		left.height = ROI_choose[i].height;
		if (left.x>0)
			qqq.push_back(left);
		//		printf("x=%d,y=%d", left.x, left.y);


		right.x = ROI_choose[i].x + 4;
		right.y = ROI_choose[i].y;
		right.width = ROI_choose[i].width;
		right.height = ROI_choose[i].height;
		if (right.x + right.width<image_width)
			qqq.push_back(right);

		up.x = ROI_choose[i].x;
		up.y = ROI_choose[i].y - 4;
		up.width = ROI_choose[i].width;
		up.height = ROI_choose[i].height;
		if (up.y > 0)
			qqq.push_back(up);

		down.x = ROI_choose[i].x;
		down.y = ROI_choose[i].y + 4;
		down.width = ROI_choose[i].width;
		down.height = ROI_choose[i].height;
		if (down.y + down.height < image_height)
			qqq.push_back(down);
	}
	return qqq;

}

char* outputplate(int predict) {
	switch (predict)
	{
	case 0:
		//printf("0");
		return "0";
		break;
	case 1:
		//printf("1");
		return "1";
		break;
	case 2:
		//printf("2");
		return "2";
		break;
	case 3:
		//printf("3");
		return "3";
		break;
	case 4:
		//printf("4");
		return "4";
		break;
	case 5:
		//printf("5");
		return "5";
		break;
	case 6:
		//printf("6");
		return "6";
		break;
	case 7:
		//printf("7");
		return "7";
		break;
	case 8:
		//printf("8");
		return "8";
		break;
	case 9:
		//printf("9");
		return "9";
		break;
	case 10:
		//printf("A");
		return "A";
		break;
	case 11:
		//printf("B");
		return "B";
		break;
	case 12:
		//printf("C");
		return "C";
		break;
	case 13:
		//printf("D");
		return "D";
		break;
	case 14:
		//printf("E");
		return "E";
		break;
	case 15:
		//printf("F");
		return "F";
		break;
	case 16:
		//printf("G");
		return "G";
		break;
	case 17:
		//printf("H");
		return "H";
		break;
	case 18:
		//printf("J");
		return "J";
		break;
	case 19:
		//printf("K");
		return "K";
		break;
	case 20:
		//printf("L");
		return "L";
		break;
	case 21:
		//printf("M");
		return "M";
		break;
	case 22:
		//printf("N");
		return "N";
		break;
	case 23:
		//printf("P");
		return "P";
		break;
	case 24:
		//printf("Q");
		return "Q";
		break;
	case 25:
		//printf("R");
		return "R";
		break;
	case 26:
		//printf("S");
		return "S";
		break;
	case 27:
		//printf("T");
		return "T";
		break;
	case 28:
		//printf("U");
		return "U";
		break;
	case 29:
		//printf("V");
		return "V";
		break;
	case 30:
		//		printf("W");
		return "W";
		break;
	case 31:
		//		printf("X");
		return "X";
		break;
	case 32:
		//		printf("Y");
		return "Y";
		break;
	case 33:
		//		printf("Z");
		return "Z";
		break;
	case 34:
		//		printf("学");
		return "学";
		break;
	case 35:
		//		printf("挂");
		return "挂";
		break;
	case 36:
		//		printf("警");
		return "警";
		break;
	case -1:
		//		printf("?");
		return "?";
		break;

	}
	//	printf(" ");

}

char* outputhanzi(int predict) {
	switch (predict)
	{
	case 0:
		//printf("藏");
		return "藏";
		break;
	case 1:
		//printf("川");
		return "川";
		break;
	case 2:
		//printf("鄂");
		return "鄂";
		break;
	case 3:
		//printf("甘");
		return "甘";
		break;
	case 4:
		//printf("赣");
		return "赣";
		break;
	case 5:
		//printf("贵");
		return "贵";
		break;
	case 6:
		//printf("桂");
		return "桂";
		break;
	case 7:
		//printf("黑");
		return "黑";
		break;
	case 8:
		//printf("沪");
		return "沪";
		break;
	case 9:
		//printf("吉");
		return "吉";
		break;
	case 10:
		//printf("冀");
		return "冀";
		break;
	case 11:
		//printf("津");
		return "津";
		break;
	case 12:
		//printf("晋");
		return "晋";
		break;
	case 13:
		//printf("京");
		return "京";
		break;
	case 14:
		//printf("辽");
		return "辽";
		break;
	case 15:
		//printf("鲁");
		return "鲁";
		break;
	case 16:
		//printf("蒙");
		return "蒙";
		break;
	case 17:
		//printf("闽");
		return "闽";
		break;
	case 18:
		//printf("宁");
		return "宁";
		break;
	case 19:
		//printf("青");
		return "青";
		break;
	case 20:
		//printf("琼");
		return "琼";
		break;
	case 21:
		//printf("陕");
		return "陕";
		break;
	case 22:
		//printf("苏");
		return "苏";
		break;
	case 23:
		//printf("皖");
		return "皖";
		break;
	case 24:
		//printf("湘");
		return "湘";
		break;
	case 25:
		//printf("新");
		return "新";
		break;
	case 26:
		//printf("渝");
		return "渝";
		break;
	case 27:
		//printf("豫");
		return "豫";
		break;
	case 28:
		//printf("粤");
		return "粤";
		break;
	case 29:
		//printf("云");
		return "云";
		break;
	case 30:
		//printf("浙");
		return "浙";
		break;
	case -1:
		//printf("?");
		return "?";
		break;

	}
	//printf(" ");
}

//BOOL sort_by_x(cv::Point2i point1, cv::Point2i point2) {
//	return (point1.x < point2.x);
//}

void show_choose_step(cv::Mat src, vector<CvRect> ROI_choose, char* windName) {
	if (showSteps)
	{
		cv::Mat shaixuan_obj;
		src.copyTo(shaixuan_obj);

		vector<cv::Scalar> color;
		cv::Scalar magenta = cv::Scalar(255, 0, 255);
		magenta = cv::Scalar(255, 0, 0);// //Draw rectangle around the face			
		color.push_back(magenta);

		magenta = cv::Scalar(0, 255, 0);// //Draw rectangle around the face			
		color.push_back(magenta);

		magenta = cv::Scalar(0, 0, 255);// //Draw rectangle around the face				
		color.push_back(magenta);

		magenta = cv::Scalar(255, 255, 0);// //Draw rectangle around the face			
		color.push_back(magenta);

		magenta = cv::Scalar(255, 0, 255);// //Draw rectangle around the face			
		color.push_back(magenta);

		magenta = cv::Scalar(0, 0, 0);// //Draw rectangle around the face			
		color.push_back(magenta);

		magenta = cv::Scalar(0, 255, 255);// //Draw rectangle around the face			
		color.push_back(magenta);

		magenta = cv::Scalar(100, 13, 200);// //Draw rectangle around the face			
		color.push_back(magenta);

		color.push_back(magenta);
		color.push_back(magenta);
		color.push_back(magenta);
		color.push_back(magenta);

		for (unsigned int j = 0; j < ROI_choose.size(); j++)
		{
			const cv::Rect& single_char_roi = ROI_choose[j];
			printf("x=%d y=%d w=%d h=%d i=%d \n", single_char_roi.x, single_char_roi.y, single_char_roi.width, single_char_roi.height, j);
			cv::Point tl(single_char_roi.x, single_char_roi.y);//Get top-left and bottom-right corner points
			cv::Point br = tl + cv::Point(single_char_roi.width, single_char_roi.height);
			cv::rectangle(shaixuan_obj, tl, br, color[j], 1, 1, 0);
			//detectROI0.push_back(single_char_roi);
		}

		namedWindow(windName, 0);
		imshow(windName, shaixuan_obj);
		//cvWaitKey();
	}
}

void RandomizeIdx(int *idx, int g_cCountTrainingSample)
{
	int i, j;

	srand((unsigned)time(0));

	for (i = 0; i<g_cCountTrainingSample; i++)
	{
		j = int((double)rand() / (double)RAND_MAX*double(g_cCountTrainingSample - 1));

		int temp = idx[i];
		idx[i] = idx[j];
		idx[j] = temp;
	}
}

struct result_
{
	vector<cv::Point2i> point;
	int label = 0;
	cv::Point2i centor = cv::Point2i(0, 0);
	int avg_width = 0;
	int avg_height = 0;
}result_init;

typedef std::pair<string, float> Prediction;
static bool PairCompare(const std::pair<float, int>& lhs,
	const std::pair<float, int>& rhs) {
	return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
	std::vector<std::pair<float, int> > pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(std::make_pair(v[i], static_cast<int>(i)));
	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

	std::vector<int> result;
	for (int i = 0; i < N; ++i)
		result.push_back(pairs[i].second);
	return result;
}

BOOL sort_by_centor(result_ result1, result_ result2) {
	return (result1.centor.x < result2.centor.x);
}
/* Return the top N predictions. */

/* Load the mean file in binaryproto format. */
BOOL sort_by_x2(cv::Point2i point1, cv::Point2i point2) {
	return (point1.x < point2.x);
}

bool sort_by_x(CvRect obj1, CvRect obj2)
{
	return obj1.x < obj2.x;
}

vector<string> split(const string &s, const string &seperator) {
	vector<string> result;
	typedef string::size_type string_size;
	string_size i = 0;

	while (i != s.size()) {
		//找到字符串中首个不等于分隔符的字母；
		int flag = 0;
		while (i != s.size() && flag == 0) {
			flag = 1;
			for (string_size x = 0; x < seperator.size(); ++x)
				if (s[i] == seperator[x]) {
					++i;
					flag = 0;
					break;
				}
		}

		//找到又一个分隔符，将两个分隔符之间的字符串取出；
		flag = 0;
		string_size j = i;
		while (j != s.size() && flag == 0) {
			for (string_size x = 0; x < seperator.size(); ++x)
				if (s[j] == seperator[x]) {
					flag = 1;
					break;
				}
			if (flag == 0)
				++j;
		}
		if (i != j) {
			result.push_back(s.substr(i, j - i));
			i = j;
		}
	}
	return result;
}

int outputplate(string predict) {
	if (predict == "0")	return 0;
	else if (predict == "1")	return 1;
	else if (predict == "2")	return 2;
	else if (predict == "3")	return 3;
	else if (predict == "4")	return 4;
	else if (predict == "5")	return 5;
	else if (predict == "6")	return 6;
	else if (predict == "7")	return 7;
	else if (predict == "8")	return 8;
	else if (predict == "9")	return 9;
	else if (predict == "A")	return 10;
	else if (predict == "B")	return 11;
	else if (predict == "C")	return 12;
	else if (predict == "D")	return 13;
	else if (predict == "E")	return 14;
	else if (predict == "F")	return 15;
	else if (predict == "G")	return 16;
	else if (predict == "H")	return 17;
	else if (predict == "J")	return 18;
	else if (predict == "K")	return 19;
	else if (predict == "L")	return 20;
	else if (predict == "M")	return 21;
	else if (predict == "N")	return 22;
	else if (predict == "P")	return 23;
	else if (predict == "Q")	return 24;
	else if (predict == "R")	return 25;
	else if (predict == "S")	return 26;
	else if (predict == "T")	return 27;
	else if (predict == "U")	return 28;
	else if (predict == "V")	return 29;
	else if (predict == "W")	return 30;
	else if (predict == "X")	return 31;
	else if (predict == "Y")	return 32;
	else if (predict == "Z")	return 33;
	else if (predict == "学")	return 34;
	else if (predict == "挂")	return 35;
	else if (predict == "警")	return 36;
	else return -1;
}

int outputhanzi(string predict) {
	if (predict == "藏")	return 0;
	else if (predict == "川")	return 1;
	else if (predict == "鄂")	return 2;
	else if (predict == "甘")	return 3;
	else if (predict == "赣")	return 4;
	else if (predict == "贵")	return 5;
	else if (predict == "桂")	return 6;
	else if (predict == "黑")	return 7;
	else if (predict == "沪")	return 8;
	else if (predict == "吉")	return 9;
	else if (predict == "冀")	return 10;
	else if (predict == "津")	return 11;
	else if (predict == "晋")	return 12;
	else if (predict == "京")	return 13;
	else if (predict == "辽")	return 14;
	else if (predict == "鲁")	return 15;
	else if (predict == "蒙")	return 16;
	else if (predict == "闽")	return 17;
	else if (predict == "宁")	return 18;
	else if (predict == "青")	return 19;
	else if (predict == "琼")	return 20;
	else if (predict == "陕")	return 21;
	else if (predict == "苏")	return 22;
	else if (predict == "皖")	return 23;
	else if (predict == "湘")	return 24;
	else if (predict == "新")	return 25;
	else if (predict == "渝")	return 26;
	else if (predict == "豫")	return 27;
	else if (predict == "粤")	return 28;
	else if (predict == "云")	return 29;
	else if (predict == "浙")	return 30;
	else return -1;
}

void detectAndDisplay(Mat frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	/*参数image为输入的灰度图像，objects为得到被检测物体的矩形框向量组，
	scaleFactor为每一个图像尺度中的尺度参数，默认值为1.1，
	minNeighbors参数为每一个级联矩形应该保留的邻近个数，默认为3，
	flags对于新的分类器没有用（但目前的haar分类器都是旧版的，
	CV_HAAR_DO_CANNY_PRUNING利用Canny边缘检测器来排除一些边缘很少或者很多的图像区域，
	CV_HAAR_SCALE_IMAGE就是按比例正常检测，
	CV_HAAR_FIND_BIGGEST_OBJECT只检测最大的物体，
	CV_HAAR_DO_ROUGH_SEARCH只做初略检测），默认为0.
	minSize和maxSize用来限制得到的目标区域的范围。*/

	car_cascade.detectMultiScale(frame_gray, faces, 1.15, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(20, 20), Size(170, 170));   // 2. ????


	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
		ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 1, 8, 0);
	}
	//-- Show what you got
	imshow("cascadetest", frame);
}

class Timer {
	using Clock = std::chrono::high_resolution_clock;
public:
	/*! \brief start or restart timer */
	inline void Tic() {
		start_ = Clock::now();
	}
	/*! \brief stop timer */
	inline void Toc() {
		end_ = Clock::now();
	}
	/*! \brief return time in ms */
	inline double Elasped() {
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_);
		return duration.count();
	}

private:
	Clock::time_point start_, end_;
};

//bool rec_char(caffe::Net& net, cv::Mat src, int& predict, double& loss) {
//	auto input = net.blob_by_name("data");
//	input->Reshape({ 1, 1, 35, 21 });//{64, 100, 1, 1}
//	float *data = input->mutable_cpu_data();//识别的数据
//	const int n = input->count();
//	//cv::Mat src = cv::imread("img/char4.jpg", 0);
//
//	cv::Mat src2;
//	src.convertTo(src2, CV_32F);
//	cv::resize(src2, src2, cv::Size(21, 35));
//	for (int i = 0; i < n; ++i) {
//		data[i] = src2.at<float>(i / src2.cols, i%src2.cols) / 256;  /* nd(gen);*///转图数据
//	}
//	// forward
//	/*Timer timer;
//	timer.Tic();*/
//	net.Forward();
//#ifndef US_CPP
//	off_netiof();//close net.txt
//#endif
//				 //timer.Toc();
//				 // visualization
//	auto images = net.blob_by_name("prob");//gconv5 结果层输出
//										   /*std::cout << net.blob_by_name("prob")->shape_string() << std::endl;*/
//	const int num = images->num();
//	const int channels = images->channels();
//	const int height = images->height();
//	const int width = images->width();
//	const int canvas_len = std::ceil(std::sqrt(num));
//	for (int i = 0; i < channels; i++) {
//		if (i == 0) {
//			loss = images->mutable_cpu_data()[i];//取得Loss
//			predict = 0;
//		}
//		else {
//			if (images->mutable_cpu_data()[i]>loss) {
//				loss = images->mutable_cpu_data()[i];
//				predict = i; //得出识别结果
//			}
//		}
//		/*std::cout << images->mutable_cpu_data()[i] << std::endl;*/
//	}
//	return true;
//}





