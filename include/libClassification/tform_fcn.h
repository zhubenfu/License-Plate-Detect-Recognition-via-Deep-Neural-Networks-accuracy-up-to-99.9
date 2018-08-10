#ifndef TFORM_FCN_H
#define TFORM_FCN_H
#include <opencv2/opencv.hpp>
class matlabexception{};
typedef struct  tfm
{
	cv::Mat forword;//向前映射
	cv::Mat inv;//反向映射
} Tfm;

typedef struct  opt
{
	int order;//向前映射
	int K;//反向映射
} options;
Tfm cp2tform_similarity(cv::Mat src, cv::Mat dst); //src源图 坐标 变形后的图坐标
cv::Mat itransform(cv::Mat img,Tfm tm,int rows,int cols);

cv::Mat onetone(cv::Mat img, cv::Mat point, int size, int emh, int eh);
#endif