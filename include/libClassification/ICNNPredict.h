
#pragma once


#ifndef interface
#define CINTERFACE
#define interface struct
#endif

//-------------------------------------------------------------------------------------------------
#ifndef IMPORT
#define IMPORT __declspec(dllimport)
#endif


//-------------------------------------------------------------------------------------------------
#ifndef EXPORT
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif
#endif

#include <vector>
#include <string>

typedef std::pair<std::string, float> Prediction;
typedef std::pair<int, float> PredictionIdx;

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>




 interface ICNNPredict
{
	 virtual bool IsCPUMode() = 0;
	virtual std::vector<Prediction> Classify(const std::string& file, int N = 5) = 0;
	virtual std::vector<Prediction> Classify(const unsigned char* pJPGBuffer, int len, int N = 5) = 0;
	virtual std::vector<Prediction> Classify(const cv::Mat& img, int N = 5) = 0;
	virtual std::vector<std::vector<PredictionIdx> > BatchClassify(const std::vector<cv::Mat>& imgs, int N = 5) = 0;
	virtual std::vector<std::vector<float> > BatchPredict(const std::vector<cv::Mat>& img)=0;
	virtual std::vector<Prediction> CropClassify(const cv::Mat& img, int std_size, int crop_num = 1, int N = 5) = 0;
	virtual std::vector<PredictionIdx> ClassifyRtnIdx(const cv::Mat& img, int N = 5) = 0;

	//virtual std::vector<float> ExtractFeature(const cv::Mat& img, const std::string& strLayerName = "") = 0;
	virtual std::vector<float> GetLayerFeatureMaps(const std::string& strLayerName, std::vector<int>& outshape) = 0;
	virtual int GetFeatureDim() = 0;
	virtual std::vector< std::vector<float> > GetLastBlockFeature(const cv::Mat& img) = 0;
	virtual std::vector<float> GetOutputFeatureMap(const cv::Mat& img, std::vector<int>& outshape) = 0;

	virtual std::vector<std::string> GetLabels() = 0;

	virtual void SetMean(const std::string& mean_file) = 0;

	virtual std::vector<float> Predict(const cv::Mat& img) = 0;
	virtual void GetInputImageSize(int &w, int &h) = 0;

	//advanced operations
	virtual float Pruning(float weight_t, const char* saveas_name=0)=0;
	virtual cv::Mat EstimateReceptiveField(const cv::Mat& img, const std::string& layerName, int x, int y, int idxNeuron = -1,bool islstm=false,int* width_parts=0) = 0;
	virtual void GetLayerFeatureMapSize(int w, int h, const std::string& layerName,int& w1, int& h1)=0;
	virtual void Release()=0;
};

 extern "C" EXPORT ICNNPredict* CreatePredictInstance(const char* model_folder, bool use_gpu);
