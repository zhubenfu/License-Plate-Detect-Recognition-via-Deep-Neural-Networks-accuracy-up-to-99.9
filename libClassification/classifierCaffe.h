#ifndef _CLASSIFIERCAFFE_H_
#define _CLASSIFIERCAFFE_H_



#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <thread>
#include "boost/make_shared.hpp"

//using namespace caffe;  // NOLINT(build/namespaces)
using std::string;
using namespace std;
double cosine(const vector<double>& v1, const vector<double>& v2);

/* Pair (label, confidence) representing a prediction. */
// change: 2 classify ¡ª (label1, confidence1) (label2, confidence2)
typedef std::pair<string, float> Prediction;
class _declspec(dllexport) Classifier {
public:
	Classifier(const string& model_file,
		const string& trained_file,
		const string& mean_file,
		const int& label_files);

	std::vector<float> Classifier::Classify(const cv::Mat& img);

	cv::Size input_geometry_;
private:
	void SetMean(const string& mean_file);

	std::vector<float> Predict(const cv::Mat& img);

	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img,
		std::vector<cv::Mat>* input_channels);

private:
	std::shared_ptr<caffe::Net<float> > net_;
	
	int num_channels_;
	cv::Mat mean_;
	std::vector<vector<string>> labels_; //multi
};



#endif // _CLASSIFIERCAFFE_H_