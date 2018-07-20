#include "classifierCaffe.h"


///////////////////////////////
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> /*用到了time函数，所以要有这个头文件*/
#include <fstream>
#include <sstream>
#include <exception>
#include <vector>
#include <io.h>

#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>



#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

extern void initTrainImage();
extern vector<string> imgNames;
//
//#define showSteps 0
//
using namespace std;
//char * configFile = "config.txt";
//
////读取config文件里的内容--
//char* trainSetPosPath = (char *)malloc(200 * sizeof(char));
//char* templateName = (char *)malloc(200 * sizeof(char));
//int frame_width = 640;
//int frame_height = 480;
//char *model_file = (char *)malloc(200 * sizeof(char));		//model_file = "LightenedCNN_C_deploy.prototxt";
//char *trained_file = (char *)malloc(200 * sizeof(char));	//trained_file = "LightenedCNN_C.caffemodel";
//int label_file = 256;
//void readConfig(char* configFile, char* trainSetPosPath){
//	fstream f;
//	char cstring[1000];
//	int readS = 0;
//	f.open(configFile, fstream::in);
//	char param1[200]; strcpy(param1, "");
//	char param2[200]; strcpy(param2, "");
//	char param3[200]; strcpy(param3, "");
//
//	//--读取第一行：--
//	f.getline(cstring, sizeof(cstring));
//	readS = sscanf(cstring, "%s %s %s", param1, param2, param3);
//	strcpy(trainSetPosPath, param3);
//
//	//--读取第2行：-- 对比人脸
//	f.getline(cstring, sizeof(cstring));
//	readS = sscanf(cstring, "%s %s %s", param1, param2, param3);
//	strcpy(templateName, param3);
//	 
//	//--读取第3行：-- 相机宽
//	f.getline(cstring, sizeof(cstring));
//	readS = sscanf(cstring, "%s %s %d", param1, param2, &frame_width);
//
//	//--读取第4行：-- 相机高
//	f.getline(cstring, sizeof(cstring));
//	readS = sscanf(cstring, "%s %s %d", param1, param2, &frame_height);
//
//	//--读取第5行：-- 训练模型
//	f.getline(cstring, sizeof(cstring));
//	readS = sscanf(cstring, "%s %s %s", param1, param2, param3);
//	strcpy(model_file, param3);
//
//	//--读取第6行：-- 训练权重
//	f.getline(cstring, sizeof(cstring));
//	readS = sscanf(cstring, "%s %s %s", param1, param2, param3);
//	strcpy(trained_file, param3);
//
//	//--读取第6行：-- 特征数量
//	f.getline(cstring, sizeof(cstring));
//	readS = sscanf(cstring, "%s %s %d", param1, param2, &label_file);
//}
//
////遍历config.txt里的根目录下的所有的文件，包括子目录。--
//// 其中子目录的名字就是label，子目录里的文件为label对于的训练测试样本---
//vector<string> imgNames;
//vector<string> imgLists;
//vector<int> imgLabels;
//int labelTemp = 0;
//
//void dfsFolder(string folderPath){
//	_finddata_t FileInfo;
//	string strfind = folderPath + "\\*";
//	long long Handle = _findfirst(strfind.c_str(), &FileInfo);
//	if (Handle == -1L)
//	{
//		cerr << "can not match the folder path" << endl;
//		exit(-1);
//	}
//	do{
//		//判断是否有子目录--
//		if (FileInfo.attrib & _A_SUBDIR)    	{
//			//	cout<<FileInfo.name<<" "<<FileInfo.attrib<<endl;
//			//这个语句很重要--
//			if ((strcmp(FileInfo.name, ".") != 0) && (strcmp(FileInfo.name, "..") != 0))   		{
//				string newPath = folderPath + "\\" + FileInfo.name;
//				cout << FileInfo.name << " " << newPath << endl;
//				//根目录下下的子目录名字就是label名，如果没有子目录则其为根目录下
//				labelTemp = atoi(FileInfo.name);
//				//	printf("%d\n",labelTemp);
//				dfsFolder(newPath);
//			}
//		}
//		else  {
//			string finalName = folderPath + "\\" + FileInfo.name;
//			//将所有的文件名写入一个txt文件--
//			//	cout << FileInfo.name << "\t";
//			//	printf("%d\t",label);
//			//	cout << folderPath << "\\" << FileInfo.name  << " " <<endl;
//			//将文件名字和label名字（子目录名字赋值给向量）--
//			imgLabels.push_back(labelTemp);
//			imgNames.push_back(finalName);
//
//
//			std::stringstream ss;
//			std::string str;
//			ss << labelTemp;
//			ss >> str;
//
//			string finalList = FileInfo.name;
//			imgLists.push_back(finalList);
//
//		}
//	} while (_findnext(Handle, &FileInfo) == 0);
//	_findclose(Handle);
//
//}
//
//void initTrainImage(){
//	readConfig(configFile, trainSetPosPath);
//
//	string folderPath = trainSetPosPath;
//	//	string folderPath = "H:\\char\\poker_rec_char_equalist_test";
//	//	string folderPath = "C:\\planeSample\\recSampleData\\rec";
//	//	string folderPath = "C:\\Users\\Administrator\\Desktop\\LPR\\hu\\";
//	dfsFolder(folderPath);
//
//}
////////////////////////////////////////////


Classifier::Classifier(const string& model_file,
	const string& trained_file,
	const string& mean_file,
	const int& label_file) {
#ifdef CPU_ONLY
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif

	/* Load the network. */
	net_.reset(new caffe::Net<float>(model_file, caffe::TEST));
	net_->CopyTrainedLayersFrom(trained_file);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	//CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	caffe::Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

	/* Load the binaryproto mean file. */
	if (mean_file != ""){
		SetMean(mean_file);
	}



	vector<string> label_array;
	for (int j = 0; j < label_file; j++)
	{
		label_array.push_back("0");
	}

	caffe::Blob<float>* output_layer = net_->output_blobs()[0];
	CHECK_EQ(label_array.size(), output_layer->channels())
		<< "Number of labels is different from the output layer dimension.";
	labels_.push_back(label_array);

}

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

/* Return the top N predictions. */
std::vector<float> Classifier::Classify(const cv::Mat& img) {
	auto output = Predict(img);

	vector<float> v1;
	for (int i = 0; i < output.size(); i++){
		v1.push_back((float)output[i]);
	}

	return v1;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
	caffe::BlobProto blob_proto;
	caffe::ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

	/* Convert from BlobProto to Blob<float> */
	caffe::Blob<float> mean_blob;
	mean_blob.FromProto(blob_proto);
	CHECK_EQ(mean_blob.channels(), num_channels_)
		<< "Number of channels of mean file doesn't match input layer.";

	/* The format of the mean file is planar 32-bit float BGR or grayscale. */
	std::vector<cv::Mat> channels;
	float* data = mean_blob.mutable_cpu_data();
	for (int i = 0; i < num_channels_; ++i) {
		/* Extract an individual channel. */
		cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
		channels.push_back(channel);
		data += mean_blob.height() * mean_blob.width();
	}

	/* Merge the separate channels into a single image. */
	cv::Mat mean;
	cv::merge(channels, mean);

	/* Compute the global mean pixel value and create a mean image
	* filled with this value. */
	cv::Scalar channel_mean = cv::mean(mean);
	mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
	caffe::Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_,
		input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Preprocess(img, &input_channels);

	net_->ForwardPrefilled();

	/* Copy the output layer to a std::vector */
	caffe::Blob<float>* output_layer1 = net_->output_blobs()[0];


	const float* begin1 = output_layer1->cpu_data();
	const float* end1 = begin1 + output_layer1->channels();


	std::vector<float> prob1(begin1, end1);

	return prob1;
}

/* Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation and we
* don't need to rely on cudaMemcpy2D. The last preprocessing
* operation will write the separate channels directly to the input
* layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
	caffe::Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void Classifier::Preprocess(const cv::Mat& img,
	std::vector<cv::Mat>* input_channels) {
	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;

	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);


	cv::Mat sample_normalized;
	if (mean_.data != NULL)
		cv::subtract(sample_float, mean_, sample_normalized);
	else  {
		sample_normalized = sample_float;

		int channels = sample_normalized.channels();
		int nRows = sample_normalized.rows;
		int nCols = sample_normalized.cols* channels;
		if (sample_normalized.isContinuous())
		{
			nCols *= nRows;
			nRows = 1;
		}
		int i, j;
		float* p;
		for (i = 0; i < nRows; ++i)
		{
			p = sample_normalized.ptr<float>(i);
			for (j = 0; j < nCols; ++j)
			{
				//if (label_file == 256)
					p[j] *= 0.00390625;
				//else if (label_file == 512){
				//	p[j] -= 128;
				//	p[j] *= 0.0078125;
				//}
				//  cout << p[j] << " ";
			}
		}
	}

	/* This operation will write the separate BGR planes directly to the
	* input layer of the network because it is wrapped by the cv::Mat
	* objects in input_channels. */
	cv::split(sample_normalized, *input_channels);

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		== net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}


double dotProduct(const vector<float>& v1, const vector<float>& v2)
{
	assert(v1.size() == v2.size());
	float ret = 0.0;
	for (vector<float>::size_type i = 0; i != v1.size(); ++i)
	{
		ret += v1[i] * v2[i];
	}
	return ret;
}

double module(const vector<float>& v)
{
	float ret = 0.0;
	for (vector<float>::size_type i = 0; i != v.size(); ++i)
	{
		ret += v[i] * v[i];
	}
	return sqrt(ret);
}

// 夹角余弦
double cosine(const vector<float>& v1, const vector<float>& v2)
{
	assert(v1.size() == v2.size());
	return dotProduct(v1, v2) / (module(v1) * module(v2));
}


//float CalcSimilarity(float const fc1,
//    float const fc2,
//    long dim) {
//  if (dim == -1) {
//    dim = 2048;
//  }
//  return simd_dot(fc1, fc2, dim)/ (
//		sqrt(simd_dot(fc1, fc1, dim))* 
//		sqrt(simd_dot(fc2, fc2, dim))
//		);
//}
//
//
//float simd_dot(const float* x, const float* y, const long& len) {
//	float inner_prod = 0.0f;
//	__m128 X, Y; // 128-bit values
//	__m128 acc = _mm_setzero_ps(); // set to (0, 0, 0, 0)
//	float temp[4];
//
//	long i;
//	for (i = 0; i + 4 < len; i += 4) {
//		X = _mm_loadu_ps(x + i); // load chunk of 4 floats
//		Y = _mm_loadu_ps(y + i);
//		acc = _mm_add_ps(acc, _mm_mul_ps(X, Y));
//	}
//	_mm_storeu_ps(&temp[0], acc); // store acc into an array
//	inner_prod = temp[0] + temp[1] + temp[2] + temp[3];
//
//	// add the remaining values
//	for (; i < len; ++i) {
//		inner_prod += x[i] * y[i];
//	}
//	return inner_prod;
//}

int main0(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);
	initTrainImage();

	string model_file = "face_deploy_mirror_normalize.prototxt";
	string trained_file = "face_train_test_iter_530000.caffemodel";  //7：6
	string mean_file = "";
	int label_file = 512;
	//string model_file = "LightenedCNN_C_deploy.prototxt";
	//string trained_file = "LightenedCNN_C.caffemodel";  //7：6
	//string mean_file = "";
	//int label_file = 256;

	std::cout << "the labels' channel:" << label_file << std::endl;
	Classifier classifier(model_file, trained_file, mean_file, label_file);

	string file = ".\\face_align\\1.jpg";
	cv::Mat img = cv::imread(file, 1);
	CHECK(!img.empty()) << "Unable to decode image " << file;
	std::cout << "---------- Prediction for "<< file << " ----------" << std::endl;
	vector<float> predictions = classifier.Classify(img);

	int imgNum = imgNames.size();
	for (int iNum = 0; iNum < imgNum; iNum++){

		cout << endl << iNum << " " << imgNames[iNum].c_str() << endl;
		//string file2 = "F:\\MTCNN-master\\vs2013_caffe_BN_multi_label\\water_meter_caffe_old\\face\\3.jpg";
		cv::Mat img2 = cv::imread(imgNames[iNum].c_str(), 1);
		//CHECK(!img2.empty()) << "Unable to decode image " << imgNames[iNum].c_str();

		if (img2.empty()) continue;
		//cv::normalize(img2,img2,255,0,CV_MINMAX);

		
		vector<float> predictions2 = classifier.Classify(img2);

		/* Print the top N predictions. */
		for (size_t i = 0; i < predictions.size(); ++i) {
			std::cout << predictions[i] << " ";
		}
		std::cout << std::endl << std::endl;

		for (size_t i = 0; i < predictions2.size(); ++i) {
			std::cout << predictions2[i] << " ";
		}

		double sim2 = cosine(predictions, predictions2);
		cout << "相似度为2 :" << sim2 << endl;
	}
	return 1;
}


Classifier * init() {
	
	initTrainImage();

	string model_file = "face_deploy_mirror_normalize.prototxt";
	string trained_file = "face_train_test_iter_530000.caffemodel";  //7：6
	string mean_file = "";
	int label_file = 512;
	//string model_file = "LightenedCNN_C_deploy.prototxt";
	//string trained_file = "LightenedCNN_C.caffemodel";  //7：6
	//string mean_file = "";
	//int label_file = 256;

	std::cout << "the labels' channel:" << label_file << std::endl;
	//Classifier classifier(model_file, trained_file, mean_file, label_file);
	Classifier *classifier;
	classifier = new Classifier(model_file, trained_file, mean_file, label_file);

	return classifier;
}


void test_(Classifier *classify) {
	Classifier *classifier = classify;

	string file = ".\\face_align\\1.jpg";
	cv::Mat img = cv::imread(file, 1);
	CHECK(!img.empty()) << "Unable to decode image " << file;
	//std::cout << "---------- Prediction for " << file << " ----------" << std::endl;
	vector<float> predictions = classifier->Classify(img);

	int imgNum = imgNames.size();
	for (int iNum = 0; iNum < imgNum; iNum++) {

		//cout << endl << iNum << " " << imgNames[iNum].c_str() << endl;
		//string file2 = "F:\\MTCNN-master\\vs2013_caffe_BN_multi_label\\water_meter_caffe_old\\face\\3.jpg";
		cv::Mat img2 = cv::imread(imgNames[iNum].c_str(), 1);
		//CHECK(!img2.empty()) << "Unable to decode image " << imgNames[iNum].c_str();

		if (img2.empty()) continue;
		//cv::normalize(img2,img2,255,0,CV_MINMAX);

		/*while (1) */{
			vector<float> predictions2 = classifier->Classify(img2);

			/* Print the top N predictions. */
			/*for (size_t i = 0; i < predictions.size(); ++i) {
				std::cout << predictions[i] << " ";
			}
			std::cout << std::endl << std::endl;

			for (size_t i = 0; i < predictions2.size(); ++i) {
				std::cout << predictions2[i] << " ";
			}*/

			double sim2 = cosine(predictions, predictions2);
			//cout << "相似度为2 :" << sim2 << endl;
		}
	}
}

int main00(int argc, char **argv) {
	::google::InitGoogleLogging(argv[0]);
	Classifier *classifier = init();
	Classifier *classifier2 = init();
	//init_rec();
	//init_detect();

	for (int i = 0; i < 10; i++)
	{
		cout << i << endl;
		std::thread first(test_, classifier); 
		std::thread second(test_, classifier2);
		//std::thread third(test_);


		first.join();
		second.join();
		//third.join();
	}

	return 1;
}

int main000(int argc, char **argv) {
	Phase phase = TEST;
	int gpuid = 0;

#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
	Caffe::SetDevice(gpuid);
#endif

	boost::shared_ptr<Net<float> >feature_net;
	boost::shared_ptr<Net<float> >feature_net1;

	//boost::thread_specific_ptr<Net<float> >feature_net;

	string protopath = "face_deploy_mirror_normalize.prototxt";
	string modelpath = "face_train_test_iter_530000.caffemodel";  //7：6

	feature_net.reset(new Net<float>(protopath, phase));
	feature_net->CopyTrainedLayersFrom(modelpath);

	feature_net1.reset(new Net<float>(protopath, phase));
	feature_net1->CopyTrainedLayersFrom(modelpath);

	const int num = 2;
	/*std::thread * thrcall = new thread[num];
	thrcall[0].set_param(0, feature_net);
	thrcall[0].start();

	thrcall[1].set_param(1, feature_net1);
	thrcall[1].start();

	thrcall[0].join();
	thrcall[1].join();*/

	return 1;
}

#endif  // USE_OPENCV