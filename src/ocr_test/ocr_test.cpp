// ocr_test.cpp : 定义控制台应用程序的入口点。
//

#ifdef _WIN32
#include "stdafx.h"
#include <windows.h> 
#include <direct.h>
#include <io.h>
	#ifndef CPU_ONLY
	#ifdef _DEBUG
	#pragma  comment(lib,"libClassificationd.lib")
	#else
	#pragma  comment(lib,"libClassification.lib")
	#endif
	
	#else

	#ifdef _DEBUG
	#pragma  comment(lib,"libClassificationCPU-MKLd.lib")
	#else
	#pragma  comment(lib,"libClassificationCPU-MKL.lib")
	#endif
	#endif
	
#elif __linux__
	#include <dirent.h>	
#endif


#include "public.h"
#include <map>

#include <libClassification/ICNNPredict.h>
#include <libClassification/caffe-gpu.h>

#include <boost/filesystem.hpp>







#include "bktree.h"
#include "levenshtein.h"

#include <time.h>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <set>
using namespace std;

#include <ctcpp.h>

#define showSteps 0
#define showdemo 1


using namespace std;
using namespace cv;
using namespace boost::filesystem;

vector<string> alphabets;
vector<string> imgNames;
path imageHome;
string inputFile;
int imageIndex = 0; 
int idxBlank = 0;
double factor = 0.709;
int minSize = 40;
int wstd = 0, hstd = 0;
ICNNPredict* pCNN;
MTCNN * detector;

string usagemodel;
int stage=1;

#define showSteps 0

void doImage(const string & file);
void doImageStage3(const string & file);
int testImage(const string & file);
int testImageStage3(const string & file);
int fineAndPredict(const Mat & image, const FaceInfo & faceInfo, vector<float> & output, Rect & roi, bool test);
int main_mtcnn_haar(int argc, char **argv);
string GetPredictString(const vector<float>& fm, int idxBlank, const vector<string>& labels);
void scanImages();
void initTrainImage();


void testAccuracy() {
	cv::TickMeter tm;
	tm.start();
	int total = imgNames.size();
	int error = 0;
	for (size_t i = 0; i < total; i++) {
		string file = imageHome.string() + "/" + imgNames[i];
		error +=  (stage == 3) ? testImageStage3(file) : testImage(file);
	}
	tm.stop();

	string line = "--------------------------------------------------------------------";
	char outbuff[2048];
	snprintf(outbuff, 2048, "summary: test %d images, error rate: %.3f, average cost %f(ms) ", 
		total, error*1.0/total, tm.getTimeMilli()/tm.getCounter() / total);
	cout  << line << endl << outbuff << endl << line << endl;
}

void currImage() {
	path p = imageHome;
	p /= imgNames[imageIndex];
	string file = p.string(); 
	printf("%6d %s\n", imageIndex, file.c_str());
	if (stage == 3) doImageStage3(file); else doImage(file);
}

void nextImage() {
	if (imageIndex < imgNames.size() - 1) {
		imageIndex++;
		currImage();
	}
	else {
		cout << "already at the end of  dataset" << endl;
	}
}
void preImage() {
	if (imageIndex > 0) {
		imageIndex--;
		currImage();
	}
	else {
		cout << "already at the head of dataset" << endl;
	}
}

int main_mtcnn_lstm_ctc(int argc, char ** argv) {
	//google::InitGoogleLogging("--minloglevel=2");   //only output error message
	//google::setlogle
	const char * keys = {
		"{help h usage |      | show help message}"
		"{dataset ds   |      | directory contains image for test}"
		"{inputfile if |      | start test from a file specified }"
		"{mtcnn_model_dir mnd | model_platecar| a directory contains MTCNN model proto and weight files }"
		"{predict_model_dir pmd  | plateCard_test | a directory contains plate character prediction model proto and weight files}"
		"{usagemodel | browse    | browse, test_accuracy}"
		"{stage      | 3       | 1 PNet, 3 ONet, 4 Predict   }"
	};

	CommandLineParser parser(argc, argv, keys);
	imageHome = path(parser.get<string>("dataset"));
	inputFile = parser.get<string>("inputfile");
	usagemodel = parser.get<string>("usagemodel"); 
	stage = parser.get<int>("stage");
	std::string proto_model_dir = parser.get<string>("mtcnn_model_dir");
	detector = new MTCNN(proto_model_dir);

#ifdef CPU_ONLY
	bool usegpu = false;
#else
	bool usegpu = true;
#endif 
	 
	//load model
	string modelfolder = parser.get<string>("predict_model_dir");
	pCNN = CreatePredictInstance(modelfolder.c_str(), usegpu);

	pCNN->GetInputImageSize(wstd, hstd);

	//get alphabet
	alphabets = pCNN->GetLabels();

	vector<string>::const_iterator it = find(alphabets.begin(), alphabets.end(), "blank");
	if (it != alphabets.end())
		idxBlank = (int)(it - alphabets.begin());


	//map<wchar_t, int> mapLabel2IDs;
	//for (size_t i = 0; i < alphabets.size(); i++)
	//{
	//	wchar_t c = 0;
	//	if (alphabets[i] == "blank")
	//		continue;
	//	wstring wlabel = string2wstring(alphabets[i], true);
	//	mapLabel2IDs.insert(make_pair(wlabel[0], i));
	//} 
	
	initTrainImage();
	if (usagemodel == "browse") { 
#ifdef showSteps
		cv::namedWindow("dst_face", 0);
		cv::namedWindow("warp_dstImage", 0);
		cv::namedWindow("warpPerspective", 0);
#endif 
		currImage();
		char k;
		while ((k = cvWaitKey(0)) != 'x') {
			switch (k) {
			case 'a':  preImage();  break;
			case 'f':  nextImage();   break;
			default:   nextImage();  break;
			}
		}
	} 
	else {    
		system("pause");
		testAccuracy();
	}

	return 0;
}
int main(int argc, char **argv) {  
	const char * keys = {
		"{help h usage |   | show help message}"
		"{recogmodel rm   |  lstm_ctc |  harr_cnn or lstm_ctc}" 
	};

	CommandLineParser parser(argc, argv, keys);
	string recogmodel = parser.get<string>("recogmodel"); 
	return recogmodel == "lstm_ctc" ? main_mtcnn_lstm_ctc(argc, argv) : main_mtcnn_haar(argc, argv);
}


int main_mtcnn_haar(int argc, char **argv) {
	return 0;
}

int GetUppercaseNum(const string& str)
{
	int n = 0;
	for (size_t i = 0; i < str.size(); i++)
	{
		if (str[i] >= 'A' && str[i] <= 'Z')
			n++;
	}
	return n;
}


//std::wstring string2wstring(const string& str, bool bSrcIsUTF8 = true)
//{
//#ifdef _WIN32
//	UINT srcCode = bSrcIsUTF8 ? CP_UTF8 : CP_ACP;
//	int len = ::MultiByteToWideChar(srcCode,
//		0,
//		str.c_str(),
//		-1,
//		NULL,
//		0);
//	if (len == 0)
//		return wstring();
//
//	WCHAR* dst = new WCHAR[len];
//	int nRet = ::MultiByteToWideChar(srcCode,
//		0,
//		str.c_str(),
//		-1,
//		dst,
//		len);
//#else
	//printf("=====str====%s,len=%lu\n", str.c_str(), str.size());
//	wstring wstr = convert_mb2wc("utf-8", "ucs-2", str);
	// 	if (wstr.size() == 0)
	// 		wstr = convert_mb2wc("gb2312", "ucs-2", str);
	// 	if(wstr.size()==0)
	// 		wstr = convert_mb2wc("ascii", "ucs-2", str);

//#endif

//	wstring wstr = dst;
	//delete[] dst;


//	return wstr;
//}



 


string int2str(int val)
{
	char buf[64] = "";
	snprintf(buf, sizeof(buf) - 1, "%d", val);
	return buf;
}

int str2int(const string &val)
{
	return atoi(val.c_str());
}


string GetPredictString(const vector<float>& fm, int idxBlank, const vector<string>& labels)
{
	string str;
	for (size_t t = 0; t < fm.size(); t++)
	{
		int idx = t;
		int label = (int)fm[idx] + 0.5f;
		if (label >= 0 && label != idxBlank)
		{
			//str += ("*" + labels[label] + "(" + int2str(label) + ")*");
			str +=  labels[label];
		}
	}
	return str;
}


float GetCTCLoss(float*activations, int timesteps, int alphabet_size, int blank_index_,
	const string& strlabel, const std::map<wchar_t, int>& mapLabel2Idx)
{
	size_t workspace_alloc_bytes_;

	ctcOptions options;
	options.loc = CTC_CPU;
	options.num_threads = 8;
	options.blank_label = blank_index_;

	int len = strlabel.size();
	ctcStatus_t status = CTC::get_workspace_size<float>(&len,
		&timesteps,
		alphabet_size,
		1,
		options,
		&workspace_alloc_bytes_);
	//CHECK_EQ(status, CTC_STATUS_SUCCESS) << "CTC Error: " << ctcGetStatusString(status);
	vector<float> workspace_(workspace_alloc_bytes_);

	vector<int> flat_labels;
	for (size_t i = 0; i < strlabel.size(); i++)
	{
		map<wchar_t, int>::const_iterator it = mapLabel2Idx.find(strlabel[i]);
		if (it != mapLabel2Idx.end())
			flat_labels.push_back(it->second);
	}
	if (flat_labels.size() != strlabel.size())
		return 0;
	float cost = 0;
	status = CTC::compute_ctc_loss_cpu<float>(activations,
		0,
		flat_labels.data(),
		&len,
		&timesteps,
		alphabet_size,
		1,
		&cost,
		workspace_.data(),
		options
		);
	return cost;
}

void test_ocr_english(const string& imgfolder, const string& modelfolder, const string& lexiconfile)
{
#ifdef CPU_ONLY
	bool usegpu = false;
#else
	bool usegpu = true;
#endif

	//load model
	ICNNPredict* pCNN = CreatePredictInstance(modelfolder.c_str(), usegpu);
	int wstd = 0, hstd = 0;
	pCNN->GetInputImageSize(wstd, hstd);

	//get alphabet
	vector<string> alphabets = pCNN->GetLabels();

	int idxBlank = 0;
	vector<string>::const_iterator it = find(alphabets.begin(), alphabets.end(), "blank");
	if (it != alphabets.end())
		idxBlank = (int)(it - alphabets.begin());


	//map<wchar_t, int> mapLabel2IDs;
	//for (size_t i = 0; i < alphabets.size(); i++)
	//{
	//	wchar_t c = 0;
	//	if (alphabets[i] == "blank")
	//		continue;
	//	wstring wlabel = string2wstring(alphabets[i], true);
	//	mapLabel2IDs.insert(make_pair(wlabel[0], i));
	//}


	//init BK-tree of lexicon
	printf("init BK-tree of lexicon\n");
	BKTree* pBKtree = bktree_new(levenshtein_distance);
	ifstream fslexicon(lexiconfile);

	int n = 0;
	int caseoffset = 'A' - 'a';
	string line;

	while (getline(fslexicon, line))
	{
		if (line.size() == 0)
			continue;
		//if(line[line.size()-1]=='\t')
		bktree_add(pBKtree, const_cast<char*>(line.c_str()), line.size());
		n++;
		if (GetUppercaseNum(line) == 0)//全部是小写的，转成大写再添加，转成首字母大写再添加
		{
			line[0] += caseoffset;
			bktree_add(pBKtree, const_cast<char*>(line.c_str()), line.size());
			n++;
			if (line.size() > 1)
			{
				for (size_t i = 1; i < line.size(); i++)
				{
					if (line[i] >= 'a' && line[i] <= 'z')
						line[i] += caseoffset;
				}
				bktree_add(pBKtree, const_cast<char*>(line.c_str()), line.size());
				n++;
			}
		}
		printf("\r%d", n);
	}
	printf("\n");


	int sumspend = 0;
	int nok_lexicon = 0;
	int nok_nolexicon = 0;

	vector<string> imgs;
	//FindAllImages(imgfolder.c_str(), imgs, false);



	for (size_t i=0;i<imgs.size();i++)
	{
		string imgfile = imgs[i];
		cv::Mat img = cv::imread(imgfile, CV_LOAD_IMAGE_COLOR);
		int w = img.cols, h = img.rows;
		if (2 * w <= h)
		{
			cv::transpose(img, img);
			cv::flip(img, img, 1);
			w = img.cols, h = img.rows;
		}


		int w1 = hstd*w / h;
		if (w1 != w && h != hstd)
			cv::resize(img, img, cv::Size(w1, hstd));

		int start = clock();

		vector<int> shape;
		vector<float> pred = pCNN->GetOutputFeatureMap(img, shape);

		int end = clock();
		sumspend += (end - start);


		string strpredict0 = GetPredictString(pred, idxBlank, alphabets);

		printf("[%d/%d]%s\n\torig result: %s\n",i+1,imgs.size(),imgs[i].c_str(), strpredict0.c_str());

		string strpredict = strpredict0;


		int dist = std::min(2, (int)strpredict0.size() / 3);
		vector< BKResult> ress = bktree_query(pBKtree, const_cast<char*>(strpredict0.c_str()), strpredict0.size(), dist);

		float min_ctc_loss = 1000;
		vector<int> outshape;
		vector<float> activitas = pCNN->GetLayerFeatureMaps("fc1x", outshape);;
		int timesteps = outshape[0];
		int min_ctc_idx = -1;
		for (size_t j = 0; j < ress.size(); j++)
		{
			//float ctcloss = GetCTCLoss(activitas.data(), timesteps, alphabets.size(), idxBlank, ress[j].str, mapLabel2IDs);
#ifdef _DEBUG
			//printf("%s, ctc loss=%f\n", ress[j].str.c_str(), ctcloss);
#endif
			//if (ctcloss < min_ctc_loss)
			//{
			//	min_ctc_loss = ctcloss;
			//	min_ctc_idx = (int)j;
			//}
		}

		if (ress.size()>0 && min_ctc_idx >= 0)
			printf("\tdic result: %s\n", ress[min_ctc_idx].str.c_str());

		//printf("%d, mean spend=%.2f",	i+1, sumspend / (float)(i+1));

	}


	bktree_destroy(pBKtree);

}


void test_ocr_chinese(const string& imgfolder, const string& modelfolder)
{
#ifdef CPU_ONLY
	bool usegpu = false;
#else
	bool usegpu = true;
#endif

	//load model
	ICNNPredict* pCNN = CreatePredictInstance(modelfolder.c_str(), usegpu);
	int wstd = 0, hstd = 0;
	pCNN->GetInputImageSize(wstd, hstd);

	//get alphabet
	vector<string> alphabets = pCNN->GetLabels();

	int idxBlank = 0;
	vector<string>::const_iterator it = find(alphabets.begin(), alphabets.end(), "blank");
	if (it != alphabets.end())
		idxBlank = (int)(it - alphabets.begin());


	//map<wchar_t, int> mapLabel2IDs;
	//for (size_t i = 0; i < alphabets.size(); i++)
	//{
	//	wchar_t c = 0;
	//	if (alphabets[i] == "blank")
	//		continue;
	//	wstring wlabel = string2wstring(alphabets[i], true);
	//	mapLabel2IDs.insert(make_pair(wlabel[0], i));
	//}


	int sumspend = 0;
	int nok_lexicon = 0;
	int nok_nolexicon = 0;

	vector<string> imgs;
	//FindAllImages(imgfolder.c_str(), imgs, false);

	for (size_t i = 0; i < imgs.size(); i++)
	{
		string imgfile = imgs[i];
		cv::Mat img = cv::imread(imgfile, CV_LOAD_IMAGE_COLOR);
		int w = img.cols, h = img.rows;
		if (2 * w <= h)
		{
			cv::transpose(img, img);
			cv::flip(img, img, 1);
			w = img.cols, h = img.rows;
		}


		//int w1 = hstd*w / h;
		int w1 = 280;
		if (w1 != w && h != hstd)
			cv::resize(img, img, cv::Size(w1, hstd));

		int start = clock();

		vector<int> shape;
		vector<float> pred = pCNN->GetOutputFeatureMap(img, shape);

		int end = clock();
		sumspend += (end - start);

		string strpredict0 = GetPredictString(pred, idxBlank, alphabets);

		printf("[%d/%d]%s: %s\n", i + 1, imgs.size(), imgs[i].c_str(), strpredict0.c_str());

		cv::namedWindow("img",0);
		cv::imshow("img", img);
		cv::waitKey();

	}
}


int rec_test_main()
{
#if 0
	string imgfolder = "I:\\OCR_Line\\synth_english\\db_read_test\\";
	string modelfolder = "I:\\OCR_Line\\synth_english\\crnn\\crnn_256\\";
	string lexiconfile = "I:\\OCR_Line\\synth_english\\lexicon.txt";

	test_ocr_english(imgfolder, modelfolder, lexiconfile);

#else
	//string imgfolder = "F:\\plate_card_rec\\data\\multilabel_fcn\\right_vision1\\";
	string imgfolder = "C:\\plate_card_BLSTM\\vs2013_caffe_BN_multi_label_kenel_w\\water_meter_caffe_old\\face\\";
	string modelfolder = ".\\plateCard_test\\";
	test_ocr_chinese(imgfolder, modelfolder);
#endif
}
 

void initTrainImage() {
	//readConfig(configFile, trainSetPosPath); 
	//string folderPath = trainSetPosPath;
	//string folderPath = "E:\\test dataset\\JPEGImages";
	scanImages();
} 

int fineAndPredict(const Mat & image, const FaceInfo & faceInfo, vector<float> & output, Rect & roi, bool test) { 
	float x = faceInfo.bbox.x1;
	float y = faceInfo.bbox.y1;
	float w = faceInfo.bbox.x2 - faceInfo.bbox.x1 + 1;
	float h = faceInfo.bbox.y2 - faceInfo.bbox.y1 + 1;

	if (x < 0) x = 0;	if (y < 0) y = 0;
	if ((y + h) > image.rows) h = image.rows - y;
	if ((x + w) > image.cols) w = image.cols - x;
	if (w < 0 || h < 0)  return -1;

	roi.x = x;
	roi.y = y;
	roi.width = w;
	roi.height = h;
	cv::Mat dst_face = Mat(image, roi); 
	FacePts facePts = faceInfo.facePts;
	cv::Point2f srcTri[4];
	cv::Point2f dstTri[4];
	cv::Mat rot_mat(2, 4, CV_32FC1);
	cv::Mat warp_mat(2, 4, CV_32FC1);

	for (int j = 0; j < 4; j++) {
		srcTri[j] = cv::Point2f(facePts.x[j] - roi.x, facePts.y[j] - roi.y);
	} 

	//corp and deskew plate area contains character just
	int padding_x = cvFloor(roi.height * 0.04 * 5);
	int padding_y = cvFloor(roi.height * 0.04 * 2);
	int x0 = 0;		int y0 = 0;
	int x1 = 120;	int y1 = 0;
	int x2 = 120;	int y2 = 48;
	int x3 = 0;		int y3 = 48;

	dstTri[0] = cv::Point2f(x0 + padding_x, y0 + padding_y);
	dstTri[1] = cv::Point2f(x1 + padding_x, y1 + padding_y);
	dstTri[2] = cv::Point2f(x2 + padding_x, y2 + padding_y);
	dstTri[3] = cv::Point2f(x3 + padding_x, y3 + padding_y);

	warp_mat = cv::getAffineTransform(srcTri, dstTri);
	cv::Mat warp_dstImage = cv::Mat::zeros(48 + 2 * padding_y, 120 + 2 * padding_x, dst_face.type());
	cv::warpAffine(dst_face, warp_dstImage, warp_mat, warp_dstImage.size()); 
#ifdef showSteps 
	if (!test) {
		cv::imshow("dst_face", dst_face);
		cv::imshow("warp_dstImage", warp_dstImage);
	}
#endif 

	cv::Mat img = warp_dstImage;
	cv::normalize(img, img, 255, 0, cv::NORM_MINMAX);
	int w_rec = img.cols, h_rec = img.rows;
	if (2 * w_rec <= h_rec)
	{
		cv::transpose(img, img);
		cv::flip(img, img, 1);
		w_rec = img.cols, h_rec = img.rows;
	} 

	//int w1 = hstd*w / h;
	int w1 = 280;
	if (w1 != w_rec && h_rec != hstd) { 
	  cv::resize(img, img, cv::Size(w1, hstd), cv::INTER_CUBIC);
    }

	
	vector<int> shape;
	vector<float> pred = pCNN->GetOutputFeatureMap(img, shape);  
	output.insert(output.begin(), pred.begin(), pred.end());
	
	return 0;  
}

// assume the image has only one plate, and the file name equals the plate number
// return 1 if neither detect plate nor plate number equals the file name 
int testImage(const string & file) {
	cout << file << " ";
	double threshold[3] = { 0.7, 0.8, 0.8 };
	Mat image = imread(file);
	std::vector<FaceInfo> faceInfos;
	cv::Mat mergeImg;
	vector<cv::Mat> splitBGR(image.channels());
	split(image, splitBGR);
	for (int i = 0; i < image.channels(); i++) {
		equalizeHist(splitBGR[i], splitBGR[i]);
	}
	merge(splitBGR, mergeImg);

	detector->Detect(mergeImg, faceInfos, minSize, threshold, factor);
	if (faceInfos.size() == 0) {
		cout << "not detector plate" << endl;
		return 1;
	}
	cv::Mat warp_dstImage;
	for (int i = 0; i < faceInfos.size(); i++) {
		FaceInfo fi = faceInfos[i];
		vector<float> output;
		Rect roi;
		int r = fineAndPredict(image, fi, output, roi, true);
		 
		if (r != -1) {
			string predict = GetPredictString(output, idxBlank, alphabets); 
			path p(file);
			string fn = p.filename().string();
			int i = fn.find_last_of(".");
			if (i != -1) {
				string sn = fn.substr(0, i);
				cout <<" predict plateNo: " << predict << endl;
				return sn != predict;
			}
			else {
				cout << "invalid file name:" << file << endl;
				return 1;
			} 
		}
		else {
			cout << "not parser plate" << endl;
			return 1;
		}		 
	}
	  
	return 0;
}

int testImageStage3(const string & file) {
	Mat img = imread(file);
	if (img.empty()) {
		cerr << "cannot read image: " << file << endl;
		exit(1);
	}
	else {
		cout << file << " ";
	}

	FaceInfo faceinfo;
	faceinfo.bbox.x1 = 0, faceinfo.bbox.y1 = 0;
	faceinfo.bbox.x2 = img.cols, faceinfo.bbox.y2 = img.rows; 
	 
	detector->stage3(faceinfo, img, 0.8);
	char buff[512]; 

	vector<float> output;
	Rect roi;
	int r = fineAndPredict(img, faceinfo, output, roi, false); 
	if (r != -1) {
		string predict = GetPredictString(output, idxBlank, alphabets);
		path p(file);
		string fn = p.filename().string();
		int i = fn.find_last_of(".");
		if (i != -1) {
			string sn = fn.substr(0, i);
			int r = sn != predict;
			if (r) {
				cout << " predict plateNo: " << predict << "  ******* error" << endl;
			}
			else {
				cout << " predict plateNo: " << predict << endl;
			}
			return r;
		}
		else {
			cout << "invalid file name:" << file << endl;
			return 1;
		}
	}
	else {
		cout << "not parser plate" << endl;
		return 1;
	}

	return 0; 
}

void doImageStage3(const string & file) {
	Mat img = imread(file);
	if (img.empty()) {
		cerr << "cannot read image: " << inputFile << endl;
		return;
	}
	
	FaceInfo faceinfo;
	faceinfo.bbox.x1 = 0, faceinfo.bbox.y1 = 0;
	faceinfo.bbox.x2 = img.cols, faceinfo.bbox.y2 = img.rows;

	TickMeter tm;
	tm.start();
	detector->stage3(faceinfo, img, 0.8);
	char buff[512];
	FacePts pts = faceinfo.facePts;
	snprintf(buff, 512, "detector point %f,%f %f,%f %f,%f %f,%f \n",
		pts.x[0], pts.y[0], pts.x[1], pts.y[1], pts.x[2], pts.y[2], pts.x[3], pts.y[3]);
	cout << buff;

	vector<float> output;
	Rect roi;
	int r = fineAndPredict(img, faceinfo, output, roi, false);
	tm.stop();
	char outBuffer[2048];
	if (r != -1) {
		string predict = GetPredictString(output, idxBlank, alphabets);
		snprintf(outBuffer, 2048, "detect plate region (%d, %d) (%d x %d) with confidence %f, predict plate %s ",
			roi.x, roi.y, roi.width, roi.height, faceinfo.bbox.score, predict.c_str());
	}
	else {
		snprintf(outBuffer, 1024, "cannot find plate  for region (%d, %d) (%d x %d)   ", roi.x, roi.y, roi.width, roi.height);
	}
	cout << "cost: " << tm.getTimeMilli() / tm.getCounter() << " (ms) " << outBuffer << endl;

	for (int i = 0; i < 4; i++) {
		cv::circle(img, Point2f(pts.x[i], pts.y[i]), 1, Scalar(52, 52, 255), 2);
	}
	imshow("stage3", img); 
}

void doImage(const string & file){
	cv::TickMeter tm;
	tm.start();

	double threshold[3] = { 0.7, 0.8, 0.8 };
	Mat image = imread(file);
	std::vector<FaceInfo> faceInfos;
	cv::Mat mergeImg;
	vector<cv::Mat> splitBGR(image.channels());
	split(image, splitBGR);
	for (int i = 0; i < image.channels(); i++) {
		equalizeHist(splitBGR[i], splitBGR[i]);
	}
	merge(splitBGR, mergeImg);
 
	detector->Detect(mergeImg, faceInfos, minSize, threshold, factor); 
	if (faceInfos.size() == 0) {
		cout << "could not detect plate box" << endl; 
	}
	cv::Mat warp_dstImage;
	for (int i = 0; i < faceInfos.size(); i++) {
		FaceInfo fi = faceInfos[i];
		vector<float> output;
		Rect roi; 
		int r = fineAndPredict(image, fi, output, roi, false);  
		char outBuffer[2048];
		if (r != -1) {
			string predict = GetPredictString(output, idxBlank, alphabets); 
			snprintf(outBuffer, 2048, "detect plate region (%d, %d) (%d x %d) with confidence %f, predict plate %s",
				roi.x, roi.y, roi.width, roi.height, fi.bbox.score, predict.c_str()); 
		}
		else {			
			snprintf(outBuffer, 1024, "cannot find plate  for region (%d, %d) (%d x %d)  \n ", roi.x, roi.y, roi.width, roi.height);
		}
		cout << outBuffer << endl;
#ifdef  showSteps
		rectangle(image, roi, Scalar(255, 0, 0), 2);
		FacePts fp = fi.facePts;
		for (int j = 0; j < 4; j++) {
			circle(image, Point(fp.x[j], fp.y[j]), 1, Scalar(2255, 255, 0), 2);
		}

#endif //  showSteps
	}
	tm.stop();
	float cost = tm.getTimeMilli() / tm.getCounter();
	cout << "cost " << cost << " (ms) " << endl;

#ifdef  showSteps
	imshow("warpPerspective", image);
#endif 

}

void scanImages(){ 
	for (directory_entry & x : directory_iterator(imageHome)) {
		if (is_regular_file(x.path()) && x.path().has_extension()) {			
			string extName = x.path().extension().string();
			if (extName.find("JPG") != -1 || extName.find("jpg") != -1) {
				imgNames.push_back(x.path().filename().string());
			}
		}
	}
	//std::sort(imgNames.begin(), imgNames.end(), [](const string & a, const string &b) {
	//	return a.compare(b);
	//});

	if (inputFile.size() > 0) {
		for (size_t i = 0; i < imgNames.size(); i++) {
			if (imgNames[i] == inputFile) {
				imageIndex = i;
				break;
			}
		}
		printf("scan and sort %d images, start at %8d %s \n", imgNames.size(), imageIndex, inputFile.c_str());
	}
	else {
		printf("scan and sort %d images\n", imgNames.size());
	}		
}


////////////////////////////////////////////

