// ocr_test.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"

#include <map>
#include <windows.h>
#include "public.h"

#include "ICNNPredict.h"


//#define CPU_ONLY

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


std::wstring string2wstring(const string& str, bool bSrcIsUTF8 = true)
{
#ifdef _WIN32
	UINT srcCode = bSrcIsUTF8 ? CP_UTF8 : CP_ACP;
	int len = ::MultiByteToWideChar(srcCode,
		0,
		str.c_str(),
		-1,
		NULL,
		0);
	if (len == 0)
		return wstring();

	WCHAR* dst = new WCHAR[len];
	int nRet = ::MultiByteToWideChar(srcCode,
		0,
		str.c_str(),
		-1,
		dst,
		len);
#else
	//printf("=====str====%s,len=%lu\n", str.c_str(), str.size());
	wstring wstr = convert_mb2wc("utf-8", "ucs-2", str);
	// 	if (wstr.size() == 0)
	// 		wstr = convert_mb2wc("gb2312", "ucs-2", str);
	// 	if(wstr.size()==0)
	// 		wstr = convert_mb2wc("ascii", "ucs-2", str);

#endif

	wstring wstr = dst;
	delete[]dst;


	return wstr;
}



void FindAllImages(const char *folder, std::vector<std::string>& vImgPaths, bool bSubFolder)
{

#ifdef WIN32
	char szPathName[MAX_PATH];
	strcpy_s(szPathName, folder);
	if (szPathName[strlen(szPathName) - 1] != '\\')
		strcat_s(szPathName, "\\");

	char szFileName[256];
	strcpy_s(szFileName, szPathName);
	strcat_s(szFileName, "*.*");

	int ret = 0;
	WIN32_FIND_DATA wfd;
	HANDLE hFind = FindFirstFile(szFileName, &wfd);
	if (hFind != INVALID_HANDLE_VALUE)
	{
		do
		{
			if (strcmp(wfd.cFileName, ".") == 0 || strcmp(wfd.cFileName, "..") == 0)
				continue;

			if (wfd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
			{
				if (bSubFolder)
				{
					strcpy_s(szFileName, szPathName);
					strcat_s(szFileName, wfd.cFileName);
					FindAllImages(szFileName, vImgPaths, bSubFolder);
				}
			}
			else
			{
				if (strlen(wfd.cFileName) >= 5)
				{
					char *ext3 = wfd.cFileName + strlen(wfd.cFileName) - 3;
					char *ext4 = ext3 - 1;
					if (_stricmp(ext3, "bmp") == 0
						|| _stricmp(ext3, "jpg") == 0
						|| _stricmp(ext3, "JPG") == 0
						|| _stricmp(ext4, "jpeg") == 0
						|| _stricmp(ext4, "JPEG") == 0
						|| _stricmp(ext3, "png") == 0
						|| _stricmp(ext3, "gif") == 0)
					{
						//printf("%s\n", wfd.cFileName);

						char filename[256];
						sprintf_s(filename, "%s%s", szPathName, wfd.cFileName);
						vImgPaths.push_back(filename);
					}
				}
			}
		} while (FindNextFile(hFind, &wfd) != 0);
	}
#else

	DIR* pDir = NULL;
	struct dirent* ent = NULL;

	pDir = opendir(folder);
	if (pDir == 0)
	{
		printf("open folder(%s) FAIL\n", folder);
		return;
	}

	std::string strFolder = folder;
	if (strFolder[strFolder.size() - 1] != '/')
		strFolder += "/";

	while (ent = readdir(pDir))
	{
		if (ent->d_type & DT_DIR)
			continue;

		int len = (int)strlen(ent->d_name);
		if (len >= 5 && strcmp(ent->d_name + len - 4, ".jpg") == 0)
		{
			char filepath[256];
			sprintf(filepath, "%s%s", strFolder.c_str(), ent->d_name);
			vImgPaths.push_back(filepath);
		}
	}

	closedir(pDir);

#endif

}


string int2str(int val)
{
	char buf[64] = "";
	_snprintf(buf, sizeof(buf) - 1, "%d", val);
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


	map<wchar_t, int> mapLabel2IDs;
	for (size_t i = 0; i < alphabets.size(); i++)
	{
		wchar_t c = 0;
		if (alphabets[i] == "blank")
			continue;
		wstring wlabel = string2wstring(alphabets[i], true);
		mapLabel2IDs.insert(make_pair(wlabel[0], i));
	}


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
	FindAllImages(imgfolder.c_str(), imgs, false);



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
			float ctcloss = GetCTCLoss(activitas.data(), timesteps, alphabets.size(), idxBlank, ress[j].str, mapLabel2IDs);
#ifdef _DEBUG
			printf("%s, ctc loss=%f\n", ress[j].str.c_str(), ctcloss);
#endif
			if (ctcloss < min_ctc_loss)
			{
				min_ctc_loss = ctcloss;
				min_ctc_idx = (int)j;
			}
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


	map<wchar_t, int> mapLabel2IDs;
	for (size_t i = 0; i < alphabets.size(); i++)
	{
		wchar_t c = 0;
		if (alphabets[i] == "blank")
			continue;
		wstring wlabel = string2wstring(alphabets[i], true);
		mapLabel2IDs.insert(make_pair(wlabel[0], i));
	}


	int sumspend = 0;
	int nok_lexicon = 0;
	int nok_nolexicon = 0;

	vector<string> imgs;
	FindAllImages(imgfolder.c_str(), imgs, false);

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



///////////////////////////////
#include <caffe/caffe.hpp>
#include "boost/make_shared.hpp"
#include "caffe-gpu.h"

using namespace caffe;



#include <string>
#include <vector>
#include <fstream>
#include <direct.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <fstream>
#include <sstream>
#include <exception>
#include <vector>
#include <io.h>



#ifndef HAAR_CASCADE
#include "common.h"
#include "classifierCaffe.h"
#include "opencv2/opencv.hpp"


typedef std::pair<CvRect, char*> Pred;
struct result_plate {
	CvRect coordinate;
	vector<Pred> one_char;
};


_declspec(dllexport) bool rec_char(Classifier &net, cv::Mat src, int& predict, double& loss) {
	
	vector<float> predictions = net.Classify(src);

	/* Print the top N predictions. */
	/*for (size_t i = 0; i < predictions.size(); ++i) {
		std::cout << predictions[i] << " ";
	}
	std::cout << std::endl << std::endl;*/

	int N = 1;
	std::vector<int> maxN = Argmax(predictions, N);
	for (int i = 0; i < N; ++i) {
		predict = maxN[i];
		loss = predictions[predict];
	}


	return true;
}


#endif // DEBUG



#define showSteps 0

using namespace std;
char * configFile = "config.txt";


char* trainSetPosPath = (char *)malloc(200 * sizeof(char));
char* templateName = (char *)malloc(200 * sizeof(char));
int frame_width = 640;
int frame_height = 480;
char *model_file = (char *)malloc(200 * sizeof(char));		 
char *trained_file = (char *)malloc(200 * sizeof(char));	 
int label_file = 256;

void readConfig(char* configFile, char* trainSetPosPath) {
	fstream f;
	char cstring[1000];
	int readS = 0;
	f.open(configFile, fstream::in);
	char param1[200]; strcpy(param1, "");
	char param2[200]; strcpy(param2, "");
	char param3[200]; strcpy(param3, "");

	//--读取第一行：--
	f.getline(cstring, sizeof(cstring));
	readS = sscanf(cstring, "%s %s %s", param1, param2, param3);
	strcpy(trainSetPosPath, param3);

	//--读取第2行：-- 对比人脸
	f.getline(cstring, sizeof(cstring));
	readS = sscanf(cstring, "%s %s %s", param1, param2, param3);
	strcpy(templateName, param3);

	//--读取第3行：-- 相机宽
	f.getline(cstring, sizeof(cstring));
	readS = sscanf(cstring, "%s %s %d", param1, param2, &frame_width);

	//--读取第4行：-- 相机高
	f.getline(cstring, sizeof(cstring));
	readS = sscanf(cstring, "%s %s %d", param1, param2, &frame_height);

	//--读取第5行：-- 训练模型
	f.getline(cstring, sizeof(cstring));
	readS = sscanf(cstring, "%s %s %s", param1, param2, param3);
	strcpy(model_file, param3);

	//--读取第6行：-- 训练权重
	f.getline(cstring, sizeof(cstring));
	readS = sscanf(cstring, "%s %s %s", param1, param2, param3);
	strcpy(trained_file, param3);

	//--读取第6行：-- 特征数量
	f.getline(cstring, sizeof(cstring));
	readS = sscanf(cstring, "%s %s %d", param1, param2, &label_file);
}

//遍历config.txt里的根目录下的所有的文件，包括子目录。--
// 其中子目录的名字就是label，子目录里的文件为label对于的训练测试样本---
vector<string> imgNames;
vector<string> imgLists;
vector<int> imgLabels;
int labelTemp = 0;

void dfsFolder(string folderPath) {
	_finddata_t FileInfo;
	string strfind = folderPath + "\\*";
	long long Handle = _findfirst(strfind.c_str(), &FileInfo);
	if (Handle == -1L)
	{
		cerr << "can not match the folder path" << endl;
		exit(-1);
	}
	do {
		//判断是否有子目录--
		if (FileInfo.attrib & _A_SUBDIR) {
			//	cout<<FileInfo.name<<" "<<FileInfo.attrib<<endl;
			//这个语句很重要--
			if ((strcmp(FileInfo.name, ".") != 0) && (strcmp(FileInfo.name, "..") != 0)) {
				string newPath = folderPath + "\\" + FileInfo.name;
				cout << FileInfo.name << " " << newPath << endl;
				//根目录下下的子目录名字就是label名，如果没有子目录则其为根目录下
				labelTemp = atoi(FileInfo.name);
				//	printf("%d\n",labelTemp);
				dfsFolder(newPath);
			}
		}
		else {
			string finalName = folderPath + "\\" + FileInfo.name;
			//将所有的文件名写入一个txt文件--
			//	cout << FileInfo.name << "\t";
			//	printf("%d\t",label);
			//	cout << folderPath << "\\" << FileInfo.name  << " " <<endl;
			//将文件名字和label名字（子目录名字赋值给向量）--
			imgLabels.push_back(labelTemp);
			imgNames.push_back(finalName);


			std::stringstream ss;
			std::string str;
			ss << labelTemp;
			ss >> str;

			string finalList = FileInfo.name;
			imgLists.push_back(finalList);

		}
	} while (_findnext(Handle, &FileInfo) == 0);
	_findclose(Handle);

}

void initTrainImage() {
	readConfig(configFile, trainSetPosPath);

	string folderPath = trainSetPosPath;
	//	string folderPath = "H:\\char\\poker_rec_char_equalist_test";
	dfsFolder(folderPath);
}


////////////////////////////////////////////




int detect_test_main(int argc, char **argv) {
	initTrainImage();

	/*::google::InitGoogleLogging(argv[0]);*/
	double threshold[3] = { 0.7, 0.5, 0.3 };
	double factor = 0.709;
	int minSize = 40;
	std::string proto_model_dir = argv[1];
	MTCNN *detector = new MTCNN(proto_model_dir);
	int imgNum = imgNames.size();
	for (int iNum = 0; iNum < imgNum; iNum++) {

		cout << endl << iNum << " " << imgNames[iNum].c_str() << endl;
		cv::VideoCapture capture(imgNames[iNum].c_str());
		//cv::VideoCapture capture(0);
		//capture.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
		//capture.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
		//
		//VideoCapture capture("F:\\MTCNN-master\\vs2013_caffe_BN_multi_label\\water_meter_caffe_old\\\img\\1.avi");
		//检测是否正常打开:成功打开时，isOpened返回ture  
		if (!capture.isOpened())
			cout << "fail to open!" << endl;
		//获取整个帧数  
		//	long totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);
		//	cout << "整个视频共" << totalFrameNumber << "帧" << endl;
		//
		//
		//设置开始帧()  
		//	long frameToStart = 300;
		//	capture.set(CV_CAP_PROP_POS_FRAMES, frameToStart);
		//	cout << "从第" << frameToStart << "帧开始读" << endl;
		//
		//
		//设置结束帧  
		//	int frameToStop = 400000;
		//
		//	if (frameToStop < frameToStart)
		//	{
		//		cout << "结束帧小于开始帧，程序错误，即将退出！" << endl;
		//		return -1;
		//	}
		//	else
		//	{
		//		cout << "结束帧为：第" << frameToStop << "帧" << endl;
		//	}
		//
		//
		//	获取帧率  
		//	double rate = capture.get(CV_CAP_PROP_FPS);
		//	cout << "帧率为:" << rate << endl;



		//定义一个用来控制读取视频循环结束的变量  
		bool stop = false;

		//显示每一帧的窗口  
		cv::namedWindow("Extracted frame_", 0);
		//两帧间的间隔时间:  
		//int delay = 1000/rate;  
		//	int delay = 1000 / rate;
		//	if (rate == 0)  delay = 1;
		//
		//利用while循环读取帧  
		//currentFrame是在循环体中控制读取到指定的帧后循环结束的变量  
		//	long currentFrame = frameToStart;
		//
		//    VideoWriter writer;
		//    writer.open("../result/SuicideSquad.mp4",CV_FOURCC('M', 'J', 'P', 'G'), 25, Size(1280,720), true);
		//承载每一帧的图像  

		cv::Mat image;
		int frame_count = 0;
		while (!stop)
		{
			//读取下一帧  
			if (!capture.read(image))
			{
				cout << "读取视频失败" << endl;
				stop = true;
				continue;
			}

			//imshow("Live", image);
			//waitKey(0);
			//for (int i = 0; i < 100; i++)capture.read(image);
			//
			//cv::flip(image, image,-1);

			std::vector<FaceInfo> faceInfo;
			clock_t t1 = clock();
			//		std::cout << "Detect " << image.rows << "X" << image.cols;
			//
			//image = image.t();

			detector->Detect(image, faceInfo, minSize, threshold, factor);
#ifdef CPU_ONLY
			std::cout << " Time Using CPU: " << (clock() - t1)*1.0 / 1000 << std::endl;
#else
			std::cout << " Time Using GPU-CUDNN: " << (clock() - t1)*1.0 / 1000 << std::endl;
#endif
			cv::Mat dst_face;

			for (int i = 0; i < faceInfo.size(); i++) {

				float x = faceInfo[i].bbox.x1;
				float y = faceInfo[i].bbox.y1;
				float w = faceInfo[i].bbox.x2 - faceInfo[i].bbox.x1 + 1;
				float h = faceInfo[i].bbox.y2 - faceInfo[i].bbox.y1 + 1;

				if (x < 0) x = 0;	if (y < 0) y = 0;
				if ((y + h) > image.rows) h = image.rows - y;
				if ((x + w) > image.cols) w = image.cols - x;
				if (w < 0) continue;
				if (h < 0) continue;

				std::cout << x << " " << y << " " << w << " " << h << std::endl;
				std::cout << image.rows << " " << image.cols << std::endl;

				dst_face = image(cv::Rect(x, y, w, h));
				char dst_name[100];
				_mkdir("face");
				sprintf_s(dst_name, "%s%d%s%d%s%d%s", "face\\", iNum, "_", frame_count++, "_", i, ".jpg");
				cv::imwrite(dst_name, dst_face);


				FacePts facePts = faceInfo[i].facePts;





				cv::Point2f srcTri[4];
				cv::Point2f dstTri[4];
				cv::Mat rot_mat(2, 4, CV_32FC1);
				cv::Mat warp_mat(2, 4, CV_32FC1);

				for (int j = 0; j < 4; j++) {
					srcTri[j] = cv::Point2f(facePts.x[j] - x, facePts.y[j] - y);
				}

				int padding_x = cvFloor(h * 0.04 * 5);
				int padding_y = cvFloor(h * 0.04 * 2);
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
				cv::namedWindow("dst_face", 0);
				cv::imshow("dst_face", dst_face);

				cv::namedWindow("warp_dstImage", 0);
				cv::imshow("warp_dstImage", warp_dstImage);

				cv::waitKey(1);
#endif

				sprintf_s(dst_name, "%s%d%s%d%s%d%s", "face\\", iNum, "_", frame_count++, "_", i, "_warp_dstImage_.jpg");
				cv::imwrite(dst_name, warp_dstImage);

#ifdef warpPerspective

				int img_height = dst_face.rows;
				int img_width = dst_face.cols;

				vector<Point2f> corners(4);

				for (int j = 0; j < 4; j++) {
					corners[j] = cv::Point2f(facePts.x[j] - x, facePts.y[j] - y);
				}

				vector<Point2f> corners_trans(4);
				corners_trans[0] = cv::Point2f(x0 + padding_x, y0 + padding_y);
				corners_trans[1] = cv::Point2f(x1 + padding_x, y1 + padding_y);
				corners_trans[2] = cv::Point2f(x2 + padding_x, y2 + padding_y);
				corners_trans[3] = cv::Point2f(x3 + padding_x, y3 + padding_y);

				getStartTime();
				Mat transform = getPerspectiveTransform(corners, corners_trans);
				//cout << transform << endl;
				Mat resultImage;
				warpPerspective(dst_face, resultImage, transform, Size(120 + 2 * padding_x, 48 + 2 * padding_y), INTER_LINEAR);

				getEndTime();
				std::cout << " 2 :" << dfTim << std::endl;

				namedWindow("warpPerspective", 0);
				imshow("warpPerspective", resultImage);

				cv::waitKey(1);



				sprintf_s(dst_name, "%s%d%s%d%s%d%s", "face\\", iNum, "_", frame_count++, "_", i, "_resultImage_.jpg");
				cv::imwrite(dst_name, resultImage);

#endif





#ifdef showSteps
				for (int j = 0; j < 4; j++) {
					std::cout << facePts.x[j] - x << " " << facePts.y[j] - y << std::endl;
					cv::circle(image, cv::Point(facePts.x[j], facePts.y[j]), 1, cv::Scalar(255, 255, 0), 2);
				}

				cv::rectangle(image, cv::Rect(x, y, w, h), cv::Scalar(255, 0, 0), 2);
				std::cout << x << " " << y << " " << w << " " << h << std::endl;
#endif
			}

			cv::imshow("Extracted frame_", image);

			int c = cv::waitKey(0);
			//按下ESC或者到达指定的结束帧后退出读取视频  
			if ((char)c == 27 /*|| currentFrame > frameToStop*/)
			{
				stop = true;
			}
			//按下按键后会停留在当前帧，等待下一次按键  
			//if (c >= 0)
			//{
			//	waitKey(0);
			//}
			//	currentFrame++;

		}

		//关闭视频文件  
		capture.release();
	}
	return 0;
}


int main(int argc, char **argv) {
	

	
	double threshold[3] = { 0.7, 0.8, 0.8 };
	double factor = 0.709;
	int minSize = 40;
	std::string proto_model_dir = "..\\..\\..\\vs2013_caffe_BN_multi_label_kenel_w\\model_platecar\\";
	MTCNN *detector = new MTCNN(proto_model_dir);




#ifdef CPU_ONLY
	bool usegpu = false;
#else
	bool usegpu = true;
#endif

	//load model
	string modelfolder = ".\\plateCard_test\\";
	ICNNPredict* pCNN = CreatePredictInstance(modelfolder.c_str(), usegpu);
	int wstd = 0, hstd = 0;
	pCNN->GetInputImageSize(wstd, hstd);

	//get alphabet
	vector<string> alphabets = pCNN->GetLabels();

	int idxBlank = 0;
	vector<string>::const_iterator it = find(alphabets.begin(), alphabets.end(), "blank");
	if (it != alphabets.end())
		idxBlank = (int)(it - alphabets.begin());


	map<wchar_t, int> mapLabel2IDs;
	for (size_t i = 0; i < alphabets.size(); i++)
	{
		wchar_t c = 0;
		if (alphabets[i] == "blank")
			continue;
		wstring wlabel = string2wstring(alphabets[i], true);
		mapLabel2IDs.insert(make_pair(wlabel[0], i));
	}


	int sumspend = 0;
	int nok_lexicon = 0;
	int nok_nolexicon = 0;




	initTrainImage();
	int imgNum = imgNames.size();
	int *idx = new int[imgNum];

	for (int i = 0; i < imgNum; i++) {
		idx[i] = i;
	}

	RandomizeIdx(idx, imgNum);
	for (int i = 0; i < imgNum; i++) {
		int iNum = idx[i];
		cout << endl << iNum << " " << imgNames[iNum].c_str() << endl;
		cv::VideoCapture capture(imgNames[iNum].c_str());
		//cv::VideoCapture capture(0);
		//capture.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
		//capture.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
		//
		//VideoCapture capture("F:\\MTCNN-master\\vs2013_caffe_BN_multi_label\\water_meter_caffe_old\\\img\\1.avi");
		//检测是否正常打开:成功打开时，isOpened返回ture  
		if (!capture.isOpened())
			cout << "fail to open!" << endl;
		//获取整个帧数  
		//	long totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);
		//	cout << "整个视频共" << totalFrameNumber << "帧" << endl;
		//
		//
		//设置开始帧()  
		//	long frameToStart = 300;
		//	capture.set(CV_CAP_PROP_POS_FRAMES, frameToStart);
		//	cout << "从第" << frameToStart << "帧开始读" << endl;
		//
		//
		//设置结束帧  
		//	int frameToStop = 400000;
		//
		//	if (frameToStop < frameToStart)
		//	{
		//		cout << "结束帧小于开始帧，程序错误，即将退出！" << endl;
		//		return -1;
		//	}
		//	else
		//	{
		//		cout << "结束帧为：第" << frameToStop << "帧" << endl;
		//	}
		//
		//
		//	获取帧率  
		//	double rate = capture.get(CV_CAP_PROP_FPS);
		//	cout << "帧率为:" << rate << endl;



		//定义一个用来控制读取视频循环结束的变量  
		bool stop = false;

		//显示每一帧的窗口  

		//两帧间的间隔时间:  
		//int delay = 1000/rate;  
		//	int delay = 1000 / rate;
		//	if (rate == 0)  delay = 1;
		//
		//利用while循环读取帧  
		//currentFrame是在循环体中控制读取到指定的帧后循环结束的变量  
		//	long currentFrame = frameToStart;
		//
		//    VideoWriter writer;
		//    writer.open("../result/SuicideSquad.mp4",CV_FOURCC('M', 'J', 'P', 'G'), 25, Size(1280,720), true);
		//承载每一帧的图像  

		cv::Mat image;
		int frame_count = 0;
		while (!stop)
		{
			//读取下一帧  
			if (!capture.read(image))
			{
				cout << "读取视频失败" << endl;
				stop = true;
				continue;
			}

			//imshow("Live", image);
			//waitKey(0);
			//for (int i = 0; i < 100; i++)capture.read(image);
			////
			//cv::flip(image, image,-1);

			std::vector<FaceInfo> faceInfo;
			clock_t t1 = clock();
			//		std::cout << "Detect " << image.rows << "X" << image.cols;
			//
			//image = image.t();

			
			cv::Mat mergeImg;//合并后的图像  
						 //用来存储各通道图片的向量  
			vector<cv::Mat> splitBGR(image.channels());
			//分割通道，存储到splitBGR中  
			split(image, splitBGR);
			//对各个通道分别进行直方图均衡化  
			for (int i = 0; i<image.channels(); i++)
				equalizeHist(splitBGR[i], splitBGR[i]);
			//合并通道  
			merge(splitBGR, mergeImg);

			detector->Detect(mergeImg, faceInfo, minSize, threshold, factor);
#ifdef CPU_ONLY
			std::cout << " Time Using CPU: " << (clock() - t1)*1.0 / 1000 << std::endl;
#else
			std::cout << " Time Using : " << (clock() - t1)*1.0 / 1000 << std::endl;
#endif
			cv::Mat dst_face;

			for (int i = 0; i < faceInfo.size(); i++) {

				float x = faceInfo[i].bbox.x1;
				float y = faceInfo[i].bbox.y1;
				float w = faceInfo[i].bbox.x2 - faceInfo[i].bbox.x1 + 1;
				float h = faceInfo[i].bbox.y2 - faceInfo[i].bbox.y1 + 1;

				std::cout << "[" << i << "]得分： " << faceInfo[i].bbox.score << std::endl;

				if (x < 0) x = 0;	if (y < 0) y = 0;
				if ((y + h) > image.rows) h = image.rows - y;
				if ((x + w) > image.cols) w = image.cols - x;
				if (w < 0) continue;
				if (h < 0) continue;

				//std::cout << x << " " << y << " " << w << " " << h << std::endl;
				//std::cout << image.rows << " " << image.cols << std::endl;

				dst_face = image(cv::Rect(x, y, w, h));
				char dst_name[100];
				_mkdir("C:\\plate_card_BLSTM\\testData\\face\\");
				sprintf_s(dst_name, "%s%d%s%d%s%d%s", "C:\\plate_card_BLSTM\\testData\\face\\", iNum, "_", frame_count++, "_", i, ".jpg");
				cv::imwrite(dst_name, dst_face);


				FacePts facePts = faceInfo[i].facePts;





				cv::Point2f srcTri[4];
				cv::Point2f dstTri[4];
				cv::Mat rot_mat(2, 4, CV_32FC1);
				cv::Mat warp_mat(2, 4, CV_32FC1);

				for (int j = 0; j < 4; j++) {
					srcTri[j] = cv::Point2f(facePts.x[j] - x, facePts.y[j] - y);
				}

				int padding_x = cvFloor(h * 0.04 * 5);
				int padding_y = cvFloor(h * 0.04 * 2);
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
				cv::namedWindow("dst_face", 0);
				cv::imshow("dst_face", dst_face);

				cv::namedWindow("warp_dstImage", 0);
				cv::imshow("warp_dstImage", warp_dstImage);

				cv::waitKey(1);
#endif

				sprintf_s(dst_name, "%s%d%s%d%s%d%s", "face\\", iNum, "_", frame_count++, "_", i, "_warp_dstImage_.jpg");
				cv::imwrite(dst_name, warp_dstImage);

#ifdef warpPerspective

				int img_height = dst_face.rows;
				int img_width = dst_face.cols;

				vector<Point2f> corners(4);

				for (int j = 0; j < 4; j++) {
					corners[j] = cv::Point2f(facePts.x[j] - x, facePts.y[j] - y);
				}

				vector<Point2f> corners_trans(4);
				corners_trans[0] = cv::Point2f(x0 + padding_x, y0 + padding_y);
				corners_trans[1] = cv::Point2f(x1 + padding_x, y1 + padding_y);
				corners_trans[2] = cv::Point2f(x2 + padding_x, y2 + padding_y);
				corners_trans[3] = cv::Point2f(x3 + padding_x, y3 + padding_y);

				getStartTime();
				Mat transform = getPerspectiveTransform(corners, corners_trans);
				//cout << transform << endl;
				Mat resultImage;
				warpPerspective(dst_face, resultImage, transform, Size(120 + 2 * padding_x, 48 + 2 * padding_y), INTER_LINEAR);

				getEndTime();
				std::cout << " 2 :" << dfTim << std::endl;

				namedWindow("warpPerspective", 0);
				imshow("warpPerspective", resultImage);

				cv::waitKey(1);



				sprintf_s(dst_name, "%s%d%s%d%s%d%s", "face\\", iNum, "_", frame_count++, "_", i, "_resultImage_.jpg");
				cv::imwrite(dst_name, resultImage);

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
				if (w1 != w_rec && h_rec != hstd)
					cv::resize(img, img, cv::Size(w1, hstd), cv::INTER_CUBIC);

				int start = clock();

				vector<int> shape;
				vector<float> pred = pCNN->GetOutputFeatureMap(img, shape);

				int end = clock();
				sumspend += (end - start);

				string strpredict0 = GetPredictString(pred, idxBlank, alphabets);

				printf("[%d]%s\n", i, strpredict0.c_str());
#ifdef showSteps
				cv::namedWindow("img", 0);
				cv::imshow("img", img);
				cv::waitKey(1);
#endif


#ifdef showSteps
				for (int j = 0; j < 4; j++) {
				//	std::cout << facePts.x[j] - x << " " << facePts.y[j] - y << std::endl;
					cv::circle(image, cv::Point(facePts.x[j], facePts.y[j]), 1, cv::Scalar(255, 255, 0), 2);
				}

				cv::rectangle(image, cv::Rect(x, y, w, h), cv::Scalar(255, 0, 0), 2);
				//std::cout << x << " " << y << " " << w << " " << h << std::endl;
#endif
			}

			if (faceInfo.size() == 0) {
				cv::Mat img = image;
				//cv::normalize(img, img, 255, 0, cv::NORM_MINMAX);
				int w_rec = img.cols, h_rec = img.rows;
				if (2 * w_rec <= h_rec)
				{
					cv::transpose(img, img);
					cv::flip(img, img, 1);
					w_rec = img.cols, h_rec = img.rows;
				}


				//int w1 = hstd*w / h;
				int w1 = 280;
				if (w1 != w_rec && h_rec != hstd)
					cv::resize(img, img, cv::Size(w1, hstd),cv::INTER_CUBIC);

				int start = clock();

				vector<int> shape;
				vector<float> pred = pCNN->GetOutputFeatureMap(img, shape);

				int end = clock();
				sumspend += (end - start);

				string strpredict0 = GetPredictString(pred, idxBlank, alphabets);

				printf("[%d]%s\n", iNum, strpredict0.c_str());

			}


			cv::namedWindow("Extracted frame_", 0);
			cv::imshow("Extracted frame_", image);

			int c = cv::waitKey(0);
			//按下ESC或者到达指定的结束帧后退出读取视频  
			if ((char)c == 27 /*|| currentFrame > frameToStop*/)
			{
				stop = true;
			}
			//按下按键后会停留在当前帧，等待下一次按键  
			//if (c >= 0)
			//{
			//	waitKey(0);
			//}
			//	currentFrame++;

		}

		//关闭视频文件  
		capture.release();
	}
	return 0;


}


int main_mtcnn_haar(int argc, char **argv) {



	double threshold[3] = { 0.7, 0.8, 0.8 };
	double factor = 0.709;
	int minSize = 40;
	std::string proto_model_dir = "C:\\plate_card_BLSTM\\vs2013_caffe_BN_multi_label_kenel_w\\model_platecar\\";
	MTCNN *detector = new MTCNN(proto_model_dir);

#ifndef HAAR_CASCADE
	cv::CascadeClassifier char_cascade;
	if (!char_cascade.load(".\\rec_test\\cascade12.xml"))
	{
		cerr << "ERROR: Could not load classifier cascade" << endl;
		return -1;
	}


	string chinese_model_file = ".\\rec_test\\chinese\\model.prototxt";
	string chinese_trained_file = ".\\rec_test\\chinese\\model.caffemodel";
	string chinese_mean_file = "";
	int chinese_label_file = 31;
	Classifier net_chinese(chinese_model_file, chinese_trained_file, chinese_mean_file, chinese_label_file);


	string char_model_file = ".\\rec_test\\char\\model.prototxt";
	string char_trained_file = ".\\rec_test\\char\\model.caffemodel";
	string char_mean_file = "";
	int char_label_file = 37;
	Classifier net_char(char_model_file, char_trained_file, char_mean_file, char_label_file);

#endif // HAAR_CASCADE

	


#ifdef CPU_ONLY
	bool usegpu = false;
#else
	bool usegpu = true;
#endif

	//load model
	string modelfolder = ".\\plateCard_test\\";
	ICNNPredict* pCNN = CreatePredictInstance(modelfolder.c_str(), usegpu);
	int wstd = 0, hstd = 0;
	pCNN->GetInputImageSize(wstd, hstd);

	//get alphabet
	vector<string> alphabets = pCNN->GetLabels();

	int idxBlank = 0;
	vector<string>::const_iterator it = find(alphabets.begin(), alphabets.end(), "blank");
	if (it != alphabets.end())
		idxBlank = (int)(it - alphabets.begin());


	map<wchar_t, int> mapLabel2IDs;
	for (size_t i = 0; i < alphabets.size(); i++)
	{
		wchar_t c = 0;
		if (alphabets[i] == "blank")
			continue;
		wstring wlabel = string2wstring(alphabets[i], true);
		mapLabel2IDs.insert(make_pair(wlabel[0], i));
	}


	int sumspend = 0;
	int nok_lexicon = 0;
	int nok_nolexicon = 0;




	initTrainImage();
	int imgNum = imgNames.size();
	for (int iNum = 0; iNum < imgNum; iNum++) {

		cout << endl << iNum << " " << imgNames[iNum].c_str() << endl;
		cv::VideoCapture capture(imgNames[iNum].c_str());
		//cv::VideoCapture capture(0);
		//capture.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
		//capture.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
		//
		//VideoCapture capture("F:\\MTCNN-master\\vs2013_caffe_BN_multi_label\\water_meter_caffe_old\\\img\\1.avi");
		//检测是否正常打开:成功打开时，isOpened返回ture  
		if (!capture.isOpened())
			cout << "fail to open!" << endl;
		//获取整个帧数  
		//	long totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);
		//	cout << "整个视频共" << totalFrameNumber << "帧" << endl;
		//
		//
		//设置开始帧()  
		//	long frameToStart = 300;
		//	capture.set(CV_CAP_PROP_POS_FRAMES, frameToStart);
		//	cout << "从第" << frameToStart << "帧开始读" << endl;
		//
		//
		//设置结束帧  
		//	int frameToStop = 400000;
		//
		//	if (frameToStop < frameToStart)
		//	{
		//		cout << "结束帧小于开始帧，程序错误，即将退出！" << endl;
		//		return -1;
		//	}
		//	else
		//	{
		//		cout << "结束帧为：第" << frameToStop << "帧" << endl;
		//	}
		//
		//
		//	获取帧率  
		//	double rate = capture.get(CV_CAP_PROP_FPS);
		//	cout << "帧率为:" << rate << endl;



		//定义一个用来控制读取视频循环结束的变量  
		bool stop = false;

		//显示每一帧的窗口  

		//两帧间的间隔时间:  
		//int delay = 1000/rate;  
		//	int delay = 1000 / rate;
		//	if (rate == 0)  delay = 1;
		//
		//利用while循环读取帧  
		//currentFrame是在循环体中控制读取到指定的帧后循环结束的变量  
		//	long currentFrame = frameToStart;
		//
		//    VideoWriter writer;
		//    writer.open("../result/SuicideSquad.mp4",CV_FOURCC('M', 'J', 'P', 'G'), 25, Size(1280,720), true);
		//承载每一帧的图像  

		cv::Mat image;
		int frame_count = 0;
		while (!stop)
		{
			//读取下一帧  
			if (!capture.read(image))
			{
				cout << "读取视频失败" << endl;
				stop = true;
				continue;
			}

			//imshow("Live", image);
			//waitKey(0);
			//for (int i = 0; i < 100; i++)capture.read(image);
			////
			//cv::flip(image, image,-1);

			std::vector<FaceInfo> faceInfo;
			clock_t t1 = clock();
			//		std::cout << "Detect " << image.rows << "X" << image.cols;
			//
			//image = image.t();


			cv::Mat mergeImg;//合并后的图像  
							 //用来存储各通道图片的向量  
			vector<cv::Mat> splitBGR(image.channels());
			//分割通道，存储到splitBGR中  
			split(image, splitBGR);
			//对各个通道分别进行直方图均衡化  
			for (int i = 0; i<image.channels(); i++)
				equalizeHist(splitBGR[i], splitBGR[i]);
			//合并通道  
			merge(splitBGR, mergeImg);

			detector->Detect(mergeImg, faceInfo, minSize, threshold, factor);
#ifdef CPU_ONLY
			std::cout << " Time Using CPU: " << (clock() - t1)*1.0 / 1000 << std::endl;
#else
			std::cout << " Time Using : " << (clock() - t1)*1.0 / 1000 << std::endl;
#endif
			cv::Mat dst_face;

#ifndef HAAR_CASCADE
			vector<result_plate> results;
			vector<result_plate>(results).swap(results);
			results.clear();
#endif // DEBUG

			cv::Mat warp_dstImage;
			for (int i = 0; i < faceInfo.size() ||(i==0 && faceInfo.size() ==0); i++) {
				
				if (faceInfo.size() != 0) {

					float x = faceInfo[i].bbox.x1;
					float y = faceInfo[i].bbox.y1;
					float w = faceInfo[i].bbox.x2 - faceInfo[i].bbox.x1 + 1;
					float h = faceInfo[i].bbox.y2 - faceInfo[i].bbox.y1 + 1;

					std::cout << "[" << i << "]得分： " << faceInfo[i].bbox.score << std::endl;

					if (x < 0) x = 0;	if (y < 0) y = 0;
					if ((y + h) > image.rows) h = image.rows - y;
					if ((x + w) > image.cols) w = image.cols - x;
					if (w < 0) continue;
					if (h < 0) continue;

					//std::cout << x << " " << y << " " << w << " " << h << std::endl;
					//std::cout << image.rows << " " << image.cols << std::endl;

					dst_face = image(cv::Rect(x, y, w, h));
					char dst_name[100];
					_mkdir("C:\\plate_card_BLSTM\\testData\\face\\");
					sprintf_s(dst_name, "%s%d%s%d%s%d%s", "C:\\plate_card_BLSTM\\testData\\face\\", iNum, "_", frame_count++, "_", i, ".jpg");
					cv::imwrite(dst_name, dst_face);


					FacePts facePts = faceInfo[i].facePts;





					cv::Point2f srcTri[4];
					cv::Point2f dstTri[4];
					cv::Mat rot_mat(2, 4, CV_32FC1);
					cv::Mat warp_mat(2, 4, CV_32FC1);

					for (int j = 0; j < 4; j++) {
						srcTri[j] = cv::Point2f(facePts.x[j] - x, facePts.y[j] - y);
					}

					int padding_x = cvFloor(h * 0.04 * 5);
					int padding_y = cvFloor(h * 0.04 * 2);
					int x0 = 0;		int y0 = 0;
					int x1 = 120;	int y1 = 0;
					int x2 = 120;	int y2 = 48;
					int x3 = 0;		int y3 = 48;

					dstTri[0] = cv::Point2f(x0 + padding_x, y0 + padding_y);
					dstTri[1] = cv::Point2f(x1 + padding_x, y1 + padding_y);
					dstTri[2] = cv::Point2f(x2 + padding_x, y2 + padding_y);
					dstTri[3] = cv::Point2f(x3 + padding_x, y3 + padding_y);

					warp_mat = cv::getAffineTransform(srcTri, dstTri);
					/*cv::Mat */warp_dstImage = cv::Mat::zeros(48 + 2 * padding_y, 120 + 2 * padding_x, dst_face.type());
					cv::warpAffine(dst_face, warp_dstImage, warp_mat, warp_dstImage.size());

#ifdef showSteps
					cv::namedWindow("dst_face", 0);
					cv::imshow("dst_face", dst_face);

					cv::namedWindow("warp_dstImage", 0);
					cv::imshow("warp_dstImage", warp_dstImage);

					cv::waitKey(1);
#endif

					sprintf_s(dst_name, "%s%d%s%d%s%d%s", "face\\", iNum, "_", frame_count++, "_", i, "_warp_dstImage_.jpg");
					cv::imwrite(dst_name, warp_dstImage);

#ifdef warpPerspective

					int img_height = dst_face.rows;
					int img_width = dst_face.cols;

					vector<Point2f> corners(4);

					for (int j = 0; j < 4; j++) {
						corners[j] = cv::Point2f(facePts.x[j] - x, facePts.y[j] - y);
					}

					vector<Point2f> corners_trans(4);
					corners_trans[0] = cv::Point2f(x0 + padding_x, y0 + padding_y);
					corners_trans[1] = cv::Point2f(x1 + padding_x, y1 + padding_y);
					corners_trans[2] = cv::Point2f(x2 + padding_x, y2 + padding_y);
					corners_trans[3] = cv::Point2f(x3 + padding_x, y3 + padding_y);

					getStartTime();
					Mat transform = getPerspectiveTransform(corners, corners_trans);
					//cout << transform << endl;
					Mat resultImage;
					warpPerspective(dst_face, resultImage, transform, Size(120 + 2 * padding_x, 48 + 2 * padding_y), INTER_LINEAR);

					getEndTime();
					std::cout << " 2 :" << dfTim << std::endl;

					namedWindow("warpPerspective", 0);
					imshow("warpPerspective", resultImage);

					cv::waitKey(1);



					sprintf_s(dst_name, "%s%d%s%d%s%d%s", "face\\", iNum, "_", frame_count++, "_", i, "_resultImage_.jpg");
					cv::imwrite(dst_name, resultImage);

#endif
				
#ifdef showSteps
					for (int j = 0; j < 4; j++) {
						//	std::cout << facePts.x[j] - x << " " << facePts.y[j] - y << std::endl;
						cv::circle(image, cv::Point(facePts.x[j], facePts.y[j]), 1, cv::Scalar(255, 255, 0), 2);
					}

					cv::rectangle(image, cv::Rect(x, y, w, h), cv::Scalar(255, 0, 0), 2);
					//std::cout << x << " " << y << " " << w << " " << h << std::endl;
#endif

				}
				else if (faceInfo.size() == 0)
				{
					warp_dstImage = image;
				}
#ifndef HAAR_CASCADE
				result_plate result;
				cv::Rect face;
				if (faceInfo.size() != 0) {
					face = cv::Rect(faceInfo[i].bbox.x1, faceInfo[i].bbox.y1, faceInfo[i].bbox.x2 - faceInfo[i].bbox.x1, faceInfo[i].bbox.y2 - faceInfo[i].bbox.y1);
				}
				else if (faceInfo.size() == 0)
				{
					face = cv::Rect(0,0,image.cols,image.rows);
					warp_dstImage = image;
				}
				result.coordinate = face;


				cv::Mat detect_obj = warp_dstImage;
				//cv::resize(detect_obj, detect_obj ,cv::Size(detect_obj.cols * 2, detect_obj.rows *2),cv::INTER_CUBIC);

				vector<CvRect> detectROI;
				cv::Mat show_detect_obj;	detect_obj.copyTo(show_detect_obj);
				cv::Mat shaixuan_obj, choose_detect_obj, normalization_detect_obj;
				shaixuan_obj = show_detect_obj.clone();//10-12
				choose_detect_obj = shaixuan_obj.clone();
				normalization_detect_obj = shaixuan_obj.clone();
				//######################################################################	
				std::vector<cv::Rect> chars_roi;
				char_cascade.detectMultiScale(detect_obj, chars_roi, 1.05, 1, 0 | CV_HAAR_SCALE_IMAGE/*, cv::Size(0, 0), cv::Size(1700, 1700)*/);
				if (showSteps)
					printf("chars_roi size = %d \n", chars_roi.size());

				for (unsigned int j = 0; j < chars_roi.size(); j++)
				{
					const cv::Rect& single_char_roi = chars_roi[j];
					detectROI.push_back(single_char_roi);
				}

				std::sort(detectROI.begin(), detectROI.end(), sort_by_x);

				if (showSteps) {
					for (int i = 0; i < detectROI.size(); i++) {
						cout << detectROI[i].x << " " << detectROI[i].y << " " << detectROI[i].width << " " << detectROI[i].height << endl;

						cv::Point tl(detectROI[i].x, detectROI[i].y);
						cv::Point br = tl + cv::Point(detectROI[i].width, detectROI[i].height);
						cv::Scalar magenta = cv::Scalar((i) * 10, 255 - (i + 1) * 10, (i + 2) * 50);//颜色选取囧

						cv::rectangle(show_detect_obj, tl, br, magenta, 1, 1, 0);
					}

					cv::namedWindow("show_detect_obj", 0);
					cv::imshow("show_detect_obj", show_detect_obj);

				}

				///**************** step 2.1 fix detected roi ******************************//
				int avg_distance_of_chars = 0;
				int ROIWIDTH = 0; //平均宽
				int ROIHEIGHT = 0;//平均高

				notfound = 1;
				if (showSteps)
					printf(" \n start \n");

				vector<CvRect> ROI_choose_paixu = roichoose(detectROI, choose_detect_obj);//选择合适的框

				if (showSteps)
					show_choose_step(shaixuan_obj, ROI_choose_paixu, "ROI_choose_paixu");
				if (showSteps)
					printf("使用排序寻找到的框数量为%d \n", ROI_choose_paixu.size());

				if (notfound == 0 || detectROI.size() < 4) {
					printf(" 检测到的车牌字符个数为： %d ，不足4个故退出 ... \n", detectROI.size());
					continue;
				}

				vector<CvRect> ROI_normalization = roinormalization(ROI_choose_paixu, normalization_detect_obj);//归一化

				if (showSteps)
					show_choose_step(shaixuan_obj, ROI_normalization, "ROI_normalization");

				vector<CvRect> ROI_choose = roicomplete(ROI_normalization, normalization_detect_obj);//补框

				if (showSteps)
					show_choose_step(shaixuan_obj, ROI_choose, "ROI_choose");

				std::sort(ROI_choose.begin(), ROI_choose.end(), sort_by_x);

				//show_choose_step(cv::Mat shaixuan_obj, vector<cv::Rect> ROI_choose, char* windName)

				///**************************  根据识别筛选框 **************************/
				IplImage *detect = &IplImage(normalization_detect_obj);
				if (ROI_choose.size() > 4)
				{
					///??????????????????????			


					if (ROI_choose.size() > 5)
					{

						/*******************************************第一个字符拿出来*******************************/
						if (showSteps)
							printf("车牌: ");
						if (ROI_choose.size() == 8)//看看去除前还是后(方法一：使用车牌中间的点的位置进行识别，在框框特别大的时候不好判断）
						{

							if (ROI_choose[2].x - ROI_choose[1].x - ROI_choose[1].width > 0 && ROI_choose[3].x - ROI_choose[2].x - ROI_choose[2].width > 0)
							{
								if (showSteps)
									printf("special");
							}
							/*	else if (ROI_choose[2].x - ROI_choose[1].x - ROI_choose[1].width > 0)  //警车是特殊情况
							ROI_choose.erase(ROI_choose.begin() + 7);
							else if (ROI_choose[3].x - ROI_choose[2].x - ROI_choose[2].width > 0)
							ROI_choose.erase(ROI_choose.begin());*/
						}

						/// ********************* 看看去除前还是后(方法二：使用识别来进行去除，过于依赖loss不准确） ********************///
						if (ROI_choose.size() == 8)
						{
							int predict0, predict1, predict11, predict7, predict6;
							predict0 = -1;
							predict1 = -1; predict11 = -1;
							predict7 = -1;
							double loss0, loss1, loss11, loss7, loss6;
							loss0 = 0.0;
							loss11 = 0.0;
							loss1 = 0.0;
							loss7 = 0.0;
							CvRect cut_single0 = ROI_choose[0];
							CvRect cut_single1 = ROI_choose[1];
							CvRect cut_single7 = ROI_choose[7];
							IplImage* img_single0_result = cvCreateImage(cvSize(cut_single0.width, cut_single0.height), detect->depth, detect->nChannels);//得到单字符二值图像
							cvSetImageROI(detect, cut_single0);
							cvCopy(detect, img_single0_result);
							cvResetImageROI(detect);

							IplImage* img_single1_result = cvCreateImage(cvSize(cut_single1.width, cut_single1.height), detect->depth, detect->nChannels);//得到单字符二值图像
							cvSetImageROI(detect, cut_single1);
							cvCopy(detect, img_single1_result);
							cvResetImageROI(detect);

							IplImage* img_single7_result = cvCreateImage(cvSize(cut_single7.width, cut_single7.height), detect->depth, detect->nChannels);//得到单字符二值图像
							cvSetImageROI(detect, cut_single7);
							cvCopy(detect, img_single7_result);
							cvResetImageROI(detect);

							IplImage* img_gray0 = cvCreateImage(cvGetSize(img_single0_result), IPL_DEPTH_8U, 1);//创建目标图像  
							cvCvtColor(img_single0_result, img_gray0, CV_BGR2GRAY);
							cv::Mat img_re0 = cv::cvarrToMat(img_gray0);
							/*IplImage* img_re0 = cvCreateImage(cvSize(charRec.getWidth(), charRec.getHeight()), 8, charRec.getChannel());
							cvResize(img_gray0, img_re0);*/

							IplImage* img_gray1 = cvCreateImage(cvGetSize(img_single1_result), IPL_DEPTH_8U, 1);//创建目标图像  
							cvCvtColor(img_single1_result, img_gray1, CV_BGR2GRAY);
							cv::Mat img_re1 = cv::cvarrToMat(img_gray1);
							/*IplImage* img_re1 = cvCreateImage(cvSize(charRec.getWidth(), charRec.getHeight()), 8, charRec.getChannel());
							cvResize(img_gray1, img_re1);*/

							IplImage* img_gray7 = cvCreateImage(cvGetSize(img_single7_result), IPL_DEPTH_8U, 1);//创建目标图像  
							cvCvtColor(img_single7_result, img_gray7, CV_BGR2GRAY);
							cv::Mat img_re7 = cv::cvarrToMat(img_gray7);
							/*IplImage* img_re7 = cvCreateImage(cvSize(charRec.getWidth(), charRec.getHeight()), 8, charRec.getChannel());
							cvResize(img_gray7, img_re7);*/
							rec_char(net_chinese, img_re0, predict0, loss0);
							rec_char(net_chinese, img_re1, predict1, loss1);
							rec_char(net_char, img_re1, predict11, loss11);
							rec_char(net_char, img_re7, predict7, loss7);

							//charRec0.recognise_form_memory_scale0_1(img_re0, &predict0, &loss0);//识别第一位是否是汉字
							//charRec0.recognise_form_memory_scale0_1(img_re1, &predict1, &loss1);//第二位是否是汉字
							//charRec.recognise_form_memory_scale0_1(img_re1, &predict11, &loss11);//第二位是否是字母
							//charRec.recognise_form_memory_scale0_1(img_re7, &predict7, &loss7);//第七位是否是字母

							if (showSteps)
								printf("loss0_cn=%f loss7_char=%f loss1_Chinese=%f loss1_char=%f \n", loss0, loss7, loss1, loss11);
							if (loss11 > loss1) {
								ROI_choose.erase(ROI_choose.begin() + 7);
							}
							else {
								ROI_choose.erase(ROI_choose.begin());
							}


						}
						/// *****************************  //多出两个时也可以处理   ***********************// 
						if (ROI_choose.size() == 9)
						{
							int predict1_cn, predict1_char, predict2_cn, predict2_char, predict3_cn, predict3_char = -1;

							double loss1_cn, loss1_char, loss2_cn, loss2_char, loss3_cn, loss3_char = 0.0;

							CvRect cut_single1 = ROI_choose[0];
							CvRect cut_single2 = ROI_choose[1];
							CvRect cut_single3 = ROI_choose[2];
							IplImage* img_single1_result = cvCreateImage(cvSize(cut_single1.width, cut_single1.height), detect->depth, detect->nChannels);//得到单字符二值图像
							cvSetImageROI(detect, cut_single1);
							cvCopy(detect, img_single1_result);
							cvResetImageROI(detect);

							IplImage* img_single2_result = cvCreateImage(cvSize(cut_single2.width, cut_single2.height), detect->depth, detect->nChannels);//得到单字符二值图像
							cvSetImageROI(detect, cut_single2);
							cvCopy(detect, img_single2_result);
							cvResetImageROI(detect);

							IplImage* img_single3_result = cvCreateImage(cvSize(cut_single3.width, cut_single3.height), detect->depth, detect->nChannels);//得到单字符二值图像
							cvSetImageROI(detect, cut_single3);
							cvCopy(detect, img_single3_result);
							cvResetImageROI(detect);

							IplImage* img_gray1 = cvCreateImage(cvGetSize(img_single1_result), IPL_DEPTH_8U, 1);//创建目标图像  
							cvCvtColor(img_single1_result, img_gray1, CV_BGR2GRAY);
							cv::Mat img_re1 = cv::cvarrToMat(img_gray1);
							/*IplImage* img_re1 = cvCreateImage(cvSize(charRec.getWidth(), charRec.getHeight()), 8, charRec.getChannel());
							cvResize(img_gray1, img_re1);*/

							IplImage* img_gray2 = cvCreateImage(cvGetSize(img_single2_result), IPL_DEPTH_8U, 1);//创建目标图像  
							cvCvtColor(img_single2_result, img_gray2, CV_BGR2GRAY);
							cv::Mat img_re2 = cv::cvarrToMat(img_gray2);
							/*IplImage* img_re2 = cvCreateImage(cvSize(charRec.getWidth(), charRec.getHeight()), 8, charRec.getChannel());
							cvResize(img_gray2, img_re2);*/

							IplImage* img_gray3 = cvCreateImage(cvGetSize(img_single3_result), IPL_DEPTH_8U, 1);//创建目标图像  
							cvCvtColor(img_single3_result, img_gray3, CV_BGR2GRAY);
							cv::Mat img_re3 = cv::cvarrToMat(img_gray3);
							/*IplImage* img_re3 = cvCreateImage(cvSize(charRec.getWidth(), charRec.getHeight()), 8, charRec.getChannel());
							cvResize(img_gray3, img_re3);*/


							rec_char(net_chinese, img_re1, predict1_cn, loss1_cn);
							rec_char(net_chinese, img_re2, predict2_cn, loss2_cn);
							rec_char(net_chinese, img_re3, predict3_cn, loss3_cn);
							rec_char(net_char, img_re1, predict1_char, loss1_char);
							rec_char(net_char, img_re2, predict2_char, loss2_char);
							rec_char(net_char, img_re3, predict3_char, loss3_char);

							//charRec0.recognise_form_memory_scale0_1(img_re1, &predict1_cn, &loss1_cn);//识别第一位是否是汉字
							//charRec0.recognise_form_memory_scale0_1(img_re2, &predict2_cn, &loss2_cn);//第二位是否是汉字
							//charRec0.recognise_form_memory_scale0_1(img_re3, &predict3_cn, &loss3_cn);//第三位是否是汉字
							//charRec.recognise_form_memory_scale0_1(img_re1, &predict1_char, &loss1_char);//第一位是否是字母
							//charRec.recognise_form_memory_scale0_1(img_re2, &predict2_char, &loss2_char);//第二位是否是字母
							//charRec.recognise_form_memory_scale0_1(img_re3, &predict3_char, &loss3_char);//第三位是否是字母

							if (loss3_cn > loss3_char)//第三位为汉字时，前两位删除
							{
								ROI_choose.erase(ROI_choose.begin());
								ROI_choose.erase(ROI_choose.begin());
							}
							else if (loss2_cn > loss2_char) //第二位为汉字时，前后各删一位
							{
								ROI_choose.erase(ROI_choose.begin() + 8);
								ROI_choose.erase(ROI_choose.begin());
							}
							else //第三位第二位都不为汉字时，删除后两位
							{
								ROI_choose.erase(ROI_choose.begin() + 8);
								ROI_choose.erase(ROI_choose.begin() + 7);
							}

							/*if (loss11 > loss1) {
							ROI_choose.erase(ROI_choose.begin() + 7);
							}
							else {
							ROI_choose.erase(ROI_choose.begin());
							}*/
							if (showSteps) {
								printf("loss1为roichoose的第一个框 \n");
								printf("loss1_cn=%f loss1_char=%f \n loss2_cn=%f loss2_char=%f \n loss3_cn=%f loss3_char=%f \n", loss1_cn, loss1_char, loss2_cn, loss2_char, loss3_cn, loss3_char);
							}
						}
						if (showSteps) {
							show_choose_step(shaixuan_obj, ROI_choose, "ROI_choose_final");
							//cvWaitKey(0);
						}


						///*****************************  进入完整识别  *********************** 4-25加入判断是否使用全卷积的程序*******************************/
						if (ROI_choose.size() >= 6)
						{

							int use_fcn = 0;
							int roiwidth = ROI_choose[0].width;
							for (int i = 3; i < ROI_choose.size(); i++)
							{
								if (ROI_choose[i].x - roiwidth - ROI_choose[i - 1].x > 0.1*roiwidth)
									use_fcn++;
							}
							//if (use_fcn >= 0)
							//{
							//	//printf("need use fcn \n");
							//
							//
							//	std::vector<struct result_> predictions = classifier.Classify(detect_obj, 1);
							//	for (int i = 0; i < predictions.size(); i++) {
							//		std::cout << predictions[i].label << " ";
							//		cv::Rect roi = cv::Rect(predictions[i].centor.x - predictions[i].avg_width / 2,
							//			predictions[i].centor.y - predictions[i].avg_height / 2,
							//			predictions[i].avg_width,
							//			predictions[i].avg_height
							//		);//200*80图片下的框
							//		cv::Rect roi_final;
							//		roi_final.x = (roi.x*detect_obj.cols) / 200;
							//		roi_final.y = (roi.y*detect_obj.rows) / 80;
							//		roi_final.width = (roi.width*detect_obj.cols) / 200;
							//		roi_final.height = (roi.height*detect_obj.rows) / 80;
							//
							//		cv::rectangle(detect_obj, roi_final, cv::Scalar(255, 0, 255), 1, 8, 0);
							//	}
							//
							//
							//	cv::namedWindow("src", 0);
							//	cv::imshow("src", detect_obj);
							//	cvWaitKey();
							//	
							//}
							//if (use_fcn > 0) {
							//	
							//	_mkdir("use_fcn");
							//
							//	SYSTEMTIME stTime;
							//	GetLocalTime(&stTime);
							//	char pVideoName[256];
							//	sprintf_s(pVideoName, 256, "\\%d_%d_%d_%d_%d_%d_%d", stTime.wYear, stTime.wMonth, stTime.wDay, stTime.wHour, stTime.wMinute, stTime.wSecond, stTime.wMilliseconds);
							//	char image_name1[500];
							//	sprintf_s(image_name1, 500, "%s%s%s%s", "use_fcn", "\\", pVideoName, ".jpg");//保存的图片名
							//	imwrite(image_name1, img_src_orignal);

							//	continue;
							//}
							/************************不使用fcn***************/
							if (1) {
								for (int i = 0; i < ROI_choose.size(); i++)
								{
									CvRect cut_single = ROI_choose[i];
									IplImage* img_single_result = cvCreateImage(cvSize(cut_single.width, cut_single.height), detect->depth, detect->nChannels);//得到单字符二值图像
									cvSetImageROI(detect, cut_single);
									cvCopy(detect, img_single_result);
									cvResetImageROI(detect);

									if (showSteps)
									{
										char windowsname[200];
										sprintf(windowsname, "%s%d", "detect", i);//分别显示分割图像
										cvNamedWindow(windowsname, 1);
										cvShowImage(windowsname, img_single_result);
									}

									IplImage* img_gray = cvCreateImage(cvGetSize(img_single_result), IPL_DEPTH_8U, 1);//创建目标图像  
									cvCvtColor(img_single_result, img_gray, CV_BGR2GRAY);
									cv::Mat img_re = cv::cvarrToMat(img_gray);
									/*IplImage* img_re = cvCreateImage(cvSize(charRec.getWidth(), charRec.getHeight()), 8, charRec.getChannel());
									cvResize(img_gray, img_re);*/
									int predict = -1;
									double loss = 0.0;

									if (i == 0) {

										rec_char(net_chinese, img_re, predict, loss);
										/*charRec0.recognise_form_memory_scale0_2(img_re, &predict, &loss);*/

										if (predict > 30) {
											predict = -1;
										}

										char* hanzi = outputhanzi(predict);

										Pred pred;
										pred.first = cvRect(ROI_choose[i].x + face.x, ROI_choose[i].y + face.y, ROI_choose[i].width, ROI_choose[i].height);
										pred.second = hanzi;
										result.one_char.push_back(pred);

										if (showSteps)
											printf("中文-loss=%f \n", loss);

									}
									else {
										rec_char(net_char, img_re, predict, loss);
										/*if (predict == -1)
										predict = 0;*/
										/*charRec.recognise_form_memory_scale0_1(img_re, &predict, &loss);*/

										if ((i == 1) && (predict == 8))
											predict = 11;
										char*  szzm = outputplate(predict);
										Pred pred;
										pred.first = cvRect(ROI_choose[i].x + face.x, ROI_choose[i].y + face.y, ROI_choose[i].width, ROI_choose[i].height);
										pred.second = szzm;
										result.one_char.push_back(pred);


										if (showSteps)
											printf("predict=%d loss=%f \n", predict, loss);
									}



									/*if (loss < 0.6)
									predict = -1;*/
								}
							}
						}
					}


					/*if (showSteps)
					{
						cvDestroyAllWindows();
					}*/

				}

				results.push_back(result);
				//#########################################################


#endif // HAAR_CASCADE

		


			}

			

			if (results.size() == NULL) {
				continue;
			}


			IplImage *showcarplate;
			if (faceInfo.size() == 0)
				showcarplate = &IplImage(image);
			else
				showcarplate = &IplImage(warp_dstImage);
			for (int num = 0; num < (int)results.size(); num++)
			{
				CvRect face = results[num].coordinate;
				cvRectangle(showcarplate, cvPoint(face.x, face.y), cvPoint(face.x + face.width, face.y + face.height), CV_RGB(255, 255, 0), 3, 4, 0);
				vector<Pred> carplate = results[num].one_char;
				///***************************  交互显示结果  *****************************///
				if (carplate.size() >= 7 && showdemo)
				{
				//	CvxText text("simhei.ttf");
					char *strID = new char(100); strID[0] = '\0';
					char *strID_CN = new char(100); strID_CN[0] = '\0';

					strcat(strID_CN, carplate[0].second);

					for (int i = 1; i < carplate.size(); i++)
					{
						strcat(strID, carplate[i].second);
					}


					CvScalar color;
					color.val[0] = face.height *0.3;
					color.val[1] = 0.5;
					color.val[2] = 0.1;
					color.val[3] = 0;
				//	text.restoreFont();

				//	text.setFont(NULL, &color, NULL, NULL);
					if (faceInfo.size() == 0) {
						cvRectangle(showcarplate, cvPoint(face.x - face.height*0.1, face.y - 5), cvPoint(face.x + face.height * 1.2, face.y - 10 - face.height*0.3), CV_RGB(0, 0, 0), -1, 4, 0);
						for (int i = 0; i < carplate.size(); i++)
						{
							cvRectangle(showcarplate, cvPoint(carplate[i].first.x, carplate[i].first.y), cvPoint(carplate[i].first.x + carplate[i].first.width, carplate[i].first.y + carplate[i].first.height), CV_RGB(0, 255, 0), 1, 4, 0);
						}
						//	text.putText(showcarplate, strID, cvPoint(face.x + face.height *0.3, face.y - 10), CV_RGB(255, 255, 255));
						//	text.putText(showcarplate, strID_CN, cvPoint(face.x, face.y - 10), CV_RGB(255, 255, 255));
					}
					else if(faceInfo.size() != 0) {
						
						
						/*cvRectangle(showcarplate, cvPoint(face.x - face.height*0.1, face.y - 5), cvPoint(face.x + face.height * 1.2, face.y - 10 - face.height*0.3), CV_RGB(0, 0, 0), -1, 4, 0);*/
						for (int i = 0; i < carplate.size(); i++)
						{
							carplate[i].first.x -= face.x; carplate[i].first.y -= face.y;
							cvRectangle(showcarplate, cvPoint(carplate[i].first.x, carplate[i].first.y), cvPoint(carplate[i].first.x + carplate[i].first.width, carplate[i].first.y + carplate[i].first.height), CV_RGB(0, 255, 0), 1, 4, 0);
						}
					}
					for (int i = 0; i < carplate.size(); i++)
					{
						printf("%s", carplate[i].second);
					}
					cvNamedWindow("showcarplate", 0);
					cvShowImage("showcarplate", showcarplate);

					carplate.clear();
					printf(" \n");
				}




				if (0)
				{
					int c = cvWaitKey();
					if ((c == 's') || (c == 'S')) {
						_mkdir("error");

						SYSTEMTIME stTime;
						GetLocalTime(&stTime);
						char pVideoName[256];
						sprintf_s(pVideoName, 256, "\\%d_%d_%d_%d_%d_%d_%d", stTime.wYear, stTime.wMonth, stTime.wDay, stTime.wHour, stTime.wMinute, stTime.wSecond, stTime.wMilliseconds);
						char image_name1[500];
						sprintf_s(image_name1, 500, "%s%s%s%s", "error", "\\", pVideoName, ".bmp");//保存的图片名				
						imwrite(image_name1, image);


						_mkdir("orig_result");
						char outName[200];
						sprintf_s(outName, 200, "%s%s%s", "orig_result", "\\", file_name[iNum].c_str());
						cout << iNum << outName << endl;
						MoveFile(img_path[iNum].c_str(), outName);    // 将D盘2.txt移动到E盘 


					}
					else if ((c == 'b') || (c == 'B')) {
						if (iNum > 1) {
							iNum--; iNum--;
						}
					}
				}



			}






			cv::namedWindow("Extracted frame_", 0);
			cv::imshow("Extracted frame_", image);

			int c = cv::waitKey(0);
			//按下ESC或者到达指定的结束帧后退出读取视频  
			if ((char)c == 27 /*|| currentFrame > frameToStop*/)
			{
				stop = true;
			}
			//按下按键后会停留在当前帧，等待下一次按键  
			//if (c >= 0)
			//{
			//	waitKey(0);
			//}
			//	currentFrame++;

		}

		//关闭视频文件  
		capture.release();
	}
	return 0;


}