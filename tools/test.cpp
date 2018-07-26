#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>

#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <windows.h>
#include <sstream> 

# pragma warning (disable:4819)

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;
using namespace cv;

#include "classification.hpp"

#ifdef _DEBUG
#pragma comment(lib,"libClassificationd.lib")
#else
#pragma comment(lib,"libClassification.lib")
#endif

#include "folder.h"
string GetExeFolder()
{
	char pathname[256];
	GetModuleFileNameA(NULL, pathname, sizeof(pathname));

	char *ppathname = pathname + strlen(pathname);
	while (*ppathname != '\\') ppathname--;
	ppathname[1] = 0;
	return pathname;
}

void FindAllImages(const char *folder, vector<string>& vImgPaths, bool bSubFolder)
{
	char szPathName[MAX_PATH];
	strcpy(szPathName, folder);
	if (szPathName[strlen(szPathName) - 1] != '\\')
		strcat(szPathName, "\\");

	char szFileName[256];
	strcpy(szFileName, szPathName);
	strcat(szFileName, "*.*");

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
					strcpy(szFileName, szPathName);
					strcat(szFileName, wfd.cFileName);
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
						|| _stricmp(ext3, "png") == 0)
					{
						//printf("%s\n", wfd.cFileName);

						char filename[256];
						sprintf(filename, "%s%s", szPathName, wfd.cFileName);
						vImgPaths.push_back(filename);
					}
				}
			}
		} while (FindNextFile(hFind, &wfd) != 0);
	}
}

int GetImgNumInset(const string& strImgFolder, const string& strSetList)
{
	vector<string> vImgs, vImgNames;
	FindAllImages(strImgFolder.c_str(), vImgs, false);

	for (size_t i = 0; i < vImgs.size(); i++)
	{
		string& imgfile = vImgs[i];
		size_t pos = imgfile.rfind('\\');
		string imgname = imgfile.substr(pos + 1, imgfile.size() - pos - 1);

		vImgNames.push_back(imgname);
	}

	set<string> setImageNames;
	std::ifstream file(strSetList);
	string strline;
	while (std::getline(file, strline))
		setImageNames.insert(strline);

	int num = 0;
	for (size_t i = 0; i < vImgNames.size(); i++)
	{
		if (setImageNames.find(vImgNames[i]) != setImageNames.end())
			num++;
	}

	return num;
}
Classifier* InitCaffeClassifier(bool vgg = true)
{
	string strImageNetFolder = FOLDER_IMAGENET;
	string strCaffeFolder = FOLDER_CAFFE;

	string strModelFolder;
	string model_file, trained_file;
	if (!vgg)
	{
		strModelFolder = strCaffeFolder+"models\\bvlc_googlenet\\";
		model_file = strModelFolder + "deploy.prototxt";
		trained_file = strModelFolder + "bvlc_googlenet.caffemodel";
	}
	else
	{
		strModelFolder = strCaffeFolder + "models\\VGG\\";
		model_file = strModelFolder + "VGG_ILSVRC_19_layers_deploy.prototxt";
		trained_file = strModelFolder + "VGG_ILSVRC_19_layers.caffemodel";
	}


	string mean_file = strImageNetFolder + "imagenet_mean.binaryproto";
	//string label_file = strImageNetFolder + "synsets.txt";
	//Classifier* clf = new Classifier(model_file, trained_file, mean_file, "");
	Classifier* clf = new Classifier();
	clf->Init(trained_file, model_file, mean_file, "", true);

	return clf;
}
vector<string> ParseConfigFile(const string& strFile, const char* titles[10], int n)
{
	vector<string> vValues;
	std::ifstream config(strFile);
	if (!config.is_open())
	{
		printf("can not found config file=%s\n", strFile.c_str());
		return vValues;
	}

	vector<pair<string, string>> vConfigs;
	string strline;
	while (getline(config, strline))
	{
		size_t pos = strline.find('=');
		if (pos == string::npos)
			continue;
		string strTitle = strline.substr(0, pos);
		string strValue = strline.substr(pos + 1, strline.size() - pos - 1);
		vConfigs.push_back(make_pair(strTitle, strValue));
	}
	for (int i = 0; i < n; i++)
	{
		string strValue;
		for (size_t j = 0; j < vConfigs.size(); j++)
		{
			if (vConfigs[j].first == titles[i])
			{
				strValue = vConfigs[j].second;
				break;
			}
		}
		vValues.push_back(strValue);
	}

	return vValues;
}

static std::string GetImgNameFromFullPath(const std::string& strPath)
{
	size_t pos = strPath.rfind('\\');
	if (pos == std::string::npos)
		return "";
	return strPath.substr(pos + 1, strPath.size() - pos - 1);
}

void TestCaffeModel()
{
	string strConfigFile = "test_caffe_model.txt";
	vector<string> vTitles;

	const char* titles[] = { "caffe-net", "caffe-model","image-mean","imgfolder","top"};
	int n = sizeof(titles) / sizeof(titles[0]);

	vector<string> vConfigs = ParseConfigFile(strConfigFile, titles, n);
	if (vConfigs.size() != n)
	{
		printf("cannot find %s or wrong format\n", strConfigFile.c_str());
		return ;
	}

	string strCaffeNet = vConfigs[0];
	string strCaffeModel = vConfigs[1];
	string strImgMean = vConfigs[2];
	string strImgFolder = vConfigs[3];
	int topn = atoi(vConfigs[4].c_str());
	if (topn < 0)
		topn = 1;

	vector<string> vImgs;
	FindAllImages(strImgFolder.c_str(), vImgs, false);
	printf("found %d images in folder=%s\n", vImgs.size(), strImgFolder.c_str());

	if (vImgs.size() == 0)
		return;


	Classifier* clf = new Classifier();
	clf->Init(strCaffeModel, strCaffeNet, strImgMean, "", true);

	clock_t start = clock();
	clock_t end = 0;

	string strExeFolder = GetExeFolder();
	string strPredResult = strExeFolder + "predict_result.txt";

	std::ofstream fileout(strPredResult);
	std::cout << "start to predict images..." << std::endl;
	for (size_t i = 0; i < vImgs.size(); i++)
	{
		cv::Mat img = cv::imread(vImgs[i], -1);
		if (img.empty())
			continue;
		std::vector<Prediction> predictions = clf->Classify(img, topn);

		string strImgName = GetImgNameFromFullPath(vImgs[i]);
		fileout << strImgName << " ";
		for (size_t j = 0; j < predictions.size(); j++)
			fileout << predictions[j].first <<":"<< predictions[j].second << "";
		fileout << std::endl;

		printf("[%d/%d]%s, %s:%f\n", i + 1, vImgs.size(), strImgName.c_str(),
			predictions[0].first.c_str(), predictions[0].second);

		//printf("\r%d/%d", i + 1, vImgs.size());
	}
	end = clock();

	delete clf;
}

void BatchPredictFoodImage(const char* imgfolder, bool vgg)
{

	string strImgFolder = imgfolder;

	string food_sub_folder = strImgFolder + "cnnfood\\";
	string nonfood_sub_folder = strImgFolder + "cnnnonfood\\";

	CreateDirectoryA(food_sub_folder.c_str(), NULL);
	CreateDirectoryA(nonfood_sub_folder.c_str(), NULL);

	vector<string> vImgs;
	FindAllImages(strImgFolder.c_str(), vImgs, false);
	printf("found %d images in folder=%\n", vImgs.size(), imgfolder);

	if (vImgs.size() == 0)
		return;

	Classifier* clf = InitCaffeClassifier(vgg);
	if (!clf)
		return;

	string cls_map_file_food = FILE_IMAGENET_FOODMAP;
	set<string> setImageNetIDsFood;
	std::ifstream idfile(cls_map_file_food);
	string strline;
	while (std::getline(idfile, strline))
		setImageNetIDsFood.insert(strline);

	clock_t start = clock();
	clock_t end = 0;
	int img_num_food = 0, food = 0;

	string strFoodListFile = food_sub_folder + "list.txt", strNonFoodListFile = nonfood_sub_folder + "list.txt";
	std::ofstream foodlist(strFoodListFile), nonfoodlist(strNonFoodListFile);
	std::cout << "start to predict food images..." << std::endl;
	for (size_t i = 0; i < vImgs.size(); i++)
	{
		cv::Mat img = cv::imread(vImgs[i], -1);
		if (img.empty())
			continue;
		std::vector<Prediction> predictions = clf->Classify(img, 6);

		img_num_food++;
		/* Print the top N predictions. */

		float confid_food = 0, confid_nonfood = 0;
		for (size_t j = 0; j < predictions.size(); ++j)
		{
			Prediction p = predictions[j];

			string& clsid = p.first;
			float confid = p.second;

			if (setImageNetIDsFood.find(clsid) != setImageNetIDsFood.end())
				confid_food += confid;
			else
				confid_nonfood += confid;
		}

		string& imgfile = vImgs[i];
		size_t pos = imgfile.rfind('\\');
		string imgname = imgfile.substr(pos + 1, imgfile.size() - pos - 1);

		if (confid_food > confid_nonfood)
		{
			food++;
			foodlist << imgname << std::endl;
			foodlist.flush();
		}
		else
		{
			nonfoodlist << imgname << std::endl;
			nonfoodlist.flush();
		}

		printf("\r%d/%d", i + 1, vImgs.size());
	}
	end = clock();
	printf("found %d food images in %d images, fps=%f\n ", food, img_num_food, vImgs.size()*1000.0f / (end - start));
	delete clf;

}

void SaveImgCNNFeature(const string& imgFolder, const char* featFile, bool vgg = true, const char* listfile = NULL)
{
	string strImgFolder = imgFolder;

	vector<string> vImgs;
	if (listfile)
	{
		std::ifstream file(listfile);
		string strline;
		while (std::getline(file, strline))
			vImgs.push_back(strImgFolder + strline);
	}
	else
	{
		FindAllImages(strImgFolder.c_str(), vImgs, false);
	}
	printf("found %d images in folder=%s, filelist=%s\n", vImgs.size(), imgFolder, listfile ? listfile : "NULL");

	if (vImgs.size() == 0)
		return;

	FILE* ff = fopen(featFile, "wb");
	if (!ff)
	{
		printf("failed to open feature file=%s\n", featFile);
		return;
	}

	Classifier* clf = InitCaffeClassifier(vgg);
	if (!clf)
		return;

	int num_feature = 0;
	int dim = clf->GetFeatureDim();
	fwrite(&num_feature, sizeof(num_feature), 1, ff);
	fwrite(&dim, sizeof(dim), 1, ff);

	string strNameListFile = strImgFolder + "list.txt";
	std::ofstream liststream;
	if (!listfile)
		liststream.open(strNameListFile.c_str());

	std::cout << "start to extact images features..." << std::endl;
	for (size_t i = 0; i < vImgs.size(); i++)
	{
		cv::Mat img = cv::imread(vImgs[i], -1);
		if (img.empty())
			continue;
		std::vector<float> fv = clf->ExtractFeature(img);

		if (fv.size() == 0)
			continue;

		string& imgfile = vImgs[i];
		size_t pos = imgfile.rfind('\\');
		string imgname = imgfile.substr(pos + 1, imgfile.size() - pos - 1);
		
		num_feature++;
		fwrite(&fv[0], sizeof(float), fv.size(), ff);
		fflush(ff);
		liststream << imgname << std::endl;
		liststream.flush();

		printf("\r%d/%d", i + 1, vImgs.size());
	}

	fseek(ff, 0, SEEK_SET);
	fwrite(&num_feature, sizeof(num_feature), 1, ff);

	delete clf;
}

void BatchPredictAndExtractFeatuer(const string& strImgFolder, const string& strFeatureFile, 
	const string & strPredictFile, bool vgg=true, const char* imglistfile=NULL)
{
	std::ofstream predfile(strPredictFile);
	if (!predfile.is_open())
	{
		printf("failed to open %s", strPredictFile.c_str());
		return;
	}
	std::ofstream featfile;
	if (strFeatureFile.size() > 0)
		featfile.open(strFeatureFile);

	vector<string> vImgs;
	if (imglistfile)
	{
		std::ifstream file(imglistfile);
		string strline;
		while (std::getline(file, strline))
			vImgs.push_back(strImgFolder + strline);
	}
	else
	{
		FindAllImages(strImgFolder.c_str(), vImgs, false);
	}
	printf("found %d images in folder=%s, filelist=%s\n",
		vImgs.size(), strImgFolder.c_str(), imglistfile ? imglistfile : "NULL");

	if (vImgs.size() == 0)
		return;

	

	Classifier* clf = InitCaffeClassifier(vgg);
	if (!clf)
		return;

	std::cout << "start to predict food images..." << std::endl;
	for (size_t i = 0; i < vImgs.size(); i++)
	{
		cv::Mat img = cv::imread(vImgs[i], -1);
		if (img.empty())
			continue;

		string& imgfile = vImgs[i];
		size_t pos = imgfile.rfind('\\');
		string imgname = imgfile.substr(pos + 1, imgfile.size() - pos - 1);

		std::vector<Prediction> predictions = clf->Classify(img, 10);

		//save predict results
		predfile << imgname << " ";
		for (size_t j = 0; j < predictions.size();j++)
			predfile << predictions[j].first << ":" << predictions[j].second<<" ";
		predfile << std::endl;
		//predfile.flush();

		if (strFeatureFile.size()>0)
		{
			//save feature
			vector<float> fv = clf->GetLayerFeature();
			featfile << imgname << " ";
			for (size_t j = 0; j < fv.size(); j++)
			{
				if (fv[j]==0)
					continue;
				featfile << j + 1 << ":" << fv[j] << " ";
			}
			featfile << std::endl;
			//featfile.flush();
		}
		

		printf("\r%d/%d", i + 1, vImgs.size());
	}

	delete clf;
}

void SplitString(const string& str, const string& strSep, std::vector<std::string>& vStrings)
{
	size_t pos = 0, prepos = 0;
	while (pos <= str.size())
	{
		pos = str.find(strSep, prepos);
		if (pos == string::npos)
		{
			if (prepos <= (int)str.size() - 1)
				vStrings.push_back(str.substr(prepos, str.size() - prepos));
			break;
		}

		if (pos - prepos > 0)
			vStrings.push_back(str.substr(prepos, pos - prepos));
		prepos = pos + strSep.size();
	}
}

void PredictLineStr2Predictions(const string& line, string& strImgName, vector<Prediction>& vPreds)
{
	vector<string> vSubStrs;
	SplitString(line, " ", vSubStrs);
	if (vSubStrs.size() < 2)
		return;

	strImgName = vSubStrs[0];
	for (size_t i = 1; i < vSubStrs.size();i++)
	{
		vector<string> ClsIDConf;
		SplitString(vSubStrs[i], ":", ClsIDConf);
		if (ClsIDConf.size()!=2)
			continue;
		vPreds.push_back(make_pair(ClsIDConf[0], (float)atof(ClsIDConf[1].c_str())));
	}
}

void ParseFeatureLine(const string& line, string& strImgName, vector<float>& feature)
{
	vector<string> vSubStrs;
	SplitString(line, ":", vSubStrs);
	if (vSubStrs.size() < 2)
		return;

	strImgName = vSubStrs[0];
	feature.resize(vSubStrs.size() - 1);
	size_t n = vSubStrs.size() - 1;
	for (size_t i = 0; i < n;i++)
	{
		feature[i] = atof(vSubStrs[i + 1].c_str());
	}
}

void LoadCNNFeatureFile(const string& featurefile, std::map<string, std::vector<float> >& features)
{
	string line;
	std::ifstream preditfile(featurefile);

	int n = 0;
	vector<float> feature;
	feature.reserve(2048);
	string strImgName;
	while (getline(preditfile, line))
	{
		feature.clear();
		ParseFeatureLine(line, strImgName, feature);

		features.insert(make_pair(strImgName, feature));

		n++;
		printf("\r%d", n);
	}
}

void GetFoodImgFromPredictFile(const string& strImgFolder,
	const string& strPreditFile,
	const string& strDstFolder)
{
	string line;
	std::ifstream preditfile(strPreditFile);

	string cls_map_file_food = FILE_IMAGENET_FOODMAP;
	set<string> setImageNetIDsFood;
	std::ifstream idfile(cls_map_file_food);
	string strline;
	while (std::getline(idfile, strline))
		setImageNetIDsFood.insert(strline);

	int n = 0;
	while (getline(preditfile, line))
	{
		vector<Prediction> vPreds;
		string strImgName;

		PredictLineStr2Predictions(line, strImgName, vPreds);

		float confid_food = 0, confid_nonfood = 0;
		size_t np = vPreds.size();
		if (np > 6) np = 6;
		for (size_t j = 0; j < np; ++j)
		{
			Prediction p = vPreds[j];

			string& clsid = p.first;
			float confid = p.second;

			if (setImageNetIDsFood.find(clsid) != setImageNetIDsFood.end())
				confid_food += confid;
			else
				confid_nonfood += confid;
		}

		if (confid_food > confid_nonfood)
		{
			string strSrcFile = strImgFolder + strImgName;
			string strDstFile = strDstFolder + strImgName;
			CopyFileA(strSrcFile.c_str(), strDstFile.c_str(), FALSE);
		}
		n++;
		printf("\r%d", n);
	}
}

void GetImgWidthHeight(int argc, char* argv[])
{
	string strImgFolder;
	if (argc < 2)
	{
		strImgFolder = GetExeFolder();
	}
	else strImgFolder = argv[1];

#ifdef _DEBUG
	strImgFolder = "c:\\work\\sample\\trash\\manual_0716\\";
#endif

	printf("folder=%s\n", strImgFolder.c_str());

	if (strImgFolder.size() == 0)
		return;
	if (strImgFolder[strImgFolder.size() - 1] != '\\')
		strImgFolder += "\\";

	vector<string> vImgs;
	FindAllImages(strImgFolder.c_str(), vImgs, false);

	string strFile = strImgFolder + "imsize.txt";
	std::ofstream file(strFile.c_str());
	for (size_t i = 0; i < vImgs.size(); i++)
	{
		string& imgfile = vImgs[i];
		cv::Mat img = cv::imread(imgfile, -1);
		if (img.empty())
			continue;
		size_t pos = imgfile.rfind('\\');
		if (pos == string::npos)
			continue;
		//printf("%s\n", imgfile.c_str());
		string imgname = imgfile.substr(pos + 1, imgfile.size() - pos - 1);
		file << imgname << " " << img.cols << " " << img.rows << std::endl;

		printf("\r%d/%d", i + 1, vImgs.size());
	}
}

//-------------------------------------------------------------------------------------------------
bool SolveLinearSystem2(int n, double *A, double *b, double *x)
{
	// P*A = L*U, b' = P*b
	int i, j, k;
	for (i = 0; i < n - 1; i++)
	{
		// find the maximum |A(i:n, i)|
		j = i;
		for (k = i + 1; k < n; k++) if (fabs(A[j*n + i]) < fabs(A[k*n + i])) j = k;
		if (fabs(A[j*n + i])<1e-6) return false;

		// swap the two rows A(i,:) and A(j,:) and two elements b(i) and b(j)
		if (j>i)
		{
			for (k = 0; k < n; k++)
			{
				double temp = A[i*n + k];
				A[i*n + k] = A[j*n + k];
				A[j*n + k] = temp;
			}
			double temp = b[i];
			b[i] = b[j];
			b[j] = temp;
		}

		// update A(i+1:n,i) /= A(i,i)
		for (k = i + 1; k < n; k++) A[k*n + i] /= A[i*n + i];

		// update A(i+1:n,i+1:n) -= A(i+1:n,i)*A(i,i+1:n)
		for (j = i + 1; j < n; j++)
		{
			for (k = i + 1; k < n; k++) A[j*n + k] -= A[j*n + i] * A[i*n + k];
		}
	}

	// Solve L*y = b'
	for (i = 0; i < n; i++)
	{
		x[i] = b[i];
		for (j = 0; j < i; j++) x[i] -= x[j] * A[i*n + j];
	}

	// Solve U*x = y
	for (i = n - 1; i >= 0; i--)
	{
		for (j = i + 1; j < n; j++) x[i] -= x[j] * A[i*n + j];
		x[i] /= A[i*n + i];
	}

	return true;
}


//-------------------------------------------------------------------------------------------------
// qua0 - source quadrangle
// qua1 - target quadrangle
bool PerspectiveTransform(double *qua0, double *qua1, double *matrix)
{
	// inverse mapping from target to source
	const int n = 8;
	double A[n*n] =
	{
		qua1[0], qua1[1], 1, 0, 0, 0, -qua0[0] * qua1[0], -qua0[0] * qua1[1],
		0, 0, 0, qua1[0], qua1[1], 1, -qua0[1] * qua1[0], -qua0[1] * qua1[1],
		qua1[2], qua1[3], 1, 0, 0, 0, -qua0[2] * qua1[2], -qua0[2] * qua1[3],
		0, 0, 0, qua1[2], qua1[3], 1, -qua0[3] * qua1[2], -qua0[3] * qua1[3],
		qua1[4], qua1[5], 1, 0, 0, 0, -qua0[4] * qua1[4], -qua0[4] * qua1[5],
		0, 0, 0, qua1[4], qua1[5], 1, -qua0[5] * qua1[4], -qua0[5] * qua1[5],
		qua1[6], qua1[7], 1, 0, 0, 0, -qua0[6] * qua1[6], -qua0[6] * qua1[7],
		0, 0, 0, qua1[6], qua1[7], 1, -qua0[7] * qua1[6], -qua0[7] * qua1[7]
	};
	double b[n] = { qua0[0], qua0[1], qua0[2], qua0[3], qua0[4], qua0[5], qua0[6], qua0[7] };
	double x[n];

	// Solve A*x = b
	//TestSolveLinearSystem(n, A, b);
	if (!SolveLinearSystem2(n, A, b, x))
	{
		printf("error: matrix is singular\n");
		return false;
	}
	double a11 = x[0], a12 = x[1], a13 = x[2];
	double a21 = x[3], a22 = x[4], a23 = x[5];
	double a31 = x[6], a32 = x[7];

	matrix[0] = a11;
	matrix[1] = a12;
	matrix[2] = a13;
	matrix[3] = a21;
	matrix[4] = a22;
	matrix[5] = a23;
	matrix[6] = a31;
	matrix[7] = a32;
	matrix[8] = 1.0000;

	return true;
}

float GetBestMatchRANSAC(const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2,
	const vector<DMatch>& matches, float sim_t=0.9f, int try_times=10)
{
	if (matches.size()<4 || keypoints1.size() < 4 || keypoints2.size() < 4)
		return 0;
	float maxsim = 0;

	for (int i = 0; i < try_times; i++)
	{
		vector<bool> selected(matches.size(), 0);
		vector<int> vIdxs;
		//随机选择4对匹配点
		while (vIdxs.size()<4)
		{
			int idx = rand() % matches.size();
			if (!selected[idx])
			{
				vIdxs.push_back(idx);
				selected[idx] = true;
			}
		}
		double qua0[8], qua1[8];
		for (int j = 0; j < 4; j++)
		{
			int ptidx1 = matches[vIdxs[j]].queryIdx;
			int ptidx2 = matches[vIdxs[j]].trainIdx;
			const KeyPoint& p1 = keypoints1[ptidx1];
			const KeyPoint& p2 = keypoints2[ptidx2];
			int idx = j * 2;
			qua0[idx] = p1.pt.x;
			qua0[idx+1] = p1.pt.y;
			qua1[idx] = p2.pt.x;
			qua1[idx + 1] = p2.pt.y;
		}
		//计算透视变换矩阵
		double H[9];
		if(!PerspectiveTransform(qua0, qua1, H))
			continue;

		//计算符合当前变换的点对的比例，作为相似度
		const double	h11 = H[0], h12 = H[1], h13 = H[2],
			h21 = H[3], h22 = H[4], h23 = H[5],
			h31 = H[6], h32 = H[7], h33 = H[8];

		int ptok = 4;
		for (size_t j = 0; j < matches.size();j++)
		{
			if (selected[j])
				continue;
			int ptidx1 = matches[j].queryIdx;
			int ptidx2 = matches[j].trainIdx;
			const Point2f& p1 = keypoints1[ptidx1].pt;
			const Point2f& p2 = keypoints2[ptidx2].pt;
			double x1 = h11*p2.x + h12*p2.y + h13;
			double y1 = h21*p2.x + h22*p2.y + h23;
			double z1 = h31*p2.x + h32*p2.y + h33;
			if (fabs(z1)<1e-6)
				continue;
			x1 /= z1; 
			y1 /= z1;
			if ((x1 - p1.x)*(x1 - p1.x) + (y1 - p1.y)*(y1 - p1.y) < 9)
				ptok++;
		}

		//如果相似度大于阈值直接返回，否则记录最大的相似度
		float sim = ptok / (float)matches.size();
		if (sim>sim_t)
			return sim;
		if (sim > maxsim)
			maxsim = sim;
	}

	return maxsim;
}

void ResizeToStd(Mat& img, int stdsize = 640)
{
	int w = img.cols, h = img.rows;
	if (max(w, h) <= stdsize)
		return;

	int w1 = w, h1 = h;
	if (w >= h)
	{
		w1 = 640;
		h1 = w1*h / w;
	}
	else
	{
		h1 = 640;
		w1 = w*h1 / h;
	}

	resize(img, img,Size(w1,h1));
}

void FilterUnSymmetryPairs(const vector<DMatch>& match1To2,int keyptNum1,
	const vector<DMatch>& match2To1, int keyptNum2,
	vector<DMatch>& symMatches)
{
	vector<int> vIdx1To2(keyptNum1,-1), vIdx2To1(keyptNum2,-1);

	for (size_t i = 0; i < match1To2.size();i++)
	{
		int idx1 = match1To2[i].queryIdx;
		int idx2 = match1To2[i].trainIdx;
		vIdx1To2[idx1] = idx2;
	}
	for (size_t i = 0; i < match2To1.size(); i++)
	{
		int idx2 = match2To1[i].queryIdx;
		int idx1 = match2To1[i].trainIdx;
		vIdx2To1[idx2] = idx1;
	}

	symMatches.reserve(match1To2.size());
	for (size_t i = 0; i < match1To2.size(); i++)
	{
		int idx1 = i;
		int idx2 = vIdx1To2[i];
		if (vIdx2To1[idx2] == idx1)
			symMatches.push_back(match1To2[i]);
	}
}

float GetImgSimByKeypointsMatch(const string& strimg1, const string& strimg2,bool showmatch=false)
{
	Mat image1 = imread(strimg1, -1);
	Mat image2 = imread(strimg2, -1);

	if (image1.empty() || image2.empty())
		return 0;

	ResizeToStd(image1);
	ResizeToStd(image2);

	// 检测surf特征点
	vector<KeyPoint> keypoints1, keypoints2;
	SurfFeatureDetector detector(600);
	detector.detect(image1, keypoints1);
	detector.detect(image2, keypoints2);
	//printf("feature points=%d in image1\n", keypoints1.size());
	//printf("feature points=%d in image2\n", keypoints2.size());

	// 描述surf特征点
	SurfDescriptorExtractor surfDesc;
	Mat descriptros1, descriptros2;
	surfDesc.compute(image1, keypoints1, descriptros1);
	surfDesc.compute(image2, keypoints2, descriptros2);


	//printf(" feature points matching\n");
	// 计算匹配点数
	BruteForceMatcher<L2<float>>matcher;
	vector<DMatch> matches1To2, matches2To1, matches;
	matcher.match(descriptros1, descriptros2, matches1To2);
	matcher.match(descriptros2, descriptros1, matches2To1);

	FilterUnSymmetryPairs(matches1To2, keypoints1.size(), matches2To1, keypoints2.size(), matches);
	//printf("match pts=%d\n", matches.size());
	if (matches.size()<10)
		return 0;

	size_t max_match = 50;// matches.size()>100 ? 100 : matches.size();
	if (max_match>matches.size())
		max_match = matches.size();
	if (max_match < matches.size())
	{
		std::nth_element(matches.begin(), matches.begin() + max_match - 1, matches.end());
		matches.erase(matches.begin() + max_match, matches.end());
	}
	float sim = GetBestMatchRANSAC(keypoints1, keypoints2, matches,1); 

	if (showmatch)
	{
		Mat imgMatch;
		drawMatches(image1, keypoints1, image2, keypoints2, matches, imgMatch);
		imshow("match", imgMatch);
	}

	return sim;
}

void GenImgPairSim(const string& strPairList)
{
	string strSimFile = strPairList + "_sim.txt";
	int cnt = 0;

	std::ifstream file(strPairList);
	if (!file.is_open())
	{
		printf("failed to open %s\n", strPairList.c_str());
		return;
	}
	std::ofstream out(strSimFile);

	string line;
	while (getline(file, line))
	{
		vector<string> vSubStrs;
		SplitString(line, " ", vSubStrs);
		if (vSubStrs.size()!=3)
			continue;
		float sim = GetImgSimByKeypointsMatch(vSubStrs[0], vSubStrs[1]);
		out << line << " " << sim <<std::endl;
		cnt++;
		printf("\r%d", cnt);
	}

}

void TestImgMatch()
{
	string strImgFolder = "C:\\work\\sample\\UGC重复图\\labelled\\";
	string strimg1 = strImgFolder + "1159927148\\group1\\8ae53975eb4a93aadf55c8b6137025fd99452.jpg";
	string strimg2 = strImgFolder + "1159927148\\group1\\10d82e3c05751277729a09c1493e8cee80791.jpg";
	
	float sim = GetImgSimByKeypointsMatch(strimg1, strimg2,true);
	printf("sim=%f\n", sim);

	printf("draw matching image\n");


	cvWaitKey(0);
}

void CvtLowercase(string& str)
{
	for (size_t i = 0; i < str.size(); i++)
	{
		if (str[i] >= 'A' && str[i] <= 'Z')
			str[i] = str[i] - 'A' + 'a';
	}
}

void PredictByCNNModel(const string& strNetworkFile, const string& strModelFile, const string& strMeanFile,
	const string& strLabelFile, const string& strImgFolder,
	const string& strLabelFilterFile,
	size_t start_img = 0, int crop_resize_stdsize = 0,float conf_t=0)
{
	printf("net=%s\n", strNetworkFile.c_str());
	printf("model=%s\n", strModelFile.c_str());
	printf("mean=%s\n", strMeanFile.c_str());
	printf("label=%s\n", strLabelFile.c_str());
	printf("imgfolder=%s\n", strImgFolder.c_str());
	printf("filterfile=%s\n", strLabelFilterFile.size()>0?strLabelFilterFile.c_str():"");
	printf("start_img=%d\n", start_img);
	printf("crop_resize_stdsize=%d\n", crop_resize_stdsize);


	Classifier* clf = new Classifier();
	clf->Init(strModelFile, strNetworkFile, strMeanFile, "", true);

	std::vector<std::string> labels;
	string line;
	std::ifstream labelfile(strLabelFile);
	while (getline(labelfile, line))
		labels.push_back(line);

	printf("%d labels found\n", labels.size());
	if (labels.size() == 0)
	{
		delete clf;
		return;
	}

	set<string> setLabelToOut;
	if (strLabelFilterFile.size() > 0)
	{
		std::ifstream filterfile(strLabelFilterFile);
		while (getline(filterfile, line))
		{
			setLabelToOut.insert(line);
		}
		if (setLabelToOut.size() == 0)
		{
			printf("no output label found\n");
		}
	}


	vector<string> vLabelFolder(labels.size());
	for (size_t i = 0; i < labels.size();i++)
	{
		if (setLabelToOut.find(labels[i]) != setLabelToOut.end())
		{
			vLabelFolder[i] = strImgFolder + labels[i] + "\\";
			CreateDirectory(vLabelFolder[i].c_str(), NULL);
		}
	}

	clock_t start = clock();

	vector<string> vImgs;
	FindAllImages(strImgFolder.c_str(), vImgs, false);
	printf("%d images found\n", vImgs.size());

	if (start_img >= vImgs.size())
		start_img = 0;

	string strPredictFile = strImgFolder + "cnn_predict.txt";
	/*
	map<string, string> mapImgName2Predicts;
	//read predict file
	if (start_img > 0)
	{
		std::ifstream infile(strPredictFile);
		if (infile.is_open())
		{
			while (getline(infile, line))
			{
				size_t imao = line.find(':');
				if (imao == string::npos)
					continue;
				size_t iblank = line.rfind(' ',imao-1);
				if(iblank==string::npos)
					continue;
				string name = line.substr(0, iblank);
				string strpred = line.substr(iblank, line.size() - iblank);
				mapImgName2Predicts.insert(make_pair(name, strpred));
			}
		}

	}
	*/
	std::ofstream predfile(strPredictFile);

	int hist[3] = { 0 };
	for (size_t i = start_img; i < vImgs.size(); i++)
	{
		Mat im = imread(vImgs[i], CV_LOAD_IMAGE_COLOR);
		if (im.empty())
			continue;
		string strName = GetImgNameFromFullPath(vImgs[i]);

		vector<Prediction> vpred;
		if (crop_resize_stdsize <= 0)
			vpred = clf->Classify(im, labels.size());
		else
			vpred = clf->CropClassify(im, crop_resize_stdsize, 1, labels.size());

		int labelidx = atoi(vpred[0].first.c_str());
		if (vLabelFolder[labelidx].size()>0 && vpred[0].second>conf_t)
		{
			string strDstFile = vLabelFolder[labelidx] + strName;
			CopyFile(vImgs[i].c_str(), strDstFile.c_str(), TRUE);
		}
		if (labelidx < 3 && vpred[0].second>conf_t)
			hist[labelidx]++;

		//save predict results
		predfile << strName << " ";
		for (size_t j = 0; j < vpred.size(); j++)
			predfile << vpred[j].first << ":" << vpred[j].second << " ";
		predfile << std::endl;

		printf("\r%d[%d,%d,%d]/%d", i + 1,
			hist[0],hist[1],hist[2],
			vImgs.size());
	}
	clock_t end = clock();

	printf("\n%d ms per image\n", (end - start) / vImgs.size());

	delete clf;
}
void PredictByCNNModel_MultiLabel(const string& strNetworkFile, const string& strModelFile, const string& strMeanFile,
	const string& strLabelFile, const string& strImgFolder,
	bool lowercase=true,int maxlen=-1)
{
	printf("net=%s\n", strNetworkFile.c_str());
	printf("model=%s\n", strModelFile.c_str());
	printf("mean=%s\n", strMeanFile.c_str());
	printf("label=%s\n", strLabelFile.c_str());
	printf("imgfolder=%s\n", strImgFolder.c_str());
	printf("lowercase=%d\n", lowercase);

#ifdef CPU_ONLY
	bool usegpu = false;
#else
	bool usegpu = true;
#endif


#ifdef _DEBUG
	usegpu = false;
#endif

	Classifier* clf = new Classifier();
	clf->Init(strModelFile, strNetworkFile, strMeanFile, strLabelFile, usegpu);

	std::vector<std::string> labels;
	string line;
	std::ifstream labelfile(strLabelFile);
	while (getline(labelfile, line))
		labels.push_back(line);

	printf("%d labels found\n", labels.size());
	if (labels.size() == 0)
	{
		delete clf;
		return;
	}


	clock_t start = clock();

	vector<string> vImgs;
	FindAllImages(strImgFolder.c_str(), vImgs, false);
	printf("%d images found\n", vImgs.size());

	int hist[3] = { 0 };
	int ok = 0;
	for (size_t i = 0; i < vImgs.size(); i++)
	{
		Mat im = imread(vImgs[i], CV_LOAD_IMAGE_COLOR);
		string strName = GetImgNameFromFullPath(vImgs[i]);

		string result;
		//vector<float> fv = clf->ExtractFeature(im);
		vector< vector<float> > vFeatures = clf->GetLastBlockFeature(im);

		vector<int> vCharIdxs;
		if (vFeatures.size() == labels.size())
		{
			int cols = vFeatures[0].size();
			int rows = labels.size();
			for (size_t j = 0; j < cols; j++)
			{
				size_t idxj = 0;
				float maxv = 0;
				for (size_t k = 0; k < rows; k++)
				{
					if (vFeatures[k][j]>maxv)
					{
						maxv = vFeatures[k][j];
						idxj = k;
					}
				}
				vCharIdxs.push_back(idxj);
			}
		}
		else if (vFeatures[0].size() == labels.size())
		{
			for (size_t j = 0; j < vFeatures.size();j++)
			{
				size_t idxv = 0;
				float maxv = 0;
				for (size_t k = 0; k < vFeatures[j].size();k++)
				{
					if (vFeatures[j][k]>maxv)
					{
						maxv = vFeatures[j][k];
						idxv = k;
					}
				}
				vCharIdxs.push_back(idxv);
			}
		}
		
		for (size_t j = 0; j < vCharIdxs.size(); j++)
		{
 			if(labels[vCharIdxs[j]]=="?")//？表示空，无字条
 				continue;
			result += labels[vCharIdxs[j]].c_str();
		}
		string title = strName.substr(0, strName.rfind('.'));
		if (maxlen > 0 && title.size() > maxlen)
		{
			string actual_labels;
			for (size_t j = 0; j < maxlen; j++)
			{
				actual_labels.push_back(title[j]);
			}
			title = actual_labels;
		}
#ifdef _DEBUG
		printf("%s: %s\n", title.c_str(), result.c_str());
#else
		if (lowercase)
			CvtLowercase(title);
		if (result == title)
			ok++;
		printf("\r%d/%d [ prec=%f ]", i + 1, vImgs.size(), (float)ok / (i + 1));

#endif

	}
	clock_t end = clock();

	printf("\n%d ms per image\n", (end - start) / vImgs.size());

	delete clf;
}


void PredictByCNNModelExclude(const string& strNetworkFile, const string& strModelFile, const string& strMeanFile,
	const string& strLabelFile, const string& strImgFolder,
	const string& strLabelFilterFile,
	size_t start_img = 0)
{
	printf("net=%s\n", strNetworkFile.c_str());
	printf("model=%s\n", strModelFile.c_str());
	printf("mean=%s\n", strMeanFile.c_str());
	printf("label=%s\n", strLabelFile.c_str());
	printf("imgfolder=%s\n", strImgFolder.c_str());
	printf("filterfile=%s\n", strLabelFilterFile.size() > 0 ? strLabelFilterFile.c_str() : "");
	printf("start_img=%d\n", start_img);

	if (strLabelFilterFile.size() == 0)
	{
		printf("no exclude label found\n");
		return;
	}

	std::vector<std::string> labels;
	string line;
	std::ifstream labelfile(strLabelFile);
	while (getline(labelfile, line))
		labels.push_back(line);

	printf("%d labels found\n", labels.size());
	if (labels.size() == 0)
	{
		return;
	}

	std::ifstream filterfile(strLabelFilterFile);
	string excludelabel;
	getline(filterfile, excludelabel);
	std::vector<std::string>::iterator it = find(labels.begin(), labels.end(), excludelabel);
	if (it == labels.end())
		return;

	size_t idx = it - labels.begin();

	//Classifier* clf = new Classifier(strNetworkFile, strModelFile, strMeanFile);
	Classifier* clf = new Classifier();
	clf->Init(strModelFile, strNetworkFile, strMeanFile, strLabelFile, true);
	
	string strOtherFolder = strImgFolder+"Other\\";
	CreateDirectory(strOtherFolder.c_str(), NULL);

	clock_t start = clock();

	vector<string> vImgs;
	FindAllImages(strImgFolder.c_str(), vImgs, false);
	printf("%d images found\n", vImgs.size());

	if (start_img >= vImgs.size())
		start_img = 0;
	for (size_t i = start_img; i < vImgs.size(); i++)
	{
		Mat im = imread(vImgs[i], CV_LOAD_IMAGE_COLOR);
		if (im.empty())
			continue;
		string strName = GetImgNameFromFullPath(vImgs[i]);

		vector<Prediction> vpred = clf->Classify(im, 3);

		int labelidx = atoi(vpred[0].first.c_str());
		if (idx != labelidx)
		{
			string strDstFile = strOtherFolder + strName;
			CopyFile(vImgs[i].c_str(), strDstFile.c_str(), FALSE);
		}

		printf("\r%d", i + 1);
	}
	clock_t end = clock();

	printf("\n%d ms per image\n", (end - start) / vImgs.size());

	delete clf;
}
void TestPredictByCNNModel_MultiLabel(int argc, char** argv)
{
#ifdef _DEBUG
// 	argc = 6;
// 	argv[1] = "L:\\验证码识别\\ResNet-18_multi-task\\Release\\ResNet-18-deploy.prototxt";
// 	argv[2] = "L:\\验证码识别\\ResNet-18_multi-task\\Release\\model.caffemodel";
// 	argv[3] = "L:\\验证码识别\\ResNet-18_multi-task\\Release\\mean.binaryproto";
// 	argv[4] = "L:\\验证码识别\\ResNet-18_multi-task\\Release\\labels.txt";
// 	argv[5] = "L:\\验证码识别\\images\\";
#endif

	if (argc < 6)
	{
		printf("predict NETWORK_FILE MODEL_FILE MEAN_FILE LABEL_FILE IMG_FOLDER [lowercase] [maxlen]\n");
		return;
	}

	bool lowercase = true;
	if (argc >= 7)
		lowercase = atoi(argv[6]) ? true : false;
	int maxlen = -1;
	if (argc >= 8)
		maxlen = atoi(argv[7]);

	PredictByCNNModel_MultiLabel(argv[1], argv[2], argv[3], argv[4], argv[5], lowercase, maxlen);
}

void TestPredictByCNNModel(int argc, char** argv)
{

	if (argc < 6)
	{
		printf("predict NETWORK_FILE MODEL_FILE MEAN_FILE LABEL_FILE IMG_FOLDER [LABEL_FILTER_FILE] [START] [crop_resize_stdsize] [conf_t]\n");
		return;
	}
	string strFilterFile;
	if (argc>=7)
	{
		strFilterFile = argv[6];
	}
	int start = 0;
	if (argc>=8)
	{
		start = atoi(argv[7]);
	}
	if (start < 0)
		start = 0;

	int crop_resize_stdsize = 0;
	if (argc >= 9)
		crop_resize_stdsize = atoi(argv[8]);
	float conf_t = 0;
	if (argc>10)
		conf_t = atof(argv[9]);
	if (conf_t<0 || conf_t>1)
		conf_t = 0;

	PredictByCNNModel(argv[1], argv[2], argv[3], argv[4], argv[5],strFilterFile, (size_t)start,crop_resize_stdsize,conf_t);
}
void TestPredictByCNNModelExclude(int argc, char** argv)
{
	if (argc < 6)
	{
		printf("predict NETWORK_FILE MODEL_FILE MEAN_FILE LABEL_FILE IMG_FOLDER [LABEL_FILTER_FILE] [START]\n");
		return;
	}
	string strFilterFile;
	if (argc >= 7)
	{
		strFilterFile = argv[6];
	}
	int start = 0;
	if (argc >= 8)
	{
		start = atoi(argv[7]);
	}
	if (start < 0)
		start = 0;

	PredictByCNNModelExclude(argv[1], argv[2], argv[3], argv[4], argv[5], strFilterFile, (size_t)start);
}
void TestPredictFeature()
{
	string strUGCFolder = FOLDER_UGC;
	string strConfigFile = "cnn_feat_pred.txt";
	vector<string> vTitles;

	const char* titles[] = { "imgfolder", "savefeature", "vgg" };
	int n = sizeof(titles) / sizeof(titles[0]);

	vector<string> vConfigs = ParseConfigFile(strConfigFile, titles, n);
	if (vConfigs.size() != n)
	{
		printf("cannot find cnn_feat_pred.txt or wrong format\n");
		return;
	}

	string strImgFolder = vConfigs[0];
	string strFeatureFile;
	bool save_feature = atoi(vConfigs[1].c_str()) ? true : false;
	if (save_feature)
		strFeatureFile = strImgFolder + "cnn_feature.txt";
	string strPredictFile = strImgFolder + "cnn_predict.txt";
	printf("imgfolder=%s, featuer file=%s, predict file=%s\n", strImgFolder.c_str(),
		save_feature ? strFeatureFile.c_str() : "", strPredictFile.c_str());

	bool vggmodel = atoi(vConfigs[2].c_str()) ? true : false;
	BatchPredictAndExtractFeatuer(strImgFolder, strFeatureFile, strPredictFile, vggmodel);
}

void FindSubFolder(const string& strFolder, vector<string>& vSubFolder, bool fullpath)
{
	string strFind = strFolder + "*";
	WIN32_FIND_DATAA wfd;
	HANDLE hFind = FindFirstFileA(strFind.c_str(), &wfd);
	if (hFind == INVALID_HANDLE_VALUE)
		return;

	do
	{
		if (strcmp(wfd.cFileName, ".") == 0 || strcmp(wfd.cFileName, "..") == 0)
			continue;

		if (wfd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
		{
			if (fullpath)
			{
				string strSubFolder = strFolder + wfd.cFileName + "\\";
				vSubFolder.push_back(strSubFolder);
			}
			else
				vSubFolder.push_back(wfd.cFileName);
		}

	} while (FindNextFileA(hFind, &wfd) != 0);
}

vector< pair<string,int> > ScanUGCData(const string& strImageFolder, const string& strLabelFile)
{
	vector< pair<string, int> > path2labelid;

	map<string, int> mapLabelName2LabelID;
	std::ifstream file(strLabelFile);
	if (!file.is_open())
	{
		printf("can not open %s\n", strLabelFile.c_str());
		return path2labelid;
	}
	string line;
	int l = 0;
	while (getline(file, line))
		mapLabelName2LabelID.insert(make_pair(line, l++));

	printf("label num found=%d in %s\n", mapLabelName2LabelID.size(), strLabelFile.c_str());
	if (mapLabelName2LabelID.size() == 0)
	{
		return path2labelid;
	}

	map<string, int> LabelName2ImgNum;


	vector<string> class1_folders;
	FindSubFolder(strImageFolder, class1_folders,false);

	for (size_t i = 0; i < class1_folders.size(); i++)
	{
		//printf("%s\n", class1_folders[i].c_str());

		string srcfolderc1 = strImageFolder + class1_folders[i] + "\\";

		vector<string> class2_folders;
		FindSubFolder(srcfolderc1, class2_folders,false);
		for (size_t j = 0; j < class2_folders.size(); j++)
		{
			if (class2_folders[j].rfind("flip") != string::npos
				|| class2_folders[j].rfind("bright") != string::npos
				|| class2_folders[j].rfind("color") != string::npos
				|| class2_folders[j].rfind("contrast") != string::npos
				)
			{
				continue;
			}

			//printf("  %s\n", class2_folders[j].c_str());


			string label;
			size_t pos = class2_folders[j].find('_');
			if (pos == string::npos)
				label = class2_folders[j];
			else
				label = class2_folders[j].substr(0, pos);

			map<string, int>::iterator it = mapLabelName2LabelID.find(label);
			if (it == mapLabelName2LabelID.end())
				continue;


			int labelid = it->second;

			string srcfolderc2 = srcfolderc1 + class2_folders[j] + "\\";
			vector<string> subclass2folder;
			FindSubFolder(srcfolderc2, subclass2folder,false);

			path2labelid.push_back(make_pair(srcfolderc2, labelid));
			for (size_t k = 0; k < subclass2folder.size(); k++)
			{
				if(subclass2folder[k]=="Other")
					continue;

				
				string strfolder = srcfolderc2 + subclass2folder[k] + "\\";
				path2labelid.push_back(make_pair(strfolder, labelid));
			}
		}
	}
	
	return path2labelid;
}

void FilterUGCTrainData(const string& strNetworkFile, const string& strModelFile, const string& strMeanFile,
	const string& strLabelFile, const string& strUGCTrainImgFolder)
{
	printf("net=%s\n", strNetworkFile.c_str());
	printf("model=%s\n", strModelFile.c_str());
	printf("mean=%s\n", strMeanFile.c_str());
	printf("label=%s\n", strLabelFile.c_str());
	printf("UGCimgfolder=%s\n", strUGCTrainImgFolder.c_str());


	vector<pair<string, int>> folder2labelid = ScanUGCData(strUGCTrainImgFolder, strLabelFile);


	std::vector<std::string> labels;
	string line;
	std::ifstream labelfile(strLabelFile);
	while (getline(labelfile, line))
		labels.push_back(line);

	printf("%d labels found\n", labels.size());
	if (labels.size() == 0)
	{
		return;
	}

	//Classifier* clf = new Classifier(strNetworkFile, strModelFile, strMeanFile);
	Classifier* clf = new Classifier();
	clf->Init(strModelFile, strNetworkFile, strMeanFile, strLabelFile, true);

	for (size_t i = 0; i < folder2labelid.size();i++)
	{
		int idx = folder2labelid[i].second;
		printf("%s\n", folder2labelid[i].first.c_str());

		string strOtherFolder = folder2labelid[i].first + "Other\\";
		CreateDirectory(strOtherFolder.c_str(), NULL);

		clock_t start = clock();

		vector<string> vImgs;
		FindAllImages(folder2labelid[i].first.c_str(), vImgs, false);
		printf("%d images found\n", vImgs.size());

		for (size_t i = 0; i < vImgs.size(); i++)
		{
			Mat im = imread(vImgs[i], CV_LOAD_IMAGE_COLOR);
			if (im.empty())
				continue;
			string strName = GetImgNameFromFullPath(vImgs[i]);

			vector<Prediction> vpred = clf->Classify(im, 3);

			int labelidx = atoi(vpred[0].first.c_str());
			if (idx != labelidx)
			{
				string strDstFile = strOtherFolder + strName;
				CopyFile(vImgs[i].c_str(), strDstFile.c_str(), FALSE);
			}

			printf("\r%d", i + 1);
		}
	}
	
	delete clf;
}

void TestFilterUGCTrainData(int argc, char* argv[])
{
#ifdef _DEBUG
	argc = 6;
	argv[1] = "F:\\UGC\\ugc_class\\ugc_googlenet\\deploy.prototxt";
	argv[2] = "F:\\UGC\\ugc_class\\ugc_googlenet\\ugc_googlenet_stage2_iter_12000.caffemodel";
	argv[3] = "F:\\UGC\\ugc_class\\ugc_googlenet\\mean.binaryproto";
	argv[4] = "F:\\UGC\\ugc_class\\train\\ugc_label.txt";
	argv[5] = "F:\\UGC\\ugc_class\\train\\";
	
#else
	if (argc < 6)
	{
		printf("predict NETWORK_FILE MODEL_FILE MEAN_FILE LABEL_FILE IMG_FOLDER\n");
		return;
	}
#endif

	FilterUGCTrainData(argv[1], argv[2], argv[3], argv[4], argv[5]);
}


void NormFeature(vector<float>& v)
{
	float sum = 0;
	for (size_t i = 0; i < v.size(); i++)
		sum += v[i] * v[i];

	if (sum < 1e-6)
		return;
	sum = 1.0f / sqrt(sum);
	for (size_t i = 0; i < v.size(); i++)
		v[i] *= sum;
}

void ExtractCNNFeature(const string& strNetworkFile, const string& strModelFile, const string& strMeanFile,
	const string& strImgFolder, const string& strFeatureFile,
	size_t start_img = 0, bool norm = true, int crop_resize_stdsize=0)
{
	printf("net=%s\n", strNetworkFile.c_str());
	printf("model=%s\n", strModelFile.c_str());
	printf("mean=%s\n", strMeanFile.c_str());
	printf("imgfolder=%s\n", strImgFolder.c_str());
	printf("start_img=%d\n", start_img);
	printf("feature file=%s\n", strFeatureFile.c_str());
	printf("normalize feature=%d\n", norm);
	printf("crop_resize_stdsize = %d\n", crop_resize_stdsize);


// 	Classifier* clf = new Classifier(strNetworkFile, strModelFile, strMeanFile);
// 	if (!clf)
// 	{
// 		printf("failed create cnn classifier\n");
// 		return;
// 	}

	Classifier* clf = new Classifier();
	clf->Init(strModelFile, strNetworkFile, strMeanFile, "", true);

	clock_t start = clock();

	

	vector<string> vImgs;
	FindAllImages(strImgFolder.c_str(), vImgs, false);
	printf("%d images found\n", vImgs.size());

	std::ofstream featfile(strFeatureFile);
	if (start_img >= vImgs.size())
		start_img = 0;
	for (size_t i = start_img; i < vImgs.size(); i++)
	{
		Mat im = imread(vImgs[i], CV_LOAD_IMAGE_COLOR);
		if (im.empty())
			continue;
		string strName = GetImgNameFromFullPath(vImgs[i]);

		if (crop_resize_stdsize>0)
			clf->CropClassify(im, crop_resize_stdsize, 1, 1);
		else
			clf->Classify(im, 1);

		//save feature
		vector<float> fv = clf->GetLayerFeature();
		if (norm)
			NormFeature(fv);

		featfile << strName << ":";
		for (size_t j = 0; j < fv.size(); j++)
		{
#if 0 //libsvm format
			if (fv[j] == 0)
				continue;
			featfile << j + 1 << ":" << fv[j] << " ";
#else
			ostringstream   outv;
			outv.precision(8);
			outv << fv[j]; 
			featfile << outv.str() << ":";
#endif
		}
		featfile << std::endl;

		printf("\r%d", i + 1);
	}
	clock_t end = clock();

	printf("\n%d ms per image\n", (end - start) / vImgs.size());

	delete clf;
}

void TestExtractCNNFeature(int argc, char** argv)
{
#ifdef _DEBUG
	//string folder = "L:\\porn_class\\ResNet101\\";
	argc = 8;
	argv[1] = "L:\\porn_class\\ResNet101\\deploy.prototxt";
	argv[2] = "L:\\porn_class\\ResNet101\\pornnet-step1_iter_172000.caffemodel";
	argv[3] = "L:\\porn_class\\mean.binaryproto";
	argv[4] = "I:\\UGC\\test\\";
	argv[5] = "I:\\UGC\\test\\cnn_feature.txt";
	argv[6] = "0";
	argv[7] = "224";
#endif

	if (argc < 6)
	{
		printf("extract_feature NETWORK_FILE MODEL_FILE MEAN_FILE IMG_FOLDER FEATURE_FILE [START] [crop_resize_stdsize]\n");
		return;
	}
	int start = 0;
	if (argc >= 7)
	{
		start = atoi(argv[6]);
	}
	if (start < 0)
		start = 0;
	int crop_resize_stdsize = 0;
	if (argc >= 8)
		crop_resize_stdsize = atoi(argv[7]);

	ExtractCNNFeature(argv[1], argv[2], argv[3], argv[4], argv[5], start,true,crop_resize_stdsize);
}

float GetVectorDis(const vector<float>& v1, const vector<float>& v2)
{
	float sum = 0;
	for (size_t i = 0; i < v1.size();i++)
		sum += (v1[i] - v2[i])*(v1[i] - v2[i]);
	return sqrt(sum);
}

float GetVectorSim(const vector<float>& v1, const vector<float>& v2)
{
	float dot = 0;
	float len1 = 0;
	float len2 = 0;
	for (size_t i = 0; i < v1.size(); i++)
	{
		dot += v1[i] * v2[i];
		len1 += v1[i] * v1[i];
		len2 += v2[i] * v2[i];
	}
	return dot/(sqrt(len1)*sqrt(len2));
}

bool less_dis(const pair<string, float>& p1, const pair<string, float>& p2)
{
	return p1.second < p2.second;
}

bool more_sim(const pair<string, float>& p1, const pair<string, float>& p2)
{
	return p1.second > p2.second;
}

void FindTopSimilarImages(int argc, char** argv)
{
#ifdef _DEBUG
	//string folder = "L:\\porn_class\\ResNet101\\";
	argc = 8;
	argv[1] = "L:\\porn_class\\ResNet101\\deploy.prototxt";
	argv[2] = "L:\\porn_class\\ResNet101\\pornnet-step1_iter_172000.caffemodel";
	argv[3] = "L:\\porn_class\\mean.binaryproto";
	argv[4] = "L:\\porn_class\\test\\sexy\\";
	argv[5] = "I:\\UGC\\test\\";
	argv[6] = "10";
	argv[7] = "224";
#endif

	if (argc < 7)
	{
		printf("exe NETWORK_FILE MODEL_FILE MEAN_FILE train_image_folder test_image_folder topn [crop_resize_stdsize]\n");
		return;
	}
	string network_file = argv[1];
	string model_file = argv[2];
	string mean_file = argv[3];
	string train_folder = argv[4];
	string test_folder = argv[5];
	int topn = atoi(argv[6]);
	if (topn <= 0)
		topn = 100;
	int crop_resize_stdsize = 0;
	if (argc >= 8)
		crop_resize_stdsize = atoi(argv[7]);

	vector<string> vImgs;
	FindAllImages(test_folder.c_str(), vImgs, false);
	printf("%d test images found\n", vImgs.size());
	if (vImgs.size() == 0)
		return;

	string feature_file = train_folder + "cnn_feature.txt";
	printf("load train features from file=%s\n", feature_file.c_str());
	map<string, vector<float> > train_features;
	LoadCNNFeatureFile(feature_file, train_features);

	printf("found %d train features\n", train_features.size());
	if (train_features.size() == 0)
		return;

	Classifier* clf = new Classifier();
	clf->Init(model_file, network_file, mean_file, "", true);
 
	string dstfolder = test_folder + "topn\\";
	CreateDirectory(dstfolder.c_str(), NULL);
	for (size_t i = 0; i < vImgs.size(); i++)
	{
		Mat im = imread(vImgs[i], CV_LOAD_IMAGE_COLOR);
		if (im.empty())
			continue;
		string strName = GetImgNameFromFullPath(vImgs[i]);

		if (crop_resize_stdsize>0)
			clf->CropClassify(im, crop_resize_stdsize, 1, 1);
		else
			clf->Classify(im, 1);

		//save feature
		vector<float> fv = clf->GetLayerFeature();
		NormFeature(fv);

		vector< pair<string,float> > sims(train_features.size());

		map<string, vector<float> >::const_iterator it = train_features.begin();
		printf("get topn of %s\n", strName.c_str());

		int n = 0;
		for (;it!=train_features.end();it++)
		{
			sims[n].first = it->first;
			//distances[n].second = GetVectorDis(fv, it->second);
			sims[n].second = GetVectorSim(fv, it->second);
			n++;
			printf("\r%d/%d", n, train_features.size());
		}
		printf("\nsort\n");
		sort(sims.begin(), sims.end(), more_sim);
		printf("copy top %d images\n", topn);
		int tops = topn > sims.size() ? sims.size() : topn;

		string testName = GetImgNameFromFullPath(vImgs[i]);
		string dstfolder1 = dstfolder + testName + "\\";
		CreateDirectory(dstfolder1.c_str(), NULL);
		for (int j = 0; j < tops;j++)
		{
			string srcfile = train_folder + sims[j].first;
			string dstfile = dstfolder1 + sims[j].first;
			CopyFile(srcfile.c_str(), dstfile.c_str(), FALSE);
		}
	}

	delete clf;
}

void TestModelPruning(int argc, char** argv)
{
	if (argc < 7)
	{
		printf("exe network_prototxt trained_model mean_file labelsfile weight_t new_model_file\n");
		return;
	}

	Classifier* clf = new Classifier();
	clf->Init(argv[2], argv[1], argv[3], argv[4], false);

	float weight_t = atof(argv[5]);

	string strNewModelFile = "";
	float pruned_ratio  = clf->Pruning(weight_t, argv[6]);
	
	printf("pruned ratio=%f\n", pruned_ratio);
}


int main1(int argc, char** argv) 
{

	::google::InitGoogleLogging(argv[0]);

	//FindTopSimilarImages(argc, argv);
	//TestExtractCNNFeature(argc, argv);

	//TestFilterUGCTrainData(argc, argv);

	//TestPredictByCNNModel(argc, argv);
	TestPredictByCNNModel_MultiLabel(argc, argv);
	//TestModelPruning(argc, argv);

	//TestPredictFeature();

	

	//system("pause");

	return 0;
}
