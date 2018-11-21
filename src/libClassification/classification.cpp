#ifdef WIN32
#include <io.h>
#endif

#include "libClassification/classification.hpp"

//#include "cblas.h"

#include "glog/logging.h"

/*
For Windows
value	mode
00		Existence only
02		Write-only
04		Read-only
06		Read and write

For linux
F_OK	Existence only
R_OK	Read-only
W_OK	Write-only
X_OK	Executable
*/
bool CheckFileExist(const char* szFile)
{
#ifdef WIN32
	return _access(szFile, 0) != -1;
#else
	return access(szFile, F_OK) != -1;
#endif

}


extern "C" EXPORT ICNNPredict* CreatePredictInstance(const char* model_folder, bool use_gpu)
{
	Classifier* p = new Classifier();
	if (!p->Init(model_folder, use_gpu))
	{
		delete p;
		p = NULL;
	}
	return p;
}

Classifier::Classifier(){  }

bool Classifier::Init(const string& model_path, bool gpu_mode) {


	const string trained_file = model_path + "/model.weights";
	const string model_file = model_path + "/deploy.cfg";
	string mean_file = model_path + "/mean.binaryproto";
	const string mean_value_file = model_path + "/mean_values.txt";
	const string label_file = model_path + "/label.txt";

	if (!CheckFileExist(mean_file.c_str()))
		mean_file = mean_value_file;

	if (!gpu_mode)
		Caffe::set_mode(Caffe::CPU);
	else
		Caffe::set_mode(Caffe::GPU);

	/* Load the network. */
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);
	//net_->set_debug_info(true);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

	/* Load the binaryproto mean file. */
	SetMean(mean_file);

	/* Load labels. */
	if (label_file.size() > 0)
	{
		std::ifstream labels(label_file.c_str());
		CHECK(labels) << "Unable to open labels file " << label_file;
		string line;
		while (std::getline(labels, line))
			labels_.push_back(string(line));
	}
	else
	{
		Blob<float>* output_layer = net_->output_blobs()[0];
		char szlabel[100];
		printf("output ch=%d\n", output_layer->channels());
		for (int i = 0; i < output_layer->channels(); i++)
		{
			sprintf(szlabel, "%d", i);
			labels_.push_back(szlabel);
		}

	}


// 	Blob<float>* output_layer = net_->output_blobs()[0];
// 	CHECK_EQ(labels_.size(), output_layer->channels())
// 		<< "Number of labels is different from the output layer dimension.";

	return true;
}
bool Classifier::Init(const string& trained_file, const string& model_file, 
	const string&mean_file, const string&label_file,
	bool gpu_mode) 
{
	if (!gpu_mode)
		Caffe::set_mode(Caffe::CPU);
	else
		Caffe::set_mode(Caffe::GPU);


	/* Load the network. */
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

	/* Load the binaryproto mean file. */
	SetMean(mean_file);

	/* Load labels. */
	if (label_file.size() > 0)
	{
		std::ifstream labels(label_file.c_str());
		CHECK(labels) << "Unable to open labels file " << label_file;
		string line;
		while (std::getline(labels, line))
			labels_.push_back(string(line));
	}
	else
	{
		Blob<float>* output_layer = net_->output_blobs()[0];
		char szlabel[100];
		printf("output ch=%d\n", output_layer->channels());
		for (int i = 0; i < output_layer->channels(); i++)
		{
			sprintf(szlabel, "%d", i);
			labels_.push_back(szlabel);
		}

	}


// 	Blob<float>* output_layer = net_->output_blobs()[0];
// 	CHECK_EQ(labels_.size(), output_layer->channels())
// 		<< "Number of labels is different from the output layer dimension.";

	return true;
}

int Classifier::FindMaxChannelLayer()
{
	const vector<shared_ptr<Blob<float> > >&blobs = net_->blobs();
	int maxchannels = 0;
	int idx = -1;
	for (int i = (int)blobs.size() - 1; i >= 0; i--)
	{
		if (blobs[i]->channels() > maxchannels)
		{
			maxchannels = blobs[i]->channels();
			idx = i;
		}
	}

	printf("max channel layer=%d, channels=%d\n", idx, maxchannels);

	return idx;
}

int Classifier::FindLayerIndex(const string& strLayerName)
{
	const vector<string>& vLayers = net_->layer_names();

	int idx = -1;
	for (int i = (int)vLayers.size() - 1; i >= 0; i--)
	{
		if (vLayers[i] == strLayerName)
		{
			idx = i;
			break;
		}
	}
	return idx;
}

static bool PairCompare(const std::pair<float, int>& lhs,
	const std::pair<float, int>& rhs) {
	return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
	std::vector<std::pair<float, int> > pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(std::make_pair(v[i], (int)i));
	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

	std::vector<int> result;
	for (int i = 0; i < N; ++i)
		result.push_back(pairs[i].second);
	return result;
}
/* Return the top Na predictions. */
std::vector<Prediction> Classifier::Classify(const string& file, int N) {

	cv::Mat img = cv::imread(file, CV_LOAD_IMAGE_COLOR);
	return Classify(img, N);
}

std::vector<Prediction> Classifier::Classify(const unsigned char* pJPGBuffer, int len, int N /*= 5*/)
{
	vector<uchar> jpg(len);
	memcpy(&jpg[0], pJPGBuffer, len);

	cv::Mat img = cv::imdecode(jpg, CV_LOAD_IMAGE_COLOR);

	return Classify(img, N);
}

std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N /*= 5*/)
{
	std::vector<Prediction> predictions;

	if (img.empty())
	{
		std::cout << "img is empty" << std::endl;
		return predictions;
	}

	std::vector<float> output = Predict(img);

#if 0
	bool softmax = false;
	float sum = 0;
	float maxv=output.size()>0 ? output[0]:0;
	for (size_t i = 0; i < output.size(); i++)
	{
		sum += output[i];
		if (sum>1.01f)
			softmax = true;
		if(output[i]>maxv)
			maxv=output[i];
	}
	if (softmax)
	{
		sum = 0;
		std::vector<float> expout(output.size());
		for (size_t i = 0; i < output.size(); i++)
		{
			expout[i] = exp(output[i]-maxv);
			sum += expout[i];
		}
		if (sum>0)
		{
			for (size_t i = 0; i < output.size(); i++)
				output[i] = expout[i] / sum;
		}
	}
#endif
	if ((int)output.size() < N)
		N = (int)output.size();
	std::vector<int> maxN = Argmax(output, N);

	for (int i = 0; i < N; ++i)
	{
		int idx = maxN[i];
		predictions.push_back(std::make_pair(labels_[idx], output[idx]));
		//std::cout << "make pair:" << idx << "," << labels_[idx] << "," << output[idx] <<std::endl;
	}

	//type = Postprocessing(predictions,test);

	return predictions;
}
std::vector< std::vector<PredictionIdx> > Classifier::BatchClassify(const std::vector<cv::Mat>& imgs, int N /*= 5*/)
{
	std::vector<std::vector<PredictionIdx> > predictions;
	if (imgs.size() == 0)
		return predictions;

	std::vector< std::vector<float> > outputs = BatchPredict(imgs);

	predictions.resize(outputs.size());
	for (size_t k = 0; k < outputs.size();k++)
	{
		std::vector<int> maxN = Argmax(outputs[k], N);
		for (int i = 0; i < N; ++i)
		{
			int idx = maxN[i];
			predictions[k].push_back(std::make_pair(idx, outputs[k][idx]));
		}
	}

	return predictions;
}



std::vector<Prediction> Classifier::CropClassify(const cv::Mat& img, int std_size, int crop_num, int N /*= 5*/)
{
	//resize shorter edge to std_size
	int w0 = img.cols, h0 = img.rows;
	int w1 = w0, h1 = w0;
	if (w0 <= h0)
	{
		w1 = std_size;
		h1 = w1*h0 / w0;
	}
	else
	{
		h1 = std_size;
		w1 = h1*w0 / h0;
	}

	cv::Mat imgresize = img;
	if (w0 != w1 || h0 != h1)
		resize(img, imgresize, cv::Size(w1, h1));

	//crop
	cv::Mat imgcrop = imgresize;
	if (crop_num == 1)//center crop
	{
		cv::Rect roi = { (w1 - input_geometry_.width) / 2 - 1, (h1 - input_geometry_.height) / 2 - 1,
			input_geometry_.width, input_geometry_.height };
		if (roi.x < 0) roi.x = 0;
		if (roi.y < 0) roi.y = 0;

		imgcrop = imgresize(roi);
	}

	return Classify(imgcrop, N);
}

std::vector<PredictionIdx> Classifier::ClassifyRtnIdx(const cv::Mat& img, int N /*= 5*/)
{
	std::vector<PredictionIdx> predictions;

	if (img.empty())
	{
		std::cout << "img is empty" << std::endl;
		return predictions;
	}

	std::vector<float> output = Predict(img);

	std::vector<int> maxN = Argmax(output, N);

	for (int i = 0; i < N; ++i)
	{
		int idx = maxN[i];
		predictions.push_back(std::make_pair(idx, output[idx]));
	}

	return predictions;
}

std::vector< std::vector<float> > Classifier::GetLastBlockFeature(const cv::Mat& img)
{
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_,
		input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Preprocess(img, &input_channels);

	net_->Forward();

	const vector<shared_ptr<Blob<float> > >&blobs = net_->blobs();

	int idx = blobs.size() - 1;

	const float* begin = blobs[idx]->cpu_data();
	int dim1 = blobs[idx]->channels();
	int dim2 = blobs[idx]->height();
	vector< vector<float> > vFeatures(dim1);
	for (int i = 0; i < dim1;i++)
	{
		vFeatures[i].resize(dim2);
		memcpy(&vFeatures[i][0], begin + i*dim2, dim2*sizeof(float));
	}

	return vFeatures;
}



/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) 
{
	if (mean_file.rfind(".txt") != string::npos)
	{
		vector<float> chmeans;
		string line;
		ifstream fs(mean_file);
		while (getline(fs, line))
		{
			if (line.size() == 0)
				continue;
			chmeans.push_back((float)atof(line.c_str()));
		}
		if (chmeans.size() != 1 && chmeans.size() != 3)
		{
			printf("wrong mean value format\n");
			return;
		}
		//int meantype = chmeans.size() == 1 ? CV_32FC1 : CV_32FC3;
		//cv::Scalar channel_mean;
		channel_mean_ = chmeans;

		return;
	}
	BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

	/* Convert from BlobProto to Blob<float> */
	Blob<float> mean_blob;
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

#if 0
	/* Compute the global mean pixel value and create a mean image
	* filled with this value. */
	cv::Scalar channel_mean = cv::mean(mean);
	mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
#else
	mean.convertTo(mean_, CV_32FC3);
#endif
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
	Blob<float>* input_layer = net_->input_blobs()[0];
	if (input_geometry_.width != input_layer->shape(3) || input_geometry_.height != input_layer->shape(2))
	{
		input_layer->Reshape(1, num_channels_,
			input_geometry_.height, input_geometry_.width);
		/* Forward dimension change to all layers. */
		net_->Reshape();
	}

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Preprocess(img, &input_channels);

	net_->Forward();

	/* Copy the output layer to a std::vector */
	Blob<float>* output_layer = net_->output_blobs()[0];
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->channels();
	return std::vector<float>(begin, end);
}

std::vector< std::vector<float> > Classifier::BatchPredict(const std::vector<cv::Mat>& imgs)
{
	PrepareBatchInputs(imgs);

	net_->Forward();

	/* Copy the output layer to a std::vector */
	Blob<float>* output_layer = net_->output_blobs()[0];

	const float* begin = output_layer->cpu_data();
	std::vector< std::vector<float> > outs(imgs.size());
	int labelnum = output_layer->channels();
	for (size_t i = 0; i < imgs.size();i++)
	{
		outs[i] = std::vector<float>(begin, begin+ labelnum);
		begin += labelnum;
	}
	
	return outs;
}
std::vector<float> Classifier::GetLayerFeatureMaps(const string& strLayerName, std::vector<int>& outshape)
{
	std::vector<float> v;

	const shared_ptr<Blob<float> >& blob = net_->blob_by_name(strLayerName);

	if (!blob)
		return v;

	const float* begin = blob->cpu_data();
	const float* end = begin + blob->count();
	outshape = blob->shape();
	return std::vector<float>(begin, end);
}
// std::vector<float> Classifier::ExtractFeature(const cv::Mat& img, const string& strLayerName) {
// 	Blob<float>* input_layer = net_->input_blobs()[0];
// 	input_layer->Reshape(1, num_channels_,
// 		input_geometry_.height, input_geometry_.width);
// 	/* Forward dimension change to all layers. */
// 	net_->Reshape();
// 
// 	std::vector<cv::Mat> input_channels;
// 	WrapInputLayer(&input_channels);
// 
// 	Preprocess(img, &input_channels);
// 
// 	net_->Forward();
// 
// 
// 	return GetLayerFeature(strLayerName);
// }


int Classifier::GetFeatureDim()
{
	const vector<shared_ptr<Blob<float> > >&blobs = net_->blobs();
	for (int i = (int)blobs.size() - 1; i >= 0; i--)
	{
		if (blobs[i]->channels() > 1000)
		{
			blobs[i]->channels();
			return blobs[i]->channels();
		}
	}

	return -1;
}


/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	int ch = input_layer->channels();
	for (int k=0;k<input_layer->num();k++)
	{
		for (int i = 0; i < ch; ++i) {
			cv::Mat channel(height, width, CV_32FC1, input_data);
			input_channels->push_back(channel);
			input_data += width * height;
		}
	}
	
}

void Classifier::Preprocess(const cv::Mat& img,
	std::vector<cv::Mat>* input_channels, bool resize_img) {
	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, CV_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, CV_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, CV_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, CV_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_resized;
	if (resize_img && (sample.size() != input_geometry_))
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;

	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	cv::Mat sample_normalized;

	if (!mean_.empty())
	{
		cv::subtract(sample_float, mean_, sample_normalized);
	}
	else
	{
		cv::Scalar channel_mean;
		for (size_t i = 0; i < channel_mean_.size(); i++)
		{
			channel_mean[i] = channel_mean_[i];
		}

		int imgtype = num_channels_ == 3 ? CV_32FC3 : CV_32FC1;
		cv::Mat curmean = cv::Mat(cv::Size(img.cols, img.rows), imgtype, channel_mean);
		cv::subtract(sample_float, curmean, sample_normalized);
	}
 
	cv::split(sample_normalized, *input_channels);

 }

void Classifier::GetInputImageSize(int &w, int &h)
{

	w = input_geometry_.width;
	h = input_geometry_.height;

}



float Classifier::Pruning(float weight_t, const char* saveas_name)
{
	const vector<shared_ptr<Layer<float> > >&layers = net_->layers();
#if 0
	int scale = 1000;
	vector<uint32_t> hist(scale*2+2,0);
	int num1 = 0, num2 = 0;
	for (size_t i = 0; i < layers.size();i++)
	{
		if(layers[i]->blobs().size()==0)
			continue;
		float* weights = layers[i]->blobs()[0]->mutable_cpu_data();
		int num = layers[i]->blobs()[0]->count();
		for (int j = 0; j < num;j++)
		{
			if (weights[j] < 0)
				num1++;
			else if (weights[j] > 0)
				num2++;

			int nw = (int)(fabs(weights[j]) * scale+0.5f);
			if (nw < 0)nw = 0;
			if (nw >= scale)
				nw = scale-1;
			hist[nw]++;
		}
	}
	for (size_t i = 0; i < hist.size();i++)
	{
		if(hist[i])
			printf("%d ", hist[i]);
	}
#endif

	uint64_t sum = 0, pruned = 0;
	for (size_t i = 0; i < layers.size(); i++)
	{
		if (layers[i]->blobs().size() == 0)
			continue;
		float* weights = layers[i]->blobs()[0]->mutable_cpu_data();
		int num = layers[i]->blobs()[0]->count();

		sum += (uint64_t)num;
		for (int j = 0; j < num; j++)
		{
			if (fabs(weights[j])<weight_t)
			{
				weights[j] = 0;
				pruned++;
			}
		}
	}

	if (saveas_name)
	{
		NetParameter net_param;
		net_->ToProto(&net_param, false);
		WriteProtoToBinaryFile(net_param, saveas_name);
	}

	return sum ? float(pruned) / sum:0;
}

//\B8\D0\CA\DCҰ\B9\C0\BCƣ\AC\D0\E8Ҫ\CA\C2\CFȼ\D3\D4\D8ģ\D0\CD
//img\A3\BA\CA\E4\C8\EBͼ\CF\F1
//layerName:\C4\C4һ\B2\E3\B5ĸо\F5Ұ
//x,y\A3\BA\D7\F8\B1\EA
//idxNeuron:\C9\F1\BE\ADԪ\CB\F7\D2\FD\A3\AC-1ʱ\BB\E1\BAϲ\A2\CB\F9\D3\D0\C9\F1\BE\ADԪ\B5ĸо\F5Ұ\A3\AC
cv::Mat Classifier::EstimateReceptiveField(const cv::Mat& img, const string& layerName, int xo, int yo, int idxNeuron, bool islstm, int* width_parts)
{
	//ͨ\B9\FD\B6\D4ȫͼ\CF\F1\CB\D8\D7\F6\D0޸ģ\AC\BF\B4ָ\B6\A8\B2\E3feature map\B5ı仯\C7\E9\BF\F6\A3\AC\C0\B4ȷ\B6\A8ָ\B6\A8\B2\E3ָ\B6\A8\C9\F1\BE\ADԪ\B5ĸо\F5Ұ
	Forward(img, layerName);
	const shared_ptr<Blob<float> >& blob = net_->blob_by_name(layerName);
	const float* begin = blob->cpu_data();
	const float* end = begin + blob->count();
	vector<int> outshape = blob->shape();//BxCxHxW, or WxBxC (lstm)
	std::vector<float> origResponse(begin, end);

	int w1 = 0, h1 = 0;
	int num_feature_maps=0;
	if (!islstm)
	{
		w1 = outshape[3];
		h1 = outshape[2];
		num_feature_maps = outshape[1];
	}
	else
	{
		w1 = outshape[0];
		h1 = 1;
	}

	if (islstm)
		yo = 0;

	if (xo < 0 || xo >= w1 || yo < 0 || yo >= h1)
		return cv::Mat();

	if (width_parts)
		*width_parts = w1;

	int w = img.cols, h = img.rows;
	cv::Mat matRF(cv::Size(w, h), CV_32FC1);
	memset(matRF.data, 0, h*matRF.step1()*sizeof(float));

	int ch = img.channels();
	int ws = img.step1();
	int dim_feature = w1*h1;
	if (islstm)
		dim_feature = outshape[2];

	if (h1 > 1)//\B8߶Ȳ\BB\CA\C71
	{
		const int batch_size = std::min(32,w*h);
		vector<cv::Mat> vImages(batch_size);
		vector<cv::Rect> vRects(batch_size);
		const int step_x = 2, step_y = 2;
		vector<cv::Rect> vRectModifieds;
		int nx = (w + step_x - 1) / step_x + 1, ny = (h + step_y - 1) / step_y + 1;
		int num = 0;
		for (int i=0;i<ny;i++)
		{
			int y0 = i*step_y, y1 = std::min(h - 1, y0 + step_y);
			for (int j=0;j<nx;j++)
			{
				cv::Mat& im = vImages[num];
				img.copyTo(im);
				int x0 = j*step_x, x1 = std::min(w-1,x0 + step_x);
				vRects[num].x = x0, vRects[num].y = y0, vRects[num].width = x1 - x0, vRects[num].height = y1 - y0;
				//modify image
				for (int y=y0;y<y1;y++)
				{
					for (int x=x0;x<x1;x++)
					{
						uchar* p = im.data + y*im.step1() + x*im.channels();
						for (int c = 0; c < ch; c++)
							p[c] = rand() % 256;
					}
				}
				num++;
				if (num >= batch_size || (i==ny-1&&j==nx-1))
				{
					BatchForward(vImages, layerName);
					const float* pout = blob->cpu_data();
					outshape = blob->shape();
					int channels = outshape[1];
					int nfeatures = channels * outshape[0];
					w1 = outshape[3];
					h1 = outshape[2];

					int dim = w1*h1;
					int offsetxoyo = w1*yo + xo;
					float sumdiff = 0;
					const float* pf0 = origResponse.data();
					for (int k=0;k<vImages.size();k++)
					{
						const float* pf1s = pout + channels*dim*k;
						for (int m=0;m<channels;m++)
						{
							sumdiff += fabs(pf1s[m*dim + offsetxoyo] - pf0[m*dim + offsetxoyo]);
						}
						sumdiff /= channels;
						int yend = vRects[k].y + vRects[k].height;
						int xend = vRects[k].x + vRects[k].width;
						for (int y = vRects[k].y; y < yend; y++)
						{
							for (int x = vRects[k].x; x < xend; x++)
							{
								matRF.at<float>(y, x) = sumdiff;
							}
						}
					}

					num = 0;
				}
				 
			}
		}
	}
	else
	{		
		cv::Mat matTemp = img.clone();
		vector<float> vDiffSum(w);

		for (int x = 0; x < w; x++)
		{
			//modify image
			memcpy(matTemp.data, img.data, h*img.step1());
			uchar* pdst = matTemp.data + x*matTemp.channels();
			//const uchar* psrc = img.data + x*matTemp.channels();
			for (int y=0;y<h;y++)
			{
				for (int c=0;c<ch;c++)
				{
					pdst[c] = rand()%256;
				}
				pdst += ws;
				//psrc += ws;
			}
			 
			
			Forward(matTemp, layerName);

			if (!islstm)
			{
				//find the difference
				if (idxNeuron >= 0 && idxNeuron < num_feature_maps)
				{
					const float* pf0 = origResponse.data() + idxNeuron*num_feature_maps + xo;
					const float* pf1 = blob->cpu_data() + idxNeuron*num_feature_maps + xo;
					float diff = fabs(*pf1 - *pf0);
					if (diff < 1e-6)
						diff = 0;
					vDiffSum[x] += diff;
				}
				else
				{
					float sum = 0;
					for (int j = 0; j < num_feature_maps; j++)
					{
						const float* pf0 = origResponse.data() + j*dim_feature + xo;
						const float* pf1 = blob->cpu_data() + j*dim_feature + xo;
						float diff = fabs(*pf1 - *pf0);
						if (diff < 1e-6)
							diff = 0;
						sum += diff;
					}
					vDiffSum[x] += sum / num_feature_maps;
				}
			}
			else
			{
				int T = w1, t = xo;
				float sum = 0;
				const float* pf0 = origResponse.data() + t*dim_feature;
				const float* pf1 = blob->cpu_data() + t*dim_feature;
				for (int i = 0; i < dim_feature; i++)
				{
					float diff = fabs(pf1[i] - pf0[i]);
					if (diff < 1e-6)
						diff = 0;
					sum += diff;
				}
				vDiffSum[x] += sum / dim_feature;
			}
		}
		
		for (int x = 0; x < w; x++)
		{
			for (int y = 0; y < h; y++)
			{
				matRF.at<float>(y,x) = vDiffSum[x]; 
			}
		}
		
	}
	 
	return matRF; 
}

void Classifier::GetLayerFeatureMapSize(int w, int h, const std::string& layerName, int& w1, int& h1)
{
	Blob<float>* input_layer = net_->input_blobs()[0];
	if (w!= input_layer->shape(3) || h != input_layer->shape(2))
	{
		input_layer->Reshape(input_layer->shape(0), input_layer->shape(1),h,w);
		/* Forward dimension change to all layers. */
		net_->Reshape();
	}


	const shared_ptr<Blob<float> >& blob = net_->blob_by_name(layerName);

	if (blob->shape().size() == 4)
	{
		w1 = blob->shape(3);
		h1 = blob->shape(2);
	}
	else if (blob->shape().size() == 3)//lstm (TxBxC)
	{
		w1 = blob->shape(0);
		h1 = 1;
	}
}

bool Classifier::IsCPUMode()
{
	return (Caffe::mode() == Caffe::CPU);
}

void Classifier::Forward(const cv::Mat& img, const string& lastLayerName)
{
	vector<cv::Mat> imgs;
	imgs.push_back(img);
	BatchForward(imgs, lastLayerName);
}



void Classifier::BatchForward(const vector<cv::Mat>& imgs, const string& lastLayerName)
{
	if (!net_->has_layer(lastLayerName))
		return;
	PrepareBatchInputs(imgs);
	net_->ForwardFromTo(0, net_->layer_index_by_name(lastLayerName));
}

void Classifier::PrepareInput(const cv::Mat& img)
{
	vector<cv::Mat> imgs;
	imgs.push_back(img);
	PrepareBatchInputs(imgs);
}

void Classifier::PrepareBatchInputs(const vector<cv::Mat>& imgs)
{
	if (imgs.size() == 0)
		return;
	Blob<float>* input_layer = net_->input_blobs()[0];
	if ((int)imgs.size() != input_layer->shape(0)//image num
		|| imgs[0].cols != input_layer->shape(3) //width
		|| imgs[0].rows != input_layer->shape(2)//height
		)
	{
		input_layer->Reshape(imgs.size(), num_channels_,
			imgs[0].rows, imgs[0].cols);
		/* Forward dimension change to all layers. */
		net_->Reshape();
	}

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);


	for (size_t i = 0; i < imgs.size(); i++)
	{
		vector<cv::Mat> vChannels;
		Preprocess(imgs[i], &vChannels,false);//\BC\F5\BE\F9ֵͼ\A1\A2\B8\A1\B5㻯\A1\A2\B7\D6ͨ\B5\C0
		for (int j = 0; j < num_channels_; j++)
			vChannels[j].copyTo(input_channels[i*num_channels_ + j]);//\B1\D8\D0\EB\D3\C3copyTo\A3\AC\B8\B3ֵ\B2\D9\D7\F7\CA\C7\C4ڴ潻\BB\BB\A3\AC\B8\B3ֵ\B2\BB\BB\E1\D0޸\C4input_layer\B5\C4\C4\DA\C8\DD
	}
}

std::vector<float> Classifier::GetOutputFeatureMap(const cv::Mat& img, std::vector<int>& outshape)
{
	PrepareInput(img);

	net_->Forward();

	Blob<float>* output_layer = net_->output_blobs()[0];
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->count();

	outshape = output_layer->shape();

	return std::vector<float>(begin, end);
}
 
