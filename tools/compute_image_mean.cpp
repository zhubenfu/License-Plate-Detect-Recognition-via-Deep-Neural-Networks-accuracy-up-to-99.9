#include <stdint.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include <fstream>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace caffe;  // NOLINT(build/namespaces)

using std::max;
using std::pair;
using boost::scoped_ptr;

DEFINE_string(backend, "lmdb",
        "The backend {leveldb, lmdb} containing the images");

static void FindAllImages(const char *folder, vector<string>& vImgPaths, bool bSubFolder)
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


int GetGlobleChannelMean(int argc, char** argv)
{

#ifdef USE_OPENCV
#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif

	gflags::SetUsageMessage("Compute the mean_image of a set of images given by"
		" a leveldb/lmdb\n"
		"Usage:\n"
		"    compute_image_mean [FLAGS] INPUT_DB [OUTPUT_FILE]\n");

	gflags::ParseCommandLineFlags(&argc, &argv, true);

	if (argc < 2 || argc > 3) {
		gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/compute_image_mean");
		return 1;
	}

	scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
	db->Open(argv[1], db::READ);
	scoped_ptr<db::Cursor> cursor(db->NewCursor());

	double sum_channels[3] = { 0 };
	double sum_pixels = 0;

	int count = 0;
	LOG(INFO) << "Starting Iteration";
	while (cursor->valid()) {
		Datum datum;
		datum.ParseFromString(cursor->value());
		cv::Mat mat = DecodeDatumToCVMatNative(datum);

		vector<cv::Mat> chmats;
		split(mat, chmats);
		if (chmats.size() == 3)
		{
			for (int i = 0; i < 3;i++)
			{
				cv::Scalar sumch = sum(chmats[i]);
				sum_channels[i] += sumch(0);
			}
			sum_pixels += chmats[0].cols*chmats[0].rows;
		
		}
		else if (chmats.size() == 1)
		{
			cv::Scalar sumch = sum(chmats[0]);
			sum_channels[0] += sumch(0);
			sum_channels[1] += sumch(0);
			sum_channels[2] += sumch(0);
			sum_pixels += chmats[0].cols*chmats[0].rows;
		}

		++count;
		if (count % 10000 == 0) {
			LOG(INFO) << "Processed " << count << " files." << "channel means: "
				<< sum_channels[0] / sum_pixels << ","
				<< sum_channels[1] / sum_pixels << ","
				<< sum_channels[2] / sum_pixels;
		}
		cursor->Next();
	}

	if (count % 10000 != 0) {
		LOG(INFO) << "Processed " << count << " files.";
	}
	// Write to disk
	if (argc == 3) {
		LOG(INFO) << "Write to " << argv[2];
		std::ofstream ofs(argv[2]);
		ofs << sum_channels[0] / sum_pixels << std::endl;
		ofs << sum_channels[1] / sum_pixels << std::endl;
		ofs << sum_channels[2] / sum_pixels << std::endl;
	}
	
#else
	LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
	return 0;
}
int GetGlobleChannelMeanFromImageFiles(int argc, char** argv)
{

#ifdef _DEBUG
	argc = 3;
	argv[1] = "I:\\homework\\BlurRegression\\images\\";
	argv[2] = "I:\\homework\\BlurRegression\\mean.txt";
#endif

#ifdef USE_OPENCV
#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif

	gflags::SetUsageMessage("Compute the mean_image of a set of images given by"
		" a image folder\n"
		"Usage:\n"
		"    compute_image_mean IMAGE_FOLDER [OUTPUT_FILE]\n");

	if (argc != 3) {
		gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/compute_image_mean");
		return 1;
	}

	vector<string> imgs;
	FindAllImages(argv[1], imgs, false);

	double sum_channels[3] = { 0 };
	double sum_pixels = 0;

	int count = 0;
	LOG(INFO) << "Starting Iteration";
	for (size_t i = 0; i < imgs.size();i++)
	{
		cv::Mat mat = cv::imread(imgs[i]);

		vector<cv::Mat> chmats;
		split(mat, chmats);
		if (chmats.size() == 3)
		{
			for (int i = 0; i < 3; i++)
			{
				cv::Scalar sumch = sum(chmats[i]);
				sum_channels[i] += sumch(0);
			}
			sum_pixels += chmats[0].cols*chmats[0].rows;

		}
		else if (chmats.size() == 1)
		{
			cv::Scalar sumch = sum(chmats[0]);
			sum_channels[0] += sumch(0);
			sum_channels[1] += sumch(0);
			sum_channels[2] += sumch(0);
			sum_pixels += chmats[0].cols*chmats[0].rows;
		}

		++count;
		if (count % 10000 == 0) 
		{
			LOG(INFO) << "Processed " << count << " files." << "channel means: "
				<< sum_channels[0] / sum_pixels << ","
				<< sum_channels[1] / sum_pixels << ","
				<< sum_channels[2] / sum_pixels;
		}
		
	}

	if (count % 10000 != 0) {
		LOG(INFO) << "Processed " << count << " files.";
	}
	// Write to disk
	if (argc == 3) {
		LOG(INFO) << "Write to " << argv[2];
		std::ofstream ofs(argv[2]);
		ofs << sum_channels[0] / sum_pixels << std::endl;
		ofs << sum_channels[1] / sum_pixels << std::endl;
		ofs << sum_channels[2] / sum_pixels << std::endl;
	}

#else
	LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
	return 0;
}


int GetChannelMean(int argc, char** argv)
{

#ifdef USE_OPENCV
#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif

	gflags::SetUsageMessage("Compute the mean_image of a set of images given by"
		" a leveldb/lmdb\n"
		"Usage:\n"
		"    compute_image_mean [FLAGS] INPUT_DB [OUTPUT_FILE]\n");

	gflags::ParseCommandLineFlags(&argc, &argv, true);

	if (argc < 2 || argc > 3) {
		gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/compute_image_mean");
		return 1;
	}

	scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
	db->Open(argv[1], db::READ);
	scoped_ptr<db::Cursor> cursor(db->NewCursor());

	BlobProto sum_blob;
	int count = 0;
	// load first datum
	Datum datum;
	datum.ParseFromString(cursor->value());

	if (DecodeDatumNative(&datum)) {
		LOG(INFO) << "Decoding Datum";
	}

	sum_blob.set_num(1);
	sum_blob.set_channels(datum.channels());
	sum_blob.set_height(datum.height());
	sum_blob.set_width(datum.width());
	const int data_size = datum.channels() * datum.height() * datum.width();
	int size_in_datum = std::max<int>(datum.data().size(),
		datum.float_data_size());
	for (int i = 0; i < size_in_datum; ++i) {
		sum_blob.add_data(0.);
	}
	LOG(INFO) << "Starting Iteration";
	while (cursor->valid()) {
		Datum datum;
		datum.ParseFromString(cursor->value());
		DecodeDatumNative(&datum);

		const std::string& data = datum.data();
		size_in_datum = std::max<int>(datum.data().size(),
			datum.float_data_size());
		CHECK_EQ(size_in_datum, data_size) << "Incorrect data field size " <<
			size_in_datum;
		if (data.size() != 0) {
			CHECK_EQ(data.size(), size_in_datum);
			for (int i = 0; i < size_in_datum; ++i) {
				sum_blob.set_data(i, sum_blob.data(i) + (uint8_t)data[i]);
			}
		}
		else {
			CHECK_EQ(datum.float_data_size(), size_in_datum);
			for (int i = 0; i < size_in_datum; ++i) {
				sum_blob.set_data(i, sum_blob.data(i) +
					static_cast<float>(datum.float_data(i)));
			}
		}
		++count;
		if (count % 10000 == 0) {
			LOG(INFO) << "Processed " << count << " files.";
		}
		cursor->Next();
	}

	if (count % 10000 != 0) {
		LOG(INFO) << "Processed " << count << " files.";
	}
	for (int i = 0; i < sum_blob.data_size(); ++i) {
		sum_blob.set_data(i, sum_blob.data(i) / count);
	}
	// Write to disk
	if (argc == 3) {
		LOG(INFO) << "Write to " << argv[2];
		WriteProtoToBinaryFile(sum_blob, argv[2]);
	}
	const int channels = sum_blob.channels();
	const int dim = sum_blob.height() * sum_blob.width();
	std::vector<float> mean_values(channels, 0.0);
	LOG(INFO) << "Number of channels: " << channels;
	for (int c = 0; c < channels; ++c) {
		for (int i = 0; i < dim; ++i) {
			mean_values[c] += sum_blob.data(dim * c + i);
		}
		LOG(INFO) << "mean_value channel [" << c << "]:" << mean_values[c] / dim;
	}
#else
	LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
	return 0;
}

vector<double> GetChannelMean(scoped_ptr<db::Cursor>& cursor)
{
	vector<double> meanv(3, 0);

	int count = 0;
	LOG(INFO) << "Starting Iteration";
	while (cursor->valid()) {
		Datum datum;
		datum.ParseFromString(cursor->value());
		DecodeDatumNative(&datum);

		const std::string& data = datum.data();
		int w = datum.width(), h = datum.height();
		int ch = datum.channels();
		int dim = w*h;
		double chmean[3] = { 0,0,0 };
		for (int i = 0; i < ch;i++)
		{
			int chstart = i*dim;
			for (int j = 0; j < dim;j++)
				chmean[i] += (uint8_t)data[chstart+j];
			chmean[i] /= dim;
		}
		if (ch == 1)
		{
			meanv[0] += chmean[0];
			meanv[1] += chmean[0];
			meanv[2] += chmean[0];
		}
		else
		{
			meanv[0] += chmean[0];
			meanv[1] += chmean[1];
			meanv[2] += chmean[2];
		}
		
		++count;
		if (count % 10000 == 0) {
			LOG(INFO) << "Processed " << count << " files.";
		}
		cursor->Next();
	}

	if (count % 10000 != 0) {
		LOG(INFO) << "Processed " << count << " files.";
	}

	for (int c = 0; c < 3; ++c) {
		LOG(INFO) << "mean_value channel [" << c << "]:" << meanv[c] / count;
	}

	return meanv;
}

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

  FLAGS_alsologtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  GetGlobleChannelMeanFromImageFiles(argc, argv);

  return 0;
}
