/**
 * @file utils.cpp
 * @brief useful functions
 * @version 1.0
 * @date 20/01/2018
 * @author Victor Garcia
 */

#include "../include/utils.h"

namespace m2d
{

// See description in header file
float getDistance(Point2f _pt1, Point2f _pt2)
{
	return sqrt(pow((_pt1.x - _pt2.x), 2) + pow((_pt1.y - _pt2.y), 2));
}

Point2f getMidPoint(Point2f _pt1, Point2f _pt2)
{
	return Point2f((_pt2.x + _pt1.x) / 2, (_pt2.y + _pt1.y) / 2);
}

void enhanceImage(Mat &_img, Mat mask)
{
	vector<Mat> channels;

	// cvtColor( _img, _img, CV_BGR2HSV);
	// split(_img, channels);
	// imgChannelStretch(channels[2], channels[2], 1, 99);
	// merge(channels, _img);
	// cvtColor( _img, _img, CV_HSV2BGR);

	split(_img, channels);
	imgChannelStretch(channels[0], channels[0], 1, 99, mask);
	imgChannelStretch(channels[1], channels[1], 1, 99, mask);
	imgChannelStretch(channels[2], channels[2], 1, 99, mask);
	merge(channels, _img);

}

int sign(double _num1)
{
	return _num1 > 0 ? 1 : -1;
}
}

void getHistogram(cv::Mat img, int *histogram, Mat mask)
{
	int i = 0, j = 0;
	//    std::cout << "gH: Initializing histogram vector" << endl;
	// Initializing the histogram. TODO: Check if there is a faster way
	for (i = 0; i < 256; i++)
	{
		histogram[i] = 0;
	}
	// by using aux variables, we decrease overhead create by multiple calls to cvMat image methods to retrieve its size
	// TODO: is it possible to measure the impact?
	int width, height;
	width = img.size().width;
	height = img.size().height;

	// Computing the histogram as a cumulative of each integer value. WARNING: this will fail for any non-integer image matrix
	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
			if (mask.at<unsigned char>(i, j) != 0)
				histogram[img.at<unsigned char>(i, j)]++;
}

void printHistogram(int histogram[256], std::string filename, cv::Scalar color)
{
	// Finding the maximum value of the histogram. It will be used to scale the
	// histogram to fit the image.
	int max = 0, i;
	for (i = 0; i < 256; i++)
	{
		if (histogram[i] > max)
			max = histogram[i];
	}
	// Creating an image from the histogram.
	cv::Mat imgHist(1480, 1580, CV_8UC3, cv::Scalar(255, 255, 255));
	cv::Point pt1, pt2;
	pt1.y = 1380;
	for (i = 0; i < 256; i++)
	{
		pt1.x = 150 + 5 * i + 1;
		pt2.x = 150 + 5 * i + 3;
		pt2.y = 1380 - 1280 * histogram[i] / max;
		cv::rectangle(imgHist, pt1, pt2, color, CV_FILLED);
	}
	// y-axis labels
	cv::rectangle(imgHist, cv::Point(130, 1400), cv::Point(1450, 80), cvScalar(0, 0, 0), 1);
	cv::putText(imgHist, std::to_string(max), cv::Point(10, 100), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0, 0, 0), 2.0);
	cv::putText(imgHist, std::to_string(max * 3 / 4), cv::Point(10, 420), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0, 0, 0), 2.0);
	cv::putText(imgHist, std::to_string(max / 2), cv::Point(10, 740), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0, 0, 0), 2.0);
	cv::putText(imgHist, std::to_string(max / 4), cv::Point(10, 1060), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0, 0, 0), 2.0);
	cv::putText(imgHist, std::to_string(0), cv::Point(10, 1380), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0, 0, 0), 2.0);
	// x-axis labels
	cv::putText(imgHist, std::to_string(0), cv::Point(152 - 7 * 1, 1430), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0, 0, 0), 2.0);
	cv::putText(imgHist, std::to_string(63), cv::Point(467 - 7 * 2, 1430), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0, 0, 0), 2.0);
	cv::putText(imgHist, std::to_string(127), cv::Point(787 - 7 * 3, 1430), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0, 0, 0), 2.0);
	cv::putText(imgHist, std::to_string(191), cv::Point(1107 - 7 * 3, 1430), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0, 0, 0), 2.0);
	cv::putText(imgHist, std::to_string(255), cv::Point(1427 - 7 * 3, 1430), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(0, 0, 0), 2.0);

	// Saving the image
	cv::imwrite(filename, imgHist);
}

// Now it will operate in a single channel of the provided image. So, future implementations will require a function call per channel (still faster)
void imgChannelStretch(cv::Mat imgOriginal, cv::Mat imgStretched, int lowerPercentile, int higherPercentile, Mat mask)
{
	// Computing the histograms
	int histogram[256];
	if (!mask.data)
		mask = Mat(imgOriginal.size(), CV_8U, Scalar(255));

	getHistogram(imgOriginal, histogram, mask);

	// Computing the percentiles. We force invalid values as initial values (just in case)
	int channelLowerPercentile = -1, channelHigherPercentile = -1;
	int height = imgOriginal.size().height;
	int width = imgOriginal.size().width;
	// Channel percentiles
	int i = 0;
	float sum = 0;

	float normImgSize = height * width / 100.0;
	// while we don't reach the highPercentile threshold...
	// This is some fashion of CFD: cumulative function distribution
	while (sum < higherPercentile * normImgSize)
	{
		if (sum < lowerPercentile * normImgSize)
			channelLowerPercentile++;
		channelHigherPercentile++;
		sum += histogram[i++];
	}
	int j;
	float m;
	cv::Scalar b;
	m = 255 / (channelHigherPercentile - channelLowerPercentile);
	b = channelLowerPercentile;
	imgStretched -= b;
	imgStretched *= m;
}

// See description in header file
std::vector<std::string> read_filenames(const std::string dir_ent)
{
	std::vector<std::string> file_names;
	DIR *dir;
	struct dirent *ent;

	if ((dir = opendir(dir_ent.c_str())) != NULL)
	{
		while ((ent = readdir(dir)) != NULL)
		{
			file_names.push_back(dir_ent + '/' + std::string(ent->d_name));
		}
		closedir(dir);
	}
	else
	{
		// If the directory could not be opened
		std::cout << "Directory could not be opened" << std::endl;
	}
	// Sorting the vector of strings so it is alphabetically ordered
	std::sort(file_names.begin(), file_names.end());
	file_names.erase(file_names.begin(), file_names.begin() + 2);

	return file_names;
}
