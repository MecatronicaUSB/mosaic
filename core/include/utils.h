/**
 * @file utils.h
 * @brief Usefull functions
 * @version 1.0
 * @date 20/01/2018
 * @author Victor Garcia
 */

#pragma once

#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <assert.h>
#include <dirent.h>
#include <stdlib.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <math.h>
#include <string>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

namespace m2d
{

const int TARGET_WIDTH = 640;
const int TARGET_HEIGHT = 480;


/**
 * @brief Calculate the euclidean distance between two given vector in 2D
 * @param _pt1 First floating point OpenCV coordinate 
 * @param _pt2 Second floating point OpenCV coordinate 
 * @return float Distance betwenn two points
 */
float getDistance(Point2f _pt1, Point2f _pt2);

Point2f getMidPoint(Point2f _pt1, Point2f _pt2);

void enhanceImage(Mat &_img, Mat mask = Mat());

void removeScale(Mat &_H);

int sign(double _num1);
}

/**
 * @brief Computes the intensity distribution histograms for the three channels
 * @function getHistogram(cv::Mat img, int histogram[3][256])
 * @param img OpenCV Matrix container input image
 * @param histogram Integer matrix to store the histogram
 */
void getHistogram(cv::Mat img, int *histogram, Mat mask);
// Histogram[3][256].
//      Histogram[0] corresponds to the Blue channel
//      Histogram[1] corresponds to the Green channel
//      Histogram[2] corresponds to the Red channel

/**
 * @brief Creates an image that represents the Histogram
 * @function printHistogram(int histogram[256], std::string filename, cv::Scalar color)
 * @param histogram Integer array correspond to a histogram
 * @param filename name of output file to print the histogram
 * @param color OpenCV Scalar containing the color of bars in histogram
 */
void printHistogram(int histogram[256], std::string filename, cv::Scalar color);

/**
 * @brief Transform imgOriginal so that, for each channel histogram, its
          lowerPercentile and higherPercentile values are moved to 0 and 255, respectively
 * @function colorChannelStretch(cv::Mat imgOriginal, cv::Mat imgStretched, int lowerPercentile, int higherPercentile)
 * @param imgOriginal OpenCV Matrix containing input image
 * @param imgStretched OpenCV Matrix to store the stretched output image
 * @param lowerPercentile Percentile to trunk the lower values
 * @param higherPercentile Percentile to trunk the higher values
 * \n
 * \b CONSTRAINTS: \n
 * \e imgOriginal and \e imgStretched must have the same dimensions.\n
 * \e lowerPercentile and \e higherPercentle must be integers between
 * 0 and 100.\n
 * \e lowerPercentile must be smaller than \e higherPercentile
 */
void imgChannelStretch(cv::Mat imgOriginal, cv::Mat imgStretched, int lowerPercentile, int higherPercentile, Mat mask = Mat());
// Transform imgOriginal so that, for each channel histogram, its
// lowerPercentile and higherPercentile values are moved to 0 and 255,
// respectively. Values in between are linearly scaled. Values smaller
// than lowerPercentile are set to 0, and values greater than
// higherPercentle are set to 1. The resulting image is saved in
// imgStretched.
// CONSTRAINTS:
//      * imgOriginal and imgStretched must have the same dimensions.
//      * lowerPercentile and higherPercentle must be integers between
//        0 and 100, and lowerPercentile must be smaller than
//        higherPercentile

/**
 * @function read_filenames(std::string dir_ent)
 * @brief Get and store the name of files from a directory
 * @param dir_ent Path of the directory to read the file names
 * @return vector<std::string> Vector container the names (sorted alphabetically) of files in the directory 
 */
std::vector<std::string> read_filenames(const std::string dir_ent);

/**
 * @brief Save the homography matrix and heypoints in a txt file
 * @param H OpenCV Matrix containing Homography transformation
 * @param keypoints Vector with OpenCV Keypoints
 * @param matches Vector with OpenCV Matches
 */
void saveHomographyData(cv::Mat _h, vector<KeyPoint> keypoints[2], vector<cv::DMatch> matches);