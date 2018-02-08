/**
 * @file stitch.h
 * @brief Independet Functions
 * @version 1.0
 * @date 20/01/2018
 * @author Victor Garcia
 */
#ifndef STITCH_STITCH_H_
#define STITCH_STITCH_H_

#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/imgproc.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include <dirent.h>
#include <stdlib.h>
#include <iostream>
#include <vector>

#define TARGET_WIDTH	640   
#define TARGET_HEIGHT	480

/**
 * @brief 
 * 
 * @param H 
 * @param width 
 * @param height 
 * @return cv::Rect 
 */
std::vector<cv::Point2f> getBoundPoints(cv::Mat H, int width, int height);

/**
 * @brief 
 * 
 * @param object 
 * @param scene 
 * @param H 
 * @return struct WarpPoly 
 */
cv::Rect stitch(cv::Mat object, cv::Mat& scene, cv::Mat H);

/**
 * @function getGoodMatches(int n_matches, std::vector<cv::DMatch> matches)
 * @brief Discard the matchs outliers
 * @param n_matches Number of matches
 * @param matches Vector of Vectors, container all the Opencv Matches
 * @return vector<cv::DMatch> Vector container good Opencv Matches
 */
std::vector<cv::DMatch> getGoodMatches(const int n_matches, const std::vector<std::vector<cv::DMatch> > matches);

/**
 * @brief Select the best frame for each regular section of the image
 * @function gridDetector(std::vector<cv::KeyPoint> keypoints, std::vector<cv::DMatch> matches)
 * @param keypoints Opecv Keypoint data that contains the position of each keypoint 
 * @param matches Opencv keypoint Match contains information of distance for each match
 * @return std::vector<cv::DMatch> Vector containing the selected matches
 */
std::vector<cv::DMatch> gridDetector(std::vector<cv::KeyPoint> keypoints, std::vector<cv::DMatch> matches);

/**
 * @brief 
 * 
 * @param img 
 * @param offsetx 
 * @param offsety 
 * @return Mat 
 */
cv::Mat translateImg(cv::Mat img, double offsetx, double offsety);

/**
 * @brief 
 * 
 * @param H 
 * @param keypoints 
 */
void saveHomographyData(cv::Mat H, std::vector<cv::KeyPoint> keypoints[2], std::vector<cv::DMatch> matches);

/**
 * @brief Function to evaluate if the transformed image is distorted
 * @param H Opencv Matrix containing Homography matrix
 * @param width Width of input image
 * @param height Height of input image
 * @return bool Boolean value, true if the image is not distorted enough, false otherwise
 */
bool imageDistortion(cv::Mat H, int width, int height);

/**
 * @brief Calculate the distance between two input points
 * @param pt1 OpenCV floating point coordinate
 * @param pt2 OpenCV floating point coordinate
 * @return float Distance between points
 */
float getDistance(cv::Point2f pt1, cv::Point2f pt2);

#endif