/**
 * @file detector.h
 * @brief Independet Functions
 * @version 1.0
 * @date 20/01/2018
 * @author Victor Garcia
 */

#ifndef DETECTOR_H
#define DETECTOR_H

#include <vector>
#include <iostream>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"

/**
 * @function getGoodMatches(int n_matches, std::vector<cv::DMatch> matches)
 * @brief Discard the matchs outliers
 * @param n_matches Number of matches
 * @param matches Vector of Vectors, container all the Opencv Matches
 * @return vector<cv::DMatch> Vector container good Opencv Matches
 */
std::vector<cv::DMatch> getGoodMatches(const std::vector<std::vector<cv::DMatch> > matches);
std::vector<cv::DMatch> getGoodMatchesBF(const std::vector<cv::DMatch> matches);
/**
 * @brief 
 * 
 * @param src 
 * @param detector 
 * @param keypoints 
 * @param descriptors 
 */
std::vector<cv::DMatch> gridDetector(std::vector<cv::KeyPoint> keypoints, std::vector<cv::DMatch> matches);

#endif