/**
 * @file stitch.h
 * @brief Independet Functions
 * @version 1.0
 * @date 20/01/2018
 * @author Victor Garcia
 */

#ifndef STITCH_H
#define STITCH_H

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
cv::Rect getBound(cv::Mat H, int width, int height);

/**
 * @brief 
 * 
 * @param obj 
 * @param scene 
 * @param H 
 * @return cv::Mat 
 */
cv::Mat stitch(cv::Mat object, cv::Mat scene, cv::Mat H);

/**
 * @function getGoodMatches(int n_matches, std::vector<cv::DMatch> matches)
 * @brief Discard the matchs outliers
 * @param n_matches Number of matches
 * @param matches Vector of Vectors, container all the Opencv Matches
 * @return vector<cv::DMatch> Vector container good Opencv Matches
 */
std::vector<cv::DMatch> getGoodMatches(const int n_matches, const std::vector<std::vector<cv::DMatch> > matches);

/**
 * @function read_filenames(std::string dir_ent)
 * @brief Get and store the name of files from a directory
 * @param dir_ent Path of the directory to read the file names
 * @return vector<std::string> Vector container the names (sorted alphabetically) of files in the directory 
 */
std::vector<std::string> read_filenames(const std::string dir_ent);

/**
 * @brief Select the best frame for each regular section of the image
 * @function gridDetector(std::vector<cv::KeyPoint> keypoints, std::vector<cv::DMatch> matches)
 * @param keypoints Opecv Keypoint data that contains the position of each keypoint 
 * @param matches Opencv keypoint Match contains information of distance for each match
 * @return std::vector<cv::DMatch> Vector containing the selected matches
 */
std::vector<cv::DMatch> gridDetector(std::vector<cv::KeyPoint> keypoints, std::vector<cv::DMatch> matches);


#endif