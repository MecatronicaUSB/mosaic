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
#include <dirent.h>
#include <iostream>
#include <stdlib.h>
#include <algorithm>
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
std::vector<cv::DMatch> getGoodMatches(const int n_matches, const std::vector<std::vector<cv::DMatch> > matches);

/**
 * @function read_filenames(std::string dir_ent)
 * @brief Get and store the name of files from a directory
 * @param dir_ent Path of the directory to read the file names
 * @return vector<std::string> Vector container the names (sorted alphabetically) of files in the directory 
 */
std::vector<std::string> read_filenames(const std::string dir_ent);

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