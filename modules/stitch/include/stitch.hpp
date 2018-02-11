/**
 * @file stitch.hpp
 * @brief Independet Functions
 * @version 1.0
 * @date 10/02/2018
 * @author Victor Garcia
 */
#ifndef STITCH_STITCH_HPP_
#define STITCH_STITCH_HPP_

using namespace std;
using namespace cv;

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

const int TARGET_WIDTH	= 640;   
const int TARGET_HEIGHT	= 480;

namespace m2d
{

enum referenceImg{
    OBJECT,
    SCENE
};
enum detector{
    USE_KAZE,
    USE_AKAZE
};
enum matcher{
    USE_BRUTE_FORCE,
    USE_FLANN
};

/**
 * @brief 
 * 
 * @param H 
 * @param keypoints 
 * @param matches 
 */
void saveHomographyData(cv::Mat H, vector<KeyPoint> keypoints[2], vector<cv::DMatch> matches);
/**
 * @brief 
 * 
 * @param keypoints 
 * @param matches 
 * @return float 
 */
float boundAreaKeypoints(std::vector<cv::KeyPoint> keypoints, std::vector<cv::DMatch> matches);
/**
 * @brief
 * @param cv::Point2f 
 * @param cv::Point2f 
 * @return float 
 */
float getDistance(cv::Point2f, cv::Point2f);

class Stitcher {
    public:
        // ---------- Atributes
        Mat img[2];         //!< 
        Mat img_ori[2];     //!<
        Size frame_size;                    //!< dimensions of frames       
        bool grid;
        int grid_cells;     //!< number (n) of cell divisions in grid detector. (nxn)
        bool scb_pre;
        // ---------- Methods
        /**
         * @brief 
         */
        Stitcher(bool _grid = false, bool _pre = false, int _width = TARGET_WIDTH, int _height = TARGET_HEIGHT,
                 int _detector = USE_KAZE, int _matcher = USE_BRUTE_FORCE);
        /**
         * @brief
         * @param int 
         */
        void setDetector(int);
        /**
         * @brief
         * @param int 
         */
        void setFeaturesMatcher(int);
        /**
         * @brief
         * @param int 
         */
        void setGrid(int);
        /**
         * @brief
         */
        void setScene(Mat);
        /**
         * @brief 
         */
        void Stitch();
        /**
         * @brief 
         * @return bool
         */
        bool goodframe();

    private:
        // ---------- Atributes

        Mat H;                              //!< Homography matrix
        Ptr<Feature2D> detector;            //!<    
        Ptr<DescriptorMatcher> matcher;     //!<
        vector<cv::DMatch> matches;         //!<                          
        vector<cv::DMatch> good_matches;    //!<     
        vector<KeyPoint> keypoints;         //!<   
        Mat descriptors[2];                 //!< 
        vector<Point2f> warp_points;
        vector<Point2f> border_points;
        Rect bound_rect;
        // ---------- Methods
        /**
         * @brief 
         * @param Mat 
         * @return vector<cv::Point2f> 
         */
        void getBoundPoints(Mat H);
        /**
         * @brief
         * @param vector<vector<cv::DMatch> > 
         * @return vector<cv::DMatch> 
         */
        void getGoodMatches();
        /**
         * @brief 
         * @param keypoints 
         * @param vector<cv::DMatch> 
         * @return vector<cv::DMatch> 
         */
        vector<cv::DMatch> gridDetector(vector<KeyPoint>, vector<cv::DMatch>);
        /**
         * @brief 
         * @param keypoints 
         * @param matches 
         * @return float 
         */
        float boundAreaKeypoints(vector<KeyPoint>, vector<cv::DMatch>);
};
}

#endif