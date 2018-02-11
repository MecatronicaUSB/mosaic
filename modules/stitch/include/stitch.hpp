/**
 * @file stitch.hpp
 * @brief Independet Functions
 * @version 1.0
 * @date 10/02/2018
 * @author Victor Garcia
 */
#ifndef STITCH_STITCH_HPP_
#define STITCH_STITCH_HPP_

#include "../../common/utils.h"
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
#include <cmath> 
#include <vector>

using namespace std;
using namespace cv;

const int TARGET_WIDTH	= 640;   
const int TARGET_HEIGHT	= 480;

namespace m2d //!< mosaic 2d namespace
{
/// Reference frame enumeration
enum ReferenceImg{
    SCENE,
    OBJECT
};
/// Keypoint detector and descriptor to use
enum Detector{
    USE_KAZE,
    USE_AKAZE,
    USE_SIFT,
    USE_SURF
};
/// Keypoint matcher to use
enum Matcher{
    USE_BRUTE_FORCE,
    USE_FLANN
};
/// offset to move scene after transform object
enum WarpOffset{
    TOP,
    BOTTOM,
    LEFT,
    RIGHT
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
        int n_img;
        Mat object;                             //!< 
        Mat scene;                              //!< 
        Mat object_ori;                         //!<
        Mat scene_ori;                          //!<
        Size frame_size;                        //!< dimensions of frames       
        bool use_grid;                          //!<
        int n_cells;                            //!< number (n) of cell divisions in grid detector. (nxn)
        bool apply_pre;                         //!<
        vector<vector<cv::DMatch> > matches;    //!< Vector of OpenCV Matches                     
        vector<cv::DMatch> good_matches;        //!< Vector of OpenCV good Matches (after discard outliers)
        vector<KeyPoint> keypoints[2];          //!< Array of Vectors containing OpenCV Keypoints
        vector<Point2f> keypoints_coord[2];     //!< X and Y coordinates of keypoints in image
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
        void setMatcher(int);
        /**
         * @brief
         */
        void setScene(Mat _scene);
        /**
         * @brief 
         */
        bool stitch(Mat _object);
        /**
         * @brief 
         * @return bool
         */
        bool goodframe();

    private:
        // ---------- Atributes
        Mat H;                                  //!< Homography matrix
        Ptr<Feature2D> detector;                //!< Pointer to OpenCV feature extractor
        Ptr<DescriptorMatcher> matcher;         //!< Pointer to OpenCV feature Matcher
        Mat descriptors[2];                     //!< Array of OpenCV Matrix conaining feature descriptors
        vector<Point2f> warp_points;            //!< Vector of points after apply homography transformation
        vector<Point2f> border_points;          //!< Vector of points in corners of original frame
        Rect2f bound_rect;                      //!< Minimum bounding rect of warped image (H*objectImg) 
        // ---------- Methods
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
        void gridDetector();
        /**
         * @brief 
         * @param keypoints 
         * @param matches 
         * @return float 
         */
        float boundAreaKeypoints();
        /**
         * @brief 
         */
        void positionFromKeypoints();
        /**
         * @brief
         * @return vector<float> 
         */
        vector<float> getWarpOffet();
        /**
         * @brief 
         */
        void blendToScene(Mat _warp_img);
        /**
         * @brief 
         */
        void cleanData();
};
}

#endif