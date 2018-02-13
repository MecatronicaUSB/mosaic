/**
 * @file stitch.hpp
 * @brief Mosaic2d Namespace and Classes
 * @version 0.2
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
enum RefImg{
    SCENE,
    OBJECT,
    SCENE_COLOR,
    OBJECT_COLOR,
    OLD_OBJECT,
    SCENE_KEYPOINTS
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
/// offset of padding to add in scene after object transformation
enum WarpOffset{
    TOP,
    BOTTOM,
    LEFT,
    RIGHT
};

/**
 * @brief Save the homography matrix and heypoints in a txt file
 * @param H OpenCV Matrix containing Homography transformation
 * @param keypoints Vector with OpenCV Keypoints
 * @param matches Vector with OpenCV Matches
 */
void saveHomographyData(cv::Mat H, vector<KeyPoint> keypoints[2], vector<cv::DMatch> matches);
/**
 * @brief Calculate the minimun bounding area containing good keypoints 
 * @param keypoints Vector with OpenCV Keypoints
 * @param matches Vector with OpenCV Matches
 * @return float Area with good keypoints inside
 */
float boundAreaKeypoints(std::vector<cv::KeyPoint> keypoints, std::vector<cv::DMatch> matches);
/**
 * @brief Calculate the euclidean distance between two given vector in 2D
 * @param cv::Point2f First floating point OpenCV coordinate 
 * @param cv::Point2f Second floating point OpenCV coordinate 
 * @return float Distance betwenn two vector
 */
float getDistance(cv::Point2f, cv::Point2f);

class Stitcher {
    public:
        // ---------- Atributes
        Mat scene_keypoints;
        Mat old_H;
        int n_img;                              //!< Number of images in current mosaic
        int cells_div;                          //!< number (n) of cell divisions in grid detector (if used)
        Mat H;                                  //!< Last transformation homography matrix
        vector<Mat> img = vector<Mat>(5);       //!< 
        Size frame_size;                        //!< dimensions of frames       
        bool use_grid;                          //!< flag to use or not the grid detection
        bool apply_pre;                         //!< flag to apply or not SCB preprocessing algorithm
        vector<vector<cv::DMatch> > matches;    //!< Vector of OpenCV Matches                     
        vector<cv::DMatch> good_matches;        //!< Vector of OpenCV good Matches (after discard outliers)
        vector<KeyPoint> keypoints[2];          //!< Array of Vectors containing OpenCV Keypoints
        vector<Point2f> keypoints_coord[2];     //!< X and Y coordinates of keypoints in image
        // ---------- Methods
        /**
         * @brief Default Stitcher constructor
         * 
         * @param _grid flag to use grid detection
         * @param _pre flag to apply or not SCB preprocessing algorith
         * @param _width width of input images
         * @param _height width of input images
         * @param _detector enum value to set the desired feature Detector and descriptor
         * @param _matcher enum value to set the desired feature matcher
         */
        Stitcher(bool _grid = false, bool _pre = false, int _width = TARGET_WIDTH, int _height = TARGET_HEIGHT,
                 int _detector = USE_KAZE, int _matcher = USE_BRUTE_FORCE);
        /**
         * @brief Change the feature Detector and descriptor to use
         * @param int enum value of desired feature Detector and descriptor
         */
        void setDetector(int);
        /**
         * @brief Change the feature matcher to use
         * @param int enum value of desired feature matcher
         */
        void setMatcher(int);
        /**
         * @brief Set the initial scene image
         * @param _scene OpenCV Matrix containing the initial image
         */
        void setScene(Mat _scene);
        /**
         * @brief Warp and stitch the object image in the current scene
         * @param _object OpenCV Matrix containing the image to add to scene
         * @return bool Return true if the stitch was successful, false otherwise
         */
        bool stitch(Mat _object);
        /**
         * @brief Measures the image distortion of next object image
         * @detail
         * Metrics:
         * - Semi-diagonals ratio
         * - Area
         * - Minimum bounding area covered by keypoints
         * @return bool Return true if the image is good enough, false otherwise
         */
        bool goodframe();

    private:
        // ---------- Atributes
        bool old_scene = false;                 //!< flag to know if scene image is a mosaic or a new image
        Ptr<Feature2D> detector;                //!< Pointer to OpenCV feature extractor
        Ptr<DescriptorMatcher> matcher;         //!< Pointer to OpenCV feature Matcher
        Mat descriptors[2];                     //!< Array of OpenCV Matrix conaining feature descriptors
        vector<Point2f> warp_points;            //!< Vector of points after apply homography transformation
        vector<Point2f> border_points;          //!< Vector of points in corners of original frame
        Rect2f bound_rect;                      //!< Minimum bounding rect of warped image (H*objectImg) 
        // ---------- Methods
        /**
         * @brief Discard outliers from initial matches vector
         */
        void getGoodMatches();
        /**
         * @brief Select the best keypoint for each cell in the defined grid
         */
        void gridDetector();
        /**
         * @brief Calculates the minimum bounding area covered by keypoints
         * @return float Area
         */
        float boundAreaKeypoints();
        /**
         * @brief transform a vector of OpenCV Keypoints to vectors of OpenCV Points
         */
        void positionFromKeypoints();
        /**
         * @brief 
         * 
         */
        void drawKeipoints();
        /**
         * @brief Computes the size of pads in the scene based on the transformation of object image
         * @return vector<float> Padd size for each side of scene image
         */
        vector<float> getWarpOffet();
        /**
         * @brief Blend the warped object image to the scene
         * @param _warp_img Input OpenCV Matrix containing warped object image
         */
        void blendToScene(Mat _warp_img);
        /**
         * @brief Clear the used vectors and OpenCV matrix used
         */
        void cleanData();
};
}

#endif