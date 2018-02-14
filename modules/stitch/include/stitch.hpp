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
enum ImgRef{
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
/// offset of padding to add in scene after object transformation
enum WarpOffset{
    TOP,
    BOTTOM,
    LEFT,
    RIGHT
};

enum FrameRef{
    PREV,
    NEXT
};

class Frame{
    public:
        Mat H;
        Rect2f bound_rect;
        Mat color;
        Mat gray;
        vector<Point2f> bound_points;
        vector<Point2f> keypoints_pos[2];
        vector<Frame*> neighbors;
        Frame *key_frame;
        bool key;

        Frame(Mat _img,  bool _key = false, int _width = TARGET_WIDTH, int _height = TARGET_HEIGHT);
        void setHReference(Mat _H);
        void trackKeypoints();
        bool isGoodFrame();
        /**
         * @brief Calculate the minimun bounding area containing good keypoints 
         * @param keypoints Vector with OpenCV Keypoints
         * @param matches Vector with OpenCV Matches
         * @return float Area with good keypoints inside
         */
        float boundAreaKeypoints();
        /**
         * @brief Calculate the euclidean distance between two given vector in 2D
         * @param cv::Point2f First floating point OpenCV coordinate 
         * @param cv::Point2f Second floating point OpenCV coordinate 
         * @return float Distance betwenn two vector
         */
        float getDistance(Point2f _pt1, Point2f _pt2);

};

class Stitcher {
    public:
        // ---------- Atributes
        Mat offset_H;
        Mat scene_keypoints;
        int cells_div;                          //!< number (n) of cell divisions in grid detector (if used)
        vector<Frame*> img = vector<Frame*>(2); //!<     
        bool use_grid;                          //!< flag to use or not the grid detection
        bool apply_pre;                         //!< flag to apply or not SCB preprocessing algorithm
        vector<vector<cv::DMatch> > matches;    //!< Vector of OpenCV Matches                     
        vector<cv::DMatch> good_matches;        //!< Vector of OpenCV good Matches (after discard outliers)
        vector<KeyPoint> keypoints[2];          //!< Array of Vectors containing OpenCV Keypoints
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
        void setScene(Frame *_frame);
        /**
         * @brief Warp and stitch the object image in the current scene
         * @param _object OpenCV Matrix containing the image to add to scene
         * @return bool Return true if the stitch was successful, false otherwise
         */
        bool stitch(Frame *_object, Frame *_scene, Mat& _final_scene);
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
        Ptr<Feature2D> detector;                //!< Pointer to OpenCV feature extractor
        Ptr<DescriptorMatcher> matcher;         //!< Pointer to OpenCV feature Matcher
        Mat descriptors[2];                     //!< Array of OpenCV Matrix conaining feature descriptors
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
        vector<float> getWarpOffet(Mat _H, Size _scene_dims);
        /**
         * @brief Blend the warped object image to the scene
         * @param _warp_img Input OpenCV Matrix containing warped object image
         */
        void blend2Scene(Mat &_final_scene);
        /**
         * @brief Clear the used vectors and OpenCV matrix used
         */
        void cleanData();
};

}

#endif