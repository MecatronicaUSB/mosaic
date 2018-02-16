/**
 * @file stitch.hpp
 * @brief Mosaic2d Namespace and Classes
 * @version 0.2
 * @date 10/02/2018
 * @author Victor Garcia
 */
#pragma once

#include "../../common/utils.h"
#include "../include/mosaic.hpp"
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


namespace m2d //!< mosaic 2d namespace
{
/// Reference image for stitching class
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
/// Keypoint matcher to use in stitcher class
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

class Frame;

struct StitchStatus{
    bool ok = false;
    vector<float> offset;
};

class Stitcher {
    public:
        // ---------- Atributes
        Mat offset_h;
        Mat scene_keypoints;                    //!< Image to draw the keypoints after one match (testing purpose)
        int cells_div;                          //!< number (n) of cell divisions in grid detector (if used)
        vector<Frame*> img = vector<Frame*>(2); //!< Vector of two frames to stitch (Pointers)
        bool use_grid;                          //!< flag to use or not the grid detection
        bool apply_pre;                         //!< flag to apply or not SCB preprocessing algorithm
        vector<vector<cv::DMatch> > matches;    //!< Vector of OpenCV Matches                     
        vector<cv::DMatch> good_matches;        //!< Vector of OpenCV good Matches (after discard outliers)
        vector<KeyPoint> keypoints[2];          //!< Array of Vectors containing OpenCV Keypoints
        // ---------- Methods
        /**
         * @brief Default Stitcher constructor
         * 
         * @param _grid flag to use grid detection (default true)
         * @param _pre flag to apply or not SCB preprocessing algorith (default false)
         * @param _width width of input images
         * @param _height width of input images
         * @param _detector enum value to set the desired feature Detector and descriptor
         * @param _matcher enum value to set the desired feature matcher
         */
        Stitcher(bool _grid = false, bool _pre = false,
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
         * @param *_frame Pointer to the reference Frame
         */
        void setScene(Frame *_frame);
        /**
         * @brief Warp and stitch the object image in the current scene
         * @param _object OpenCV Matrix containing the image to add to scene
         * @return bool Return true if the stitch was successful, false otherwise
         */
        /**
         * @brief Warp and stitch the object image in the current scene
         * @param *_object Contains the Frame to be add in the current scene
         * @param *_scene Contains the last Frame added in the mosaic
         * @param _final_scene OpenCV Matrix containing the final image to stitch the object
         * @return true If the stitch was sucessfull
         * @return false If the stitch wasn't sucessfull
         */
        struct StitchStatus stitch(Frame *_object, Frame *_scene);
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
        void drawKeipoints(vector<float> _warp_offset, Mat &_final_scene);
        /**
         * @brief Computes the size of pads in the scene based on the transformation of object image
         * @param _H Homography matrix
         * @param _scene_dims Dimensions of scene image
         * @return vector<float> Padd size for each side of scene image
         */
        vector<float> getWarpOffet(Mat _H, Size _scene_dims);
        /**
         * @brief  Blend the warped object image to the scene
         * @param _final_scene Input OpenCV Matrix containing warped object image
         */
        void blend2Scene(Mat &_final_scene);
        /**
         * @brief Clear the used vectors and OpenCV matrix used
         */
        void cleanData();
};

}
