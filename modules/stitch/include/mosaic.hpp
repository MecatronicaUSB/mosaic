/**
 * @file mosaic.hpp
 * @brief Mosaic2d Namespace and Classes
 * @version 0.2
 * @date 10/02/2018
 * @author Victor Garcia
 */
#pragma once

#include "../../common/utils.h"
#include "../include/stitch.hpp"
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
const int TARGET_WIDTH	= 640;   
const int TARGET_HEIGHT	= 480;

class Stitcher;
/**
 * @brief All data for each image in a mosaic
 * @detail Contains All data of each image refered to current sub-mosaic or mosaic
 */
class Frame{
    public:
        // ---------- Atributes
        bool key;                         //!< Flag to specify reference frame
        Mat H;                            //!< Homography matrix based on previous frame
        Rect2f bound_rect;                //!< Minimum bounding rectangle of transformed image
        Mat color;                        //!< OpenCV Matrix containing the original image
        Mat gray;                         //!< OpenCV Matrix containing a gray scale version of image
        vector<Point2f> bound_points;     //!< Points of the transformmed image (initially at corners)
        vector<Point2f> keypoints_pos[2]; //!< Position (X,Y) of good keypoints in image 
        vector<Frame*> neighbors;         //!< Vector containing all spatially close Frames (Pointers)

        // ---------- Methods
        /**
         * @brief Default class constructor
         * @param _img OpenCV Matrix containing the BGR version of image
         * @param _key Flag to assign this frame as reference (usefull for SubMosaic Class)
         * @param _width Width to resize the image (Speed purpose)
         * @param _height Height to resize the image (Speed purpose)
         */
        Frame(Mat _img,  bool _key = false, int _width = TARGET_WIDTH, int _height = TARGET_HEIGHT);
        /**
         * @brief Change the reference for Homography matrix (Not yet implemented)
         * @param _H Homography matrix
         */
        void setHReference(Mat _H);
        /**
         * @brief Transform coordimates of keypoints by own homography matrix
         * @detail Usefull to detect keypoints in original image and thack them to the transformed one
         */
        void trackKeypoints();
        /**
         * @brief Check if the frame is too much distorted
         * @detail The distortion is besed on follow criteria:
         * - Ratio of semi-diagonals distance. \n
         * - Area. \n
         * - Mininum area covered by good keypoints. \n
         * @return true if the frame is good enought 
         * @return false otherwise
         */
        bool isGoodFrame();
        /**
         * @brief Calculate the minimun bounding area containing good keypoints 
         * @return float Area with good keypoints inside
         */
        float boundAreaKeypoints();
        /**
         * @brief Calculate the euclidean distance between two given vector in 2D
         * @param _pt1 First floating point OpenCV coordinate 
         * @param _pt2 Second floating point OpenCV coordinate 
         * @return float Distance betwenn two points
         */
        float getDistance(Point2f _pt1, Point2f _pt2);

};

/**
 * @brief 
 */
class SubMosaic{
    public:
        // ---------- Atributes
        int n_frames;                       //!< Number of frames in sub-mosaic
        vector<Frame*> frames;              //!< Vector containing all the frames (Pointers) in sub-mosaic 
        Frame* key_frame;                   //!< Pointer to reference frame in sub-mosaic
        Mat final_scene;                    //!< Image containing all blended images (the sub-mosaic)
        Mat avH;                            //!< Average Homography matrix (Matrix that reduces the dostortion error)
        Stitcher* stitcher;                 //!< Stitcher class
        struct Hierarchy{                   //!< Struct to relate two SubMosaics
            SubMosaic* mosaic;              //!< Pointer to SubMosaic
            float overlap;                  //!< Overlap area between this sub-mosaic and pointed one
        };                  
        vector<struct Hierarchy> neighbors; //!< Vector with all the neighbors SubMosaics (spatially close)

        // ---------- Methods
        /**
         * @brief Default constructor
         */
        SubMosaic() : n_frames(0){};
        /**
         * @brief Set the reference frame to the sub-mosaic. Create a Frame class with the input image
         * @param _scene OpenCV Matrix containig the BGR image
         */
        void setRerenceFrame(Mat _scene);
        /**
         * @brief Using the Stitcher class, add the object image to the current sub-mosaic
         * @param _object OpenCV Matrix containig the BGR image to add in the sub-mosaic
         * @return true If the stitch was sucesussfull
         * @return false If the stitch wasn't sucesussfull
         */
        bool add2Mosaic(Mat _object);
        /**
         * @brief Calculate the Homography matrix that reduce the distortion error in the sub-mosaic
         * (Not yet implemented)
         */
        void calcAverageH();
};

/**
 * @brief Save the homography matrix and heypoints in a txt file
 * @param H OpenCV Matrix containing Homography transformation
 * @param keypoints Vector with OpenCV Keypoints
 * @param matches Vector with OpenCV Matches
 */
void saveHomographyData(cv::Mat H, vector<KeyPoint> keypoints[2], vector<cv::DMatch> matches, std::string file);
}