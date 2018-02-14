/**
 * @file mosaic.hpp
 * @brief Mosaic2d Namespace and Classes
 * @version 0.2
 * @date 10/02/2018
 * @author Victor Garcia
 */
#ifndef SUB_MOSAIC_
#define SUB_MOSAIC_

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

//TODO-----
// class Mosaic{
//     public:
//         int n_frames;
//         int n_subs;
//         vector<SubMosaic*> sub_mosaic;
//         Stitcher* stitcher;

//         Mosaic();
//         SubMosaic* addSubMosaic(SubMosaic* _sub_mosaic);
// };

/**
 * @brief Save the homography matrix and heypoints in a txt file
 * @param H OpenCV Matrix containing Homography transformation
 * @param keypoints Vector with OpenCV Keypoints
 * @param matches Vector with OpenCV Matches
 */
void saveHomographyData(cv::Mat H, vector<KeyPoint> keypoints[2], vector<cv::DMatch> matches);
}

#endif
