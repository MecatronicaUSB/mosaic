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
/// termporal frame reference for Frame class
enum FrameRef{
    PREV,
    NEXT
};

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

class Stitcher {
    public:
        // ---------- Atributes
        Mat offset_H;
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
        bool stitch(Frame *_object, Frame *_scene, Mat& _final_scene);
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

#endif