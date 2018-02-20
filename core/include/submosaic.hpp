/**
 * @file submosaic.hpp
 * @brief Implementation of Frame, SubMosaic and Mosaic classes
 * @version 0.2
 * @date 10/02/2018
 * @author Victor Garcia
 */
#pragma once
#include "utils.h"
#include "stitch.hpp"
#include "blend.hpp"

using namespace std;
using namespace cv;

namespace m2d //!< mosaic 2d namespace
{

/// termporal frame reference for Frame class
enum FrameRef{
    PREV,
    NEXT
};

/**
 * @brief Calculate the euclidean distance between two given vector in 2D
 * @param _pt1 First floating point OpenCV coordinate 
 * @param _pt2 Second floating point OpenCV coordinate 
 * @return float Distance betwenn two points
 */
float getDistance(Point2f _pt1, Point2f _pt2);

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
        Rect2f bound_rect2;                //!< Minimum bounding rectangle of transformed image
        Mat color;                        //!< OpenCV Matrix containing the original image
        Mat gray;                         //!< OpenCV Matrix containing a gray scale version of image
        vector<Point2f> bound_points;     //!< Points of the transformmed image (initially at corners)
        vector<Point2f> bound_points2;
        vector<Point2f> keypoints_pos[2]; //!< Position (X,Y) of good keypoints in image 
        vector<Frame *> neighbors;         //!< Vector containing all spatially close Frames (Pointers)
        vector<KeyPoint> keypoints;
        Mat descriptors;
        float frame_error;

        // ---------- Methods
        /**
         * @brief Default class constructor
         * @param _img OpenCV Matrix containing the BGR version of image
         * @param _key Flag to assign this frame as reference (usefull for SubMosaic Class)
         * @param _width Width to resize the image (Speed purpose)
         * @param _height Height to resize the image (Speed purpose)
         */
        Frame(Mat _img,  bool _pre = true, int _width = TARGET_WIDTH, int _height = TARGET_HEIGHT);
        /**
         * @brief 
         */
        void resetFrame();
        /**
         * @brief Change the reference for Homography matrix (Not yet implemented)
         * @param _H Homography matrix
         */
        void setHReference(Mat _h);
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

};
typedef struct {                   //!< Struct to relate two SubMosaics
    SubMosaic* mosaic;              //!< Pointer to SubMosaic
    float overlap;                  //!< Overlap area between this sub-mosaic and pointed one
} Hierarchy; 
/**
 * @brief 
 */
class SubMosaic{
    public:
        // ---------- Atributes
        int n_frames;                       //!< Number of frames in sub-mosaic
        vector<Frame *> frames;             //!< Vector containing all the frames (Pointers) in sub-mosaic 
        Frame *key_frame;                  //!< Pointer to reference frame in sub-mosaic
        Frame *last_frame;
        Mat final_scene;                    //!< Image containing all blended images (the sub-mosaic)
        Mat avg_H;                            //!< Average Homography matrix (Matrix that reduces the dostortion error)
                 
        vector<Hierarchy> neighbors; //!< Vector with all the neighbors SubMosaics (spatially close)
        float distortion;
        Size2f scene_size;
        bool is_complete;
        // ---------- Methods
        /**
         * @brief Default constructor
         */
        SubMosaic() : n_frames(0), scene_size(Size2f(TARGET_WIDTH, TARGET_HEIGHT)), is_complete(false){};
        /**
         * @brief Using the Stitcher class, add the object image to the current sub-mosaic
         * @param _object OpenCV Matrix containig the BGR image to add in the sub-mosaic
         * @return true If the stitch was sucesussfull
         * @return false If the stitch wasn't sucesussfull
         */
        void addFrame(Frame *_frame);
        /**
         * @brief 
         * @param _object 
         * @param _scene 
         * @return float 
         */
        float calcKeypointsError(Frame *_first, Frame *_second);
        /**
         * @brief 
         * @return float 
         */
        float calcDistortion();
        /**
         * @brief 
         */
        void computeOffset();
        /**
         * @brief 
         * @param _frames 
         */
        void updateOffset(vector<float> _offset);
        /**
         * @brief 
         */
        void correct();
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
void saveHomographyData(cv::Mat _h, vector<KeyPoint> keypoints[2], vector<cv::DMatch> matches);
}