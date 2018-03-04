/**
 * @file stitch.hpp
 * @brief Description of Stitcher Class
 * @version 0.2
 * @date 10/02/2018
 * @author Victor Garcia
 */

#pragma once
#include "frame.hpp"

using namespace std;
using namespace cv;

namespace m2d //!< mosaic 2d namespace
{

/// Reference image for stitching class
enum ImgRef
{
    SCENE,
    OBJECT
};
/// Keypoint detector and descriptor to use
enum Detector
{
    USE_KAZE,
    USE_AKAZE,
    USE_SIFT,
    USE_SURF
};
/// Keypoint matcher to use in stitcher class
enum Matcher
{
    USE_BRUTE_FORCE,
    USE_FLANN
};

enum StitchStatus
{
    OK,
    BAD_DISTORTION,
    NO_KEYPOINTS,
    NO_HOMOGRAPHY
};

class Stitcher
{
  public:
    // ---------- Atributes
    bool use_grid; //!< flag to use or not the grid detection
    int cells_div; //!< number (n) of cell divisions in grid detector (if used)
    Mat offset_h;
    vector<vector<vector<DMatch>>> matches;   //!< Vector of OpenCV Matches
    vector<vector<DMatch>> good_matches;      //!< Vector of OpenCV good Matches (after discard outliers)
    vector<Frame *> img = vector<Frame *>(2); //!< Vector of two frames to stitch (Pointers)
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
    Stitcher(bool _grid = false, int _detector = USE_KAZE, int _matcher = USE_BRUTE_FORCE);
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
    int stitch(Frame *_object, Frame *_scene, Size _scene_dims);

  private:
    // ---------- Atributes
    Ptr<Feature2D> detector;                  //!< Pointer to OpenCV feature extractor
    Ptr<DescriptorMatcher> matcher;           //!< Pointer to OpenCV feature Matcher
    vector<Mat> descriptors = vector<Mat>(2); //!< Array of OpenCV Matrix conaining feature descriptors
    vector<vector<Point2f>> neighbors_kp;
    vector<Mat> points_pos = vector<Mat>(2);
    // ---------- Methods
    /**
         * @brief Discard outliers from initial matches vector
         */
    void getGoodMatches(float _thresh = 0.8);
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
         * @param _H 
         * @param _points 
         */
    void trackKeypoints();
    /**
         * @brief 
         */
    void drawKeipoints(vector<float> _warp_offset, Mat &_final_scene);
    /**
         * @brief 
         */
    void cleanNeighborsData();
    /**
         * @brief Computes the size of pads in the scene based on the transformation of object image
         * @param _H Homography matrix
         * @param _scene_dims Dimensions of scene image
         * @return vector<float> Padd size for each side of scene image
         */
    void getBoundPoints();
    /**
         * @brief 
         */
    void updateNeighbors();
};
}
