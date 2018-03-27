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

class Stitcher
{
  public:
    // ---------- Atributes
    bool use_grid;                              //!< flag to use or not the grid detection
    int cells_div;                              //!< number (n) of cell divisions in grid detector (if used)
    vector<vector<vector<DMatch>>> matches;     //!< Vector of OpenCV Matches
    vector<vector<DMatch>> good_matches;        //!< Vector of OpenCV good Matches (after discard outliers)
    vector<Frame *> img = vector<Frame *>(2);   //!< Vector of two frames to stitch (Pointers)
    // ---------- Methods
    /**
     * @brief Default Stitcher constructor
     * @param _grid flag to use grid detection (default true)
     * @param _detector enum value to set the desired feature Detector and descriptor
     * @param _matcher enum value to set the desired feature matcher
     */
    Stitcher(bool _grid = false, int _detector = USE_KAZE, int _matcher = USE_BRUTE_FORCE);
    /**
     * @brief Calculate the transformation matrix to map _object frame into _scene
     * @param _object frame to be map
     * @param _scene objetive frame
     * @return vector<Mat> Perspective and euclidean transformations
     */
    vector<Mat> stitch(Frame *_object, Frame *_scene);
    /**
     * @brief detec and save feature keypoints and descriptors
     * @param _frames 
     */
    void detectFeatures(vector<Frame *> &_frames);

  private:
    // ---------- Atributes
    Ptr<Feature2D> detector;                    //!< Pointer to OpenCV feature extractor
    Ptr<DescriptorMatcher> matcher;             //!< Pointer to OpenCV feature Matcher
    vector<vector<Point2f>> neighbors_kp;       //!< Position of neighbors keypoints
    vector<Point2f> euclidean_points;           //!< points tracked by euclidean transformation
    vector<Point2f> scene_points;               //!< points tracked by perspective transformation
    vector<Point2f> object_points;              //!< points of object frame
    // ---------- Methods
    /**
     * @brief Discard outliers from initial matches vector
     * @param _thresh input threshold
     */
    void getGoodMatches(float _thresh = 0.8);
    /**
     * @brief Select the best keypoint for each cell in the defined grid
     */
    void gridDetector();
    /**
     * @brief transform a vector of OpenCV Keypoints to vectors of OpenCV Points
     */
    void positionFromKeypoints();
    /**
     * @brief track the scene and it's neighbors points by correspond transformation matrix
     * @return vector<Point2f> points tracked by euclidean transformation
     */
    void trackKeypoints();
    /**
     * @brief Correct homography transformation acording to the best euclidean
     * @param _H perspective transformation
     * @param _E best euclidean transformation
     */
    void correctHomography(Mat &_H, Mat _E);
    /**
     * @brief clean all used data
     */
    void cleanData();
};
}
