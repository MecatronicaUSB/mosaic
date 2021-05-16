/**
 * @file frame.hpp
 * @brief Description of Frame class
 * @version 0.2
 * @date 10/02/2018
 * @author Victor Garcia
 */

#pragma once
#include "utils.h"

using namespace std;
using namespace cv;

namespace m2d //!< mosaic 2d namespace
{

/// temporal frame reference for Frame class
enum FrameRef
{
	PREV,
	NEXT
};
/// corner points referring to each transformation class
enum BoundPointsRef
{
	PERSPECTIVE,	//!< Points transformed by perspective transformation
	EUCLIDEAN,		//!< Points transformed by euclidean transformation
	RANSAC			//!< Points transformed by temporal perspective transformation (RANSAC algorithm)
};
/// offset of padding to add in scene after object transformation
enum WarpOffset
{
	TOP,
	BOTTOM,
	LEFT,
	RIGHT
};

/**
 * @brief All data for each image in a mosaic
 * @detail Contains All data of each image referred to current sub-mosaic or mosaic
 */
class Frame
{
  	public:
		// ---------- Attributes
		Mat descriptors;					    //!< Feature descriptors
		Mat color;                              //!< OpenCV Matrix containing the original image
		Mat gray;                               //!< OpenCV Matrix containing a gray scale version of image
		Mat H;                                  //!< Homography matrix based on previous frame
		Mat E;                                  //!< Euclidean matrix based on previous frame
		Rect2f bound_rect;                      //!< Minimum bounding rectangle of transformed image
		vector<vector<Point2f>> bound_points;   //!< Points of the transformed image (initially at corners)
		vector<vector<Point2f>> grid_points;	//!< Position (X,Y) of good key points after grid detector
		vector<vector<Point2f>> good_points;	//!< Position (X,Y) of good key points (inliers)
		vector<KeyPoint> keypoints;				//!< Feature OpenCV Key point object
		vector<Frame *> neighbors;				//!< Vector containing all spatially close Frames (Pointers)
		vector<Frame *> good_neighbors;			//!< Vector containing frames with almost one good match (Pointers)
		// ---------- Methods
		/**
		 * @brief Default class constructor
		 * @param _img OpenCV Matrix containing the BGR version of image
		 * @param _key Flag to assign this frame as reference (useful for SubMosaic Class)
		 * @param _width Width to resize the image (Speed purpose)
		 * @param _height Height to resize the image (Speed purpose)
		 */
		Frame(Mat _img, bool _pre = true, int _width = TARGET_WIDTH, int _height = TARGET_HEIGHT);
		/**
		 * @brief Object destructor
		 */
		~Frame();
		/**
		 * @brief Function to clone all object data
		 * @return Frame* New object containing same data
		 */
		Frame *clone();
		/**
		 * @brief Restore default values (except image data)
		 */
		void resetFrame();
		/**
		 * @brief Get the perspective transformation from current location to default position 
		 * (top left corner at 0,0)
		 * @return Mat Resulting perspective transformation
		 */
		Mat getResetTransform(int _ref = PERSPECTIVE);
		/**
		 * @brief Change the reference for Homography matrix
		 * @param _H Input transformation matrix 
		 * @param _ref Reference to which data transform (FrameRef enumeration options)
		 * @detail Modify all frame data:
		 * - transformation matrix
		 * - bounding points
		 * - keypoints position in image (referring to previous and next frame)
		 */
		void setHReference(Mat _H, int _ref = PERSPECTIVE);
		/**
		 * @brief Calculate the frame distortion
		 * @return float resulting distortion
		 * @detail Metrics:
		 * - Ratios of length of opposite sides \n
		 * - Ratios of length of consecutive sides \n
		 * - Angles of consecutive sides \n
		 * - Ratio of resulting and original area \n
		 */
		float frameDistortion(int _ref = PERSPECTIVE);
		/**
		 * @brief Check if the frame is too much distorted
		 * @return true if the frame is good enough
		 * @return false otherwise
		 * @detail The distortion is based on follow criteria:
		 * - Ratio of semi-diagonals distance. \n
		 * - Area. \n
		 * - Minimum area covered by good key points. (currently unused) \n
		 */
		bool isGoodFrame();
		/**
		 * @brief (Currently unused) Calculate the minimum bounding area containing good key points 
		 * @return float Area with good key points inside
		 */
		float boundAreaKeypoints();
		/**
		 * @brief Update neighbors considering only spatial information
		 * @detail Search over all neighbors of input frame, and stored it if the bounding
		 * boxes collides
		 */
		void updateNeighbors(Frame *_scene);
		/**
		 * @brief Check if bounding box of input frame intersect the self bounding rectangle
		 * @param _object Input frame
		 * @return true if Both frames collide
		 * @return false otherwise
		 */
		bool checkCollision(Frame *_object);
		/**
		 * @brief To check if the current frame have enough key points
		 * @return true if have more than 3 key points
		 * @return false if have less than 4 key points
		 */
		bool haveKeypoints();
};
}