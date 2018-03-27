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

/// termporal frame reference for Frame class
enum FrameRef
{
	PREV,
	NEXT
};
/// corner ponints refereing to each transformation class
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
 * @detail Contains All data of each image refered to current sub-mosaic or mosaic
 */
class Frame
{
  	public:
		// ---------- Atributes
		Mat descriptors;					    //!< Feature descriptors
		Mat color;                              //!< OpenCV Matrix containing the original image
		Mat gray;                               //!< OpenCV Matrix containing a gray scale version of image
		Mat H;                                  //!< Homography matrix based on previous frame
		Mat E;                                  //!< Euclidean matrix based on previous frame
		Rect2f bound_rect;                      //!< Minimum bounding rectangle of transformed image
		vector<vector<Point2f>> bound_points;   //!< Points of the transformmed image (initially at corners)
		vector<vector<Point2f>> grid_points;	//!< Position (X,Y) of good keypoints after grid detector
		vector<vector<Point2f>> good_points;	//!< Position (X,Y) of good keypoints (inliers)
		vector<KeyPoint> keypoints;				//!< Feature OpenCV Keypoint object
		vector<Frame *> neighbors;				//!< Vector containing all spatially close Frames (Pointers)
		vector<Frame *> good_neighbors;			//!< Vector containing frames with almost one good match (Pointers)
		// ---------- Methods
		/**
		 * @brief Default class constructor
		 * @param _img OpenCV Matrix containing the BGR version of image
		 * @param _key Flag to assign this frame as reference (usefull for SubMosaic Class)
		 * @param _width Width to resize the image (Speed purpose)
		 * @param _height Height to resize the image (Speed purpose)
		 */
		Frame(Mat _img, bool _pre = true, int _width = TARGET_WIDTH, int _height = TARGET_HEIGHT);
		/**
		 * @brief Object desctructor
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
		 * @param _ref Reference to wich data transform (FrameRef enum options)
		 * @detail Modify all frame data:
		 * - transformation matrix
		 * - bounding points
		 * - keypoints position in image (refering to previous and next frame)
		 */
		void setHReference(Mat _H, int _ref = PERSPECTIVE);
		/**
		 * @brief Calculate the frame distiortion
		 * @return float resulting distortion
		 * @detail Metrics:
		 * - Ratios of lenht of oposite sides \n
		 * - Ratios of lenht of consecutive sides \n
		 * - Angles of consecutive sides \n
		 * - Ratio of resulting and original area \n
		 */
		float frameDistortion(int _ref = PERSPECTIVE);
		/**
		 * @brief Check if the frame is too much distorted
		 * @return true if the frame is good enought 
		 * @return false otherwise
		 * @detail The distortion is besed on follow criteria:
		 * - Ratio of semi-diagonals distance. \n
		 * - Area. \n
		 * - Mininum area covered by good keypoints. (currently unused) \n
		 */
		bool isGoodFrame();
		/**
		 * @brief (Currently unused) Calculate the minimun bounding area containing good keypoints 
		 * @return float Area with good keypoints inside
		 */
		float boundAreaKeypoints();
		/**
		 * @brief Update neighbors considering only spatial information
		 * @detail Search over all neighbors of input frame, and stored it if the bounding
		 * boxes collides
		 */
		void updateNeighbors(Frame *_scene);
		/**
		 * @brief Check if bounding box of input frame instersect the self bounding rect
		 * @param _object Input frame
		 * @return true if Both frames collide
		 * @return false otherwise
		 */
		bool checkCollision(Frame *_object);
		/**
		 * @brief To check if the current frame have enought keypoints
		 * @return true if have more than 3 keypoints
		 * @return false if have less than 4 keypoints
		 */
		bool haveKeypoints();
};
}