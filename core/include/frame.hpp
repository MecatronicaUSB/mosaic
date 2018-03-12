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

const int KEY = 0;

/// termporal frame reference for Frame class
enum FrameRef
{
	PREV,
	NEXT
};

enum RansacReference
{
	FIRST,
	RANSAC,
	SECOND
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
		float frame_error;
		Mat descriptors;
		Mat color;                             //!< OpenCV Matrix containing the original image
		Mat gray;                              //!< OpenCV Matrix containing a gray scale version of image
		Mat H;                                 //!< Homography matrix based on previous frame
		Rect2f bound_rect;                     //!< Minimum bounding rectangle of transformed image
		vector<vector<Point2f>> bound_points;  //!< Points of the transformmed image (initially at corners)
		vector<vector<Point2f>> grid_points; //!< Position (X,Y) of good keypoints in image
		vector<vector<Point2f>> good_points; //!< Position (X,Y) of good keypoints in image
		vector<KeyPoint> keypoints;
		vector<Frame *> neighbors; //!< Vector containing all spatially close Frames (Pointers)

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
		 * @brief 
		 */
		~Frame();
		/**
		 * @brief 
		 * @return Frame* 
		 */
		Frame *clone();
		/**
		 * @brief 
		 */
		void resetFrame();
		/**
		 * @brief Change the reference for Homography matrix (Not yet implemented)
		 * @param _H Homography matrix
		 */
		void setHReference(Mat _H);
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
		 * @brief 
		 */
		void enhance();
		/**
		 * @brief Calculate the minimun bounding area containing good keypoints 
		 * @return float Area with good keypoints inside
		 */
		float boundAreaKeypoints();
		/**
		 * @brief 
		 */
		void updateBoundRect();
		/**
		 * @brief 
		 * @return true 
		 * @return false 
		 */
		bool haveKeypoints();
};
}