/**
 * @file frame.cpp
 * @brief Implementation of Frame Class functions 
 * @version 0.2
 * @date 10/02/2018
 * @author Victor Garcia
 */

#include "../include/frame.hpp"

using namespace std;
using namespace cv;

namespace m2d //!< mosaic 2d namespace
{

// See description in header file
Frame::Frame(Mat _img, bool _pre, int _width, int _height)
{
	bound_points = vector<vector<Point2f>>(3);
	grid_points = vector<vector<Point2f>>(2);
	good_points = vector<vector<Point2f>>(2);
	// create camera and coefficients matrix

	// resize image to default size
	float ratio = (float) _width / (float)_img.cols;
	resize(_img, color, cv::Size(ratio * _img.cols, ratio * _img.rows), 0, 0, CV_INTER_LINEAR);

	// resize(_img, color, Size(_width, _height));
	// create a gray scale version of image
	cvtColor(color, gray, CV_BGR2GRAY);
	// apply scb to gray image
	if (_pre)
		imgChannelStretch(gray, gray, 1, 99, Mat());
	// default bounding rectangle
	bound_rect = Rect2f(0, 0, (float)_width, (float)_height);
	// corner points
	bound_points[PERSPECTIVE].push_back(Point2f(0, 0));
	bound_points[PERSPECTIVE].push_back(Point2f(_width, 0));
	bound_points[PERSPECTIVE].push_back(Point2f(_width, _height));
	bound_points[PERSPECTIVE].push_back(Point2f(0, _height));
	// center point
	bound_points[PERSPECTIVE].push_back(Point2f(_width / 2, _height / 2));
	// points for euclidean model
	bound_points[EUCLIDEAN] = bound_points[PERSPECTIVE];
	// default transformation matrices
	H = Mat::eye(3, 3, CV_64F);
	E = Mat::eye(3, 3, CV_64F);
}

// See description in header file
Frame::~Frame()
{
	// delete all data
	H.release();
	E.release();
	gray.release();
	color.release();
	descriptors.release();
	grid_points.clear();
	good_points.clear();
	bound_points.clear();
	keypoints.clear();
	neighbors.clear();
	good_neighbors.clear();
}

// See description in header file
Frame *Frame::clone()
{
	Frame *new_frame = new Frame(color, true);

	new_frame->descriptors = descriptors.clone();
	new_frame->H = H.clone();
	new_frame->E = E.clone();
	new_frame->bound_rect = bound_rect;
	new_frame->bound_points = bound_points;
	new_frame->grid_points = grid_points;
	new_frame->good_points = good_points;	
	new_frame->keypoints = keypoints;
	new_frame->neighbors = neighbors;
	new_frame->good_neighbors = good_neighbors;

	return new_frame;
}

// See description in header file
void Frame::resetFrame()
{
	// restore default values
	H = Mat::eye(3, 3, CV_64F);
	E = Mat::eye(3, 3, CV_64F);

	bound_points[PERSPECTIVE][0] = Point2f(0, 0);
	bound_points[PERSPECTIVE][1] = Point2f(color.cols, 0);
	bound_points[PERSPECTIVE][2] = Point2f(color.cols, color.rows);
	bound_points[PERSPECTIVE][3] = Point2f(0, color.rows);
	bound_points[PERSPECTIVE][4] = Point2f(color.cols / 2, color.rows / 2);
	bound_points[EUCLIDEAN] = bound_points[PERSPECTIVE]; 
	bound_rect = Rect2f(0, 0, (float)color.cols, (float)color.rows);

	grid_points[NEXT].clear();
	good_points[NEXT].clear();
	neighbors.clear();
	good_neighbors.clear();
}

// See description in header file
Mat Frame::getResetTransform(int _ref)
{
	// current points
	vector<Point2f> curr_points = {
		bound_points[_ref][0],
		bound_points[_ref][1],
		bound_points[_ref][2],
		bound_points[_ref][3]
	};
	// default points (top-left corner at 0,0)
	vector<Point2f> org_points = {
		Point2f(0, 0),
		Point2f(TARGET_WIDTH, 0),
		Point2f(TARGET_WIDTH, TARGET_HEIGHT),
		Point2f(0, TARGET_HEIGHT),
	};
	// return transformation
	return getPerspectiveTransform(curr_points, org_points);
}

// See description in header file
float Frame::frameDistortion(int _ref)
{
	Point2f line[4][2];
	Point2f v1, v2;
	float side[4], cosine[4];
	float frame_dims = (float)TARGET_WIDTH * (float)TARGET_HEIGHT;
	float ratio_dims = (float)TARGET_HEIGHT / (float)TARGET_WIDTH;
	float o_sides_error = 0, c_sides_error = 0, angle_error = 0;
	float area_error = 0, min_ratio = 0;
	// get the lines of each edge
	for (int i = 0; i < 4; i++)
	{
		line[i][0] = bound_points[_ref][i];
		line[i][1] = bound_points[_ref][i < 3 ? i + 1 : 0];
		// get the length of each edge
		side[i] = getDistance(line[i][0], line[i][1]);
	}
	// calculate the cosine of each internal angle
	for (int i = 0; i < 4; i++)
	{
		v1.x = abs(line[i][1].x - line[i][0].x);
		v1.y = abs(line[i][1].y - line[i][0].y);
		v2.x = abs(line[i < 3 ? i + 1 : 0][1].x - line[i < 3 ? i + 1 : 0][0].x);
		v2.y = abs(line[i < 3 ? i + 1 : 0][1].y - line[i < 3 ? i + 1 : 0][0].y);

		cosine[i] = (v1.x * v2.x + v1.y * v2.y) / (sqrt(v1.x * v1.x + v1.y * v1.y) * sqrt(v2.x * v2.x + v2.y * v2.y));
	}
	// opposite sides ratio
	o_sides_error = 2 - 0.5 * (min(side[0] / side[2], side[2] / side[0]) +
								min(side[1] / side[3], side[3] / side[1]));

	min_ratio = min(side[0] / side[1], min(side[1] / side[2], min(side[2] / side[3], side[3] / side[0])));
	// consecutive sides ratio
	c_sides_error = 1 - min(min_ratio / ratio_dims, ratio_dims / min_ratio);
	// corner points
	vector<Point2f> aux_points = {
		bound_points[_ref][0],
		bound_points[_ref][1],
		bound_points[_ref][2],
		bound_points[_ref][3]
	};
	// ratio between current and default area
	area_error = 1 - min(contourArea(aux_points) / frame_dims,
												frame_dims / contourArea(aux_points));
	// angle error, considering max cosine of internal angles
	angle_error = pow(max(cosine[0], max(cosine[1], max(cosine[2], cosine[3]))), 5);

	return o_sides_error + c_sides_error + area_error + angle_error;
}

// See description in header file
bool Frame::isGoodFrame()
{
	float deformation, area, keypoints_area;
	float semi_diag[4], ratio[2];
	float diagonal_error=0, area_error=0;
	// calculate each semi diagonal length
	for (int i = 0; i < 4; i++)
	{
		// 5th point correspond to center of image
		// Getting the distance between corner points to the center (all semi diagonal distances)
		semi_diag[i] = getDistance(bound_points[PERSPECTIVE][i], bound_points[PERSPECTIVE][4]);
	}
	// ratio between semi diagonals
	ratio[0] = max(semi_diag[0] / semi_diag[2], semi_diag[2] / semi_diag[0]);
	ratio[1] = max(semi_diag[1] / semi_diag[3], semi_diag[3] / semi_diag[1]);
	// get the max semi diagonal ratio
	diagonal_error = max(ratio[0], ratio[1]);
	// corner points of current frame
	vector<Point2f> aux_points = {
		bound_points[PERSPECTIVE][0],
		bound_points[PERSPECTIVE][1],
		bound_points[PERSPECTIVE][2],
		bound_points[PERSPECTIVE][3]
	};
	// Area of distorted image
	area = contourArea(aux_points);
	// ratio between current and default area
	area_error = max(area/(color.cols*color.rows), (color.cols*color.rows)/area);

	// enclosing area with good key points (currently unused)
	// keypoints_area = boundAreaKeypoints();

	// 1.3 initial threshold value, must be adjusted in future tests
	if (area_error > 1.15) // 1.15
		return false;
	// 1.3 initial threshold value, must be adjusted in future tests
	if (diagonal_error > 1.5)
		return false;
	// if (keypoints_area < 0.2 * color.cols * color.rows)
	// 	return false;

	return true;
}

// See description in header file
float Frame::boundAreaKeypoints()
{
	vector<Point2f> hull;
	// area covered by key points
	convexHull(grid_points[PREV], hull);

	return contourArea(hull);
}

// See description in header file
void Frame::setHReference(Mat _H, int _ref)
{
	// modify corner points by input transformation
	perspectiveTransform(bound_points[_ref], bound_points[_ref], _H);
	// for perspective transform, modify all frame points
	if (_ref == PERSPECTIVE)
	{
		if (grid_points[PREV].size())
			perspectiveTransform(grid_points[PREV], grid_points[PREV], _H);
		if (grid_points[NEXT].size())
			perspectiveTransform(grid_points[NEXT], grid_points[NEXT], _H);

		// updateBoundRect();
		bound_rect = boundingRectFloat(bound_points[PERSPECTIVE]);
		H = _H * H;
	}
	// for euclidean model, modify only transformation matrix
	else
		E = _H * E;
}

// See description in header file
bool Frame::checkCollision(Frame *_scene)
{
	// check horizontal collision
	if (_scene->bound_rect.x > bound_rect.x + bound_rect.width)
		return false;
    // check vertical collision
    if (_scene->bound_rect.x + _scene->bound_rect.width < bound_rect.x)
		return false;
	// check horizontal collision
	if (_scene->bound_rect.y > bound_rect.y + bound_rect.height)
		return false;
	// check vertical collision
    if (_scene->bound_rect.y + _scene->bound_rect.height < bound_rect.y)
        return false;

	return true;
}

// See description in header file
void Frame::updateNeighbors(Frame *_scene)
{
	// for spatial neighbors
	if (checkCollision(_scene));
		neighbors.push_back(_scene);
	// check for neighbors of neighbor
	for (Frame *neighbor: _scene->neighbors)
		if (checkCollision(neighbor))
			neighbors.push_back(neighbor);
}

// See description in header file
bool Frame::haveKeypoints()
{
	// minimum number of key points
	return keypoints.size() >= 4 ? true : false;
}
}
