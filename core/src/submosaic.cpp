/**
 * @file submosaic.cpp
 * @brief Implementation of SubMosaic class functions 
 * @version 0.2
 * @date 10/02/2018
 * @author Victor Garcia
 */

#include "../include/submosaic.hpp"

using namespace std;
using namespace cv;

namespace m2d //!< mosaic 2d namespace
{

// See description in header file
SubMosaic::SubMosaic()
{
	n_frames = 0;
	scene_size = Size2f(TARGET_WIDTH, TARGET_HEIGHT);
	next_H = Mat::eye(3, 3, CV_64F);
	next_E = Mat::eye(3, 3, CV_64F);
}

// See description in header file
SubMosaic::~SubMosaic()
{
	// delete each frame
	for (Frame *frame : frames)
		delete frame;
	// delete rest of data
	final_scene.release();
	next_H.release();
	next_E.release();
	neighbors.clear();
}

// See description in header file
SubMosaic *SubMosaic::clone()
{
	SubMosaic *new_sub_mosaic = new SubMosaic();
	new_sub_mosaic->final_scene = final_scene.clone();
	new_sub_mosaic->next_H = next_H.clone();
	new_sub_mosaic->next_E = next_E.clone();
	new_sub_mosaic->scene_size = scene_size;
	new_sub_mosaic->neighbors = neighbors;
	for (Frame *frame : frames)
		new_sub_mosaic->addFrame(frame->clone());

	return new_sub_mosaic;
}

// See description in header file
void SubMosaic::addFrame(Frame *_frame)
{
	// update neighbors of input frame
	if (n_frames > 0)
		_frame->updateNeighbors(last_frame);
	// add new frame
	frames.push_back(_frame);
	last_frame = _frame;
	n_frames++;
}

// See description in header file
void SubMosaic::referenceToZero()
{
	// calculate the offset by the location of the bounding rect
	vector<float> offset = {-frames[0]->bound_rect.y, 0, -frames[0]->bound_rect.x, 0};
	// translate all sub mosaic
	updateOffset(offset);
}

// See description in header file
void SubMosaic::computeOffset()
{
	float top = TARGET_HEIGHT, bottom = 0, left = TARGET_WIDTH, right = 0;
	// find the border coordinated of sub mosaic
	for (Frame *frame : frames)
	{
		for (Point2f point : frame->bound_points[PERSPECTIVE])
		{
			if (point.x < left)
				left = point.x;
			if (point.y < top)
				top = point.y;
			if (point.x > right)
				right = point.x;
			if (point.y > bottom)
				bottom = point.y;
		}
	}
	// update the final image size
	scene_size.width = right - left + 10;
	scene_size.height = bottom - top + 10;

	vector<float> offset(4);
	offset[TOP] = -top;
	offset[LEFT] = -left;
	// translate the mosaic based on prevoious offset
	updateOffset(offset);
}

// See description in header file
void SubMosaic::updateOffset(vector<float> _total_offset)
{
	// if is in positive side, do nothing.
	// TODO: translate to 0,0 -> corner top-left
	if (!_total_offset[TOP] && !_total_offset[LEFT])
		return;
	// create translation matrix
	Mat T = Mat::eye(3, 3, CV_64F);
	T.at<double>(0, 2) = _total_offset[LEFT];
	T.at<double>(1, 2) = _total_offset[TOP];
	// apply the transformation to each frame
	for (Frame *frame : frames)
	{
		frame->setHReference(T, PERSPECTIVE);
		frame->setHReference(T, EUCLIDEAN);
	}
	// update transfomrationt to last frame of mosaic
	next_H = T * next_H;
	next_E = T * next_E;
}

// See description in header file
float SubMosaic::calcKeypointsError(Frame *_first, Frame *_second)
{
	float error = 0;
	// cumulative distance between matches
	int i;
	for (i = 0; i < _first->grid_points[PREV].size(); i++)
		error += getDistance(_first->grid_points[PREV][i], _second->grid_points[NEXT][i]);
	// return the average error
	return error/i;
}

// See description in header file
float SubMosaic::calcDistortion(int _ref)
{
	float tot_error = 0;
	// distortion of worst frame
	// TODO: try cumulative distortion.
	for (Frame *frame : frames)
		tot_error = max(tot_error, frame->frameDistortion(RANSAC));
	
	return tot_error;
}

// See description in header file
void SubMosaic::correct()
{
	// get corner ponints of mosaic, based on first and last frame
	vector<vector<Point2f> > corner_points = getCornerPoints();
	// calculate perspective transform from perspective points to eucldiean ones
	Mat correct_H = getPerspectiveTransform(corner_points[PERSPECTIVE], corner_points[EUCLIDEAN]);
	// update each frame of mosaic
	for (Frame *frame : frames)
		frame->setHReference(correct_H);
	// update transfomrationt to last frame of mosaic
	next_H = correct_H * next_H;
	next_E = correct_H * next_E;
}

// See description in header file
vector<vector<Point2f> > SubMosaic::getCornerPoints()
{
	vector<Frame *> corner_frames;
	//save first and last frame
	corner_frames.push_back(frames[0]);
	corner_frames.push_back(frames[n_frames-1]);

	vector<vector<CornerPoint>> corner_points(2);
	for (int i=0; i<corner_frames.size(); i++)
	{
		// loop over bounding points
		for (int j=0; j<4; j++)
		{
			corner_points[i].push_back(CornerPoint(corner_frames[i]->bound_points[EUCLIDEAN][j], j));
			// get distance from current point to center point of opposite frame
			corner_points[i][j].distance = getDistance(corner_points[i][j].point,
													   corner_frames[!i]->bound_points[EUCLIDEAN][4]);
		}
		// sort by distance
		sort(corner_points[i].begin(), corner_points[i].end());
	}
	vector<vector<Point2f> > border_points(2);
	// save points with bigger distance for each frame
	vector<Point2f> euclidean_points;
	euclidean_points.push_back(corner_points[0][2].point);
	euclidean_points.push_back(corner_points[0][3].point);
	euclidean_points.push_back(corner_points[1][2].point);
	euclidean_points.push_back(corner_points[1][3].point);
	// save points with bigger distance for each frame, using the point index
	vector<Point2f> perspective_points;
	perspective_points.push_back(corner_frames[0]->bound_points[PERSPECTIVE][corner_points[0][2].index]);
	perspective_points.push_back(corner_frames[0]->bound_points[PERSPECTIVE][corner_points[0][3].index]);
	perspective_points.push_back(corner_frames[1]->bound_points[PERSPECTIVE][corner_points[1][2].index]);
	perspective_points.push_back(corner_frames[1]->bound_points[PERSPECTIVE][corner_points[1][3].index]);

	border_points[EUCLIDEAN]= euclidean_points;
	border_points[PERSPECTIVE]= perspective_points;

	return border_points;
}

// See description in header file
Point2f SubMosaic::getCentroid()
{
	Point2f centroid(0, 0);
	int n_points=0;
	// loop over all keypoints positon
	for (Frame *frame: frames) {
	    for (const Point2f point: frame->bound_points[PERSPECTIVE]) {
	        centroid += point;
	        n_points++;
	    }
	}
	// return the average position
	return centroid /= n_points;
}

// See description in header file
Mat SubMosaic::buildMap(int _type, Scalar _color)
{
	// positioning sub mosaic in a positive location
	this->computeOffset();
	// declare map image
	Mat map = Mat(scene_size, CV_8UC3, Scalar(255, 255, 255));
	// check type of map
	if (_type == RECTANGLE)
	{
		// print form of each frame
		vector<vector<Point>> map_points;
		vector<Point> aux_points(4);
		for (Frame *frame : frames)
		{
			aux_points[0] = Point(frame->bound_points[PERSPECTIVE][0].x, 
								  frame->bound_points[PERSPECTIVE][0].y);
			aux_points[1] = Point(frame->bound_points[PERSPECTIVE][1].x, 
								  frame->bound_points[PERSPECTIVE][1].y);
			aux_points[2] = Point(frame->bound_points[PERSPECTIVE][2].x, 
								  frame->bound_points[PERSPECTIVE][2].y);
			aux_points[3] = Point(frame->bound_points[PERSPECTIVE][3].x, 
								  frame->bound_points[PERSPECTIVE][3].y);
			map_points.push_back(aux_points);
		}
		polylines(map, map_points, true, _color, 1);
	}
	else
	{
		// print circles at center of each frame, and links to each neighbor
		Scalar line_color = Scalar(0, 0, 255);
		Point center_point, neighbor_point;
		// first print the gray links (spatial neighbors)
		for (Frame *frame : frames)
		{
			center_point = Point(frame->bound_points[PERSPECTIVE][4].x,
								 frame->bound_points[PERSPECTIVE][4].y);
			for (Frame *neighbor : frame->neighbors)
			{
				neighbor_point = Point(neighbor->bound_points[PERSPECTIVE][4].x,
								 	   neighbor->bound_points[PERSPECTIVE][4].y);
				line(map, center_point, neighbor_point, Scalar(190, 190, 190), 1);
			}
		}
		// second print the colored links (good neighbors)		
		for (Frame *frame : frames)
		{
			center_point = Point(frame->bound_points[PERSPECTIVE][4].x,
								 frame->bound_points[PERSPECTIVE][4].y);
			for (Frame *neighbor : frame->good_neighbors)
			{
				neighbor_point = Point(neighbor->bound_points[PERSPECTIVE][4].x,
								 	   neighbor->bound_points[PERSPECTIVE][4].y);
				line(map, center_point, neighbor_point, line_color, 1);
			}
		}
		// third print a circle at the center of each frame
		for (Frame *frame : frames)
		{
			center_point = Point(frame->bound_points[PERSPECTIVE][4].x,
								 frame->bound_points[PERSPECTIVE][4].y);
			circle(map, center_point, 8, _color, -1);
		}
	}
	return map;
}


// See description in header file
void saveHomographyData(Mat _h, vector<KeyPoint> keypoints[2], std::vector<DMatch> matches)
{
	ofstream file;
	file.open("homography-data.txt");
	for (int i = 0; i < _h.cols; i++)
	{
		for (int j = 0; j < _h.rows; j++)
		{
			file << _h.at<double>(i, j);
			file << " ";
		}
		file << "\n";
	}
	file << matches.size() << "\n";
	for (auto m : matches)
	{
		file << keypoints[0][m.queryIdx].pt.x << " ";
		file << keypoints[0][m.queryIdx].pt.y << "\n";
	}
	for (auto m : matches)
	{
		file << keypoints[1][m.trainIdx].pt.x << " ";
		file << keypoints[1][m.trainIdx].pt.y << "\n";
	}
	file.close();
}
}
