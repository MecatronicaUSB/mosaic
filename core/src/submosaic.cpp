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

SubMosaic::SubMosaic()
{
	n_frames = 0;
	corrected = false;
	scene_size = Size2f(TARGET_WIDTH, TARGET_HEIGHT);
	avg_H = Mat::eye(3, 3, CV_64F);
	avg_E = Mat::eye(3, 3, CV_64F);
}

SubMosaic::~SubMosaic()
{
	for (Frame *frame : frames)
		delete frame;
	
	final_scene.release();
	avg_H.release();
	avg_E.release();
	neighbors.clear();
}

// See description in header file
SubMosaic *SubMosaic::clone()
{
	SubMosaic *new_sub_mosaic = new SubMosaic();
	new_sub_mosaic->final_scene = final_scene.clone();
	new_sub_mosaic->avg_H = avg_H.clone();
	new_sub_mosaic->avg_E = avg_E.clone();
	new_sub_mosaic->scene_size = scene_size;
	//new_sub_mosaic->neighbors = neighbors;
	for (Frame *frame : frames)
		new_sub_mosaic->addFrame(frame->clone());

	return new_sub_mosaic;
}

// See description in header file
void SubMosaic::addFrame(Frame *_frame)
{
	if (n_frames > 0)
		_frame->updateNeighbors(last_frame);
	frames.push_back(_frame);
	last_frame = _frame;
	n_frames++;
}

void SubMosaic::referenceToZero()
{
	vector<float> offset = {-frames[0]->bound_rect.y, 0, -frames[0]->bound_rect.x, 0};

	updateOffset(offset);
}

void SubMosaic::computeOffset()
{

	float top = TARGET_HEIGHT, bottom = 0, left = TARGET_WIDTH, right = 0;
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
	scene_size.width = right - left + 10;
	scene_size.height = bottom - top + 10;

	vector<float> offset(4);
	offset[TOP] = -top;
	offset[LEFT] = -left;

	updateOffset(offset);
}

void SubMosaic::updateOffset(vector<float> _total_offset)
{

	if (!_total_offset[TOP] && !_total_offset[LEFT])
		return;

	Mat T = Mat::eye(3, 3, CV_64F);
	T.at<double>(0, 2) = _total_offset[LEFT];
	T.at<double>(1, 2) = _total_offset[TOP];

	for (Frame *frame : frames)
	{
		frame->setHReference(T, PERSPECTIVE);
		frame->setHReference(T, EUCLIDEAN);
	}
	
	avg_H = T * avg_H;
	avg_E = T * avg_E;
}

// See description in header file
float SubMosaic::calcKeypointsError(Frame *_first, Frame *_second)
{
	float error = 0;
	for (int i = 0; i < _first->grid_points[PREV].size(); i++)
		error += getDistance(_first->grid_points[PREV][i], _second->grid_points[NEXT][i]);

	return error;
}

// See description in header file
float SubMosaic::calcDistortion()
{
	float tot_error = 0;

	for (Frame *frame : frames)
		tot_error = max(tot_error, frame->frameDistortion(RANSAC));
	
	return tot_error;
}

// See description in header file
void SubMosaic::correct()
{
	vector<vector<Point2f> > corner_points = getCornerPoints();
	Mat correct_H = getPerspectiveTransform(corner_points[PERSPECTIVE], corner_points[EUCLIDEAN]);

	for (Frame *frame : frames)
		frame->setHReference(correct_H);
	
	avg_H = correct_H * avg_H;
	avg_E = correct_H * avg_E;
}

vector<vector<Point2f> > SubMosaic::getCornerPoints()
{
	int pi1=0, pi2=0, pi3=0, pi4=0;
	int fi1=0, fi2=0, fi3=0, fi4=0;
	float top = TARGET_HEIGHT, bottom = 0, left = TARGET_WIDTH, right = 0;

	int frame_index=0, point_index;
	for (Frame *frame: frames)
	{
		point_index=0;
		for (Point2f point: frame->bound_points[EUCLIDEAN])
		{	
			if (point.y < top)
			{
				top = point.y;
				pi1 = point_index;
				fi1 = frame_index;
			}
			if (point.y > bottom)
			{
				bottom = point.y;
				pi2 = point_index;
				fi2 = frame_index;
			}
			if (point.x < left)
			{
				left = point.x;
				pi3 = point_index;
				fi3 = frame_index;
			}
			if (point.x > right)
			{
				right = point.x;
				pi4 = point_index;
				fi4 = frame_index;
			}
			point_index++;
		}
		frame_index++;
	}

	vector<vector<Point2f> > corner_points(2);

	vector<Point2f> euclidean_points;
	euclidean_points.push_back(frames[fi1]->bound_points[EUCLIDEAN][pi1]);
	euclidean_points.push_back(frames[fi2]->bound_points[EUCLIDEAN][pi2]);
	euclidean_points.push_back(frames[fi3]->bound_points[EUCLIDEAN][pi3]);
	euclidean_points.push_back(frames[fi4]->bound_points[EUCLIDEAN][pi4]);
	vector<Point2f> perspective_points;
	perspective_points.push_back(frames[fi1]->bound_points[PERSPECTIVE][pi1]);
	perspective_points.push_back(frames[fi2]->bound_points[PERSPECTIVE][pi2]);
	perspective_points.push_back(frames[fi3]->bound_points[PERSPECTIVE][pi3]);
	perspective_points.push_back(frames[fi4]->bound_points[PERSPECTIVE][pi4]);

	corner_points[EUCLIDEAN]= euclidean_points;
	corner_points[PERSPECTIVE]= perspective_points;

	return corner_points;
}

// See description in header file
Point2f SubMosaic::getCentroid()
{
	Point2f centroid(0, 0);

	int n_points=0;
	for (Frame *frame: frames) {
	    for (const Point2f point: frame->bound_points[PERSPECTIVE]) {
	        centroid += point;
	        n_points++;
	    }
	}
	centroid /= n_points;

	return centroid;
}

// See description in header file
bool SubMosaic::isEmpty()
{
	return n_frames == 0 ? true : false;
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
