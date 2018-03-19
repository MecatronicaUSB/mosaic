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
	scene_size = Size2f(TARGET_WIDTH, TARGET_HEIGHT);
	avg_H = Mat::eye(3, 3, CV_64F);
}

SubMosaic::~SubMosaic()
{
	for (Frame *frame : frames)
		delete frame;
	
	final_scene.release();
	avg_H.release();
	neighbors.clear();
}

// See description in header file
SubMosaic *SubMosaic::clone()
{
	SubMosaic *new_sub_mosaic = new SubMosaic();
	new_sub_mosaic->final_scene = final_scene.clone();
	new_sub_mosaic->avg_H = avg_H.clone();
	new_sub_mosaic->scene_size = scene_size;
	//new_sub_mosaic->neighbors = neighbors;
	for (Frame *frame : frames)
		new_sub_mosaic->addFrame(frame->clone());

	return new_sub_mosaic;
}

// See description in header file
void SubMosaic::addFrame(Frame *_frame)
{
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
		frame->setHReference(T);
	
	avg_H = T * avg_H;
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
	Point2f line[4][2];
	Point2f v1, v2;
	float side[4], cosine[4];
	float frame_dims = (float)TARGET_WIDTH * (float)TARGET_HEIGHT;
	float ratio_dims = (float)TARGET_HEIGHT / (float)TARGET_WIDTH;
	float o_sides_error = 0, c_sides_error = 0, angle_error = 0;
	float area_error = 0, min_ratio = 0;
	float tot_error = 0;

	for (Frame *frame : frames)
	{
		for (int i = 0; i < 4; i++)
		{
			line[i][0] = frame->bound_points[RANSAC][i];
			line[i][1] = frame->bound_points[RANSAC][i < 3 ? i + 1 : 0];

			side[i] = getDistance(line[i][0], line[i][1]);
		}
		for (int i = 0; i < 4; i++)
		{
			v1.x = abs(line[i][1].x - line[i][0].x);
			v1.y = abs(line[i][1].y - line[i][0].y);
			v2.x = abs(line[i < 3 ? i + 1 : 0][1].x - line[i < 3 ? i + 1 : 0][0].x);
			v2.y = abs(line[i < 3 ? i + 1 : 0][1].y - line[i < 3 ? i + 1 : 0][0].y);

			cosine[i] = (v1.x * v2.x + v1.y * v2.y) / (sqrt(v1.x * v1.x + v1.y * v1.y) * sqrt(v2.x * v2.x + v2.y * v2.y));
		}

		o_sides_error = 2 - 0.5 * (min(side[0] / side[2], side[2] / side[0]) +
									min(side[1] / side[3], side[3] / side[1]));

		min_ratio = min(side[0] / side[1], min(side[1] / side[2], min(side[2] / side[3], side[3] / side[0])));
		c_sides_error = 1 - min(min_ratio / ratio_dims, ratio_dims / min_ratio);

		vector<Point2f> aux_points = {
			frame->bound_points[RANSAC][0],
			frame->bound_points[RANSAC][1],
			frame->bound_points[RANSAC][2],
			frame->bound_points[RANSAC][3]
		};

		area_error = 1 - min(contourArea(aux_points) / frame_dims,
												 frame_dims / contourArea(aux_points));

		angle_error = pow(max(cosine[0], max(cosine[1], max(cosine[2], cosine[3]))), 5);

		//tot_error += max(o_sides_error, max(c_sides_error, max(area_error, angle_error)));
		tot_error = max(tot_error, o_sides_error + c_sides_error + area_error + angle_error);
	}

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
}

vector<vector<Point2f> > SubMosaic::getCornerPoints()
{
	Point2f first, second, third, fourth;
	float d1, d2, d3, d4;
	int i1, i2, i3, i4;
	int ref = 0, prev = 1;
	for (int i=0; i<2; i++)
	{	
		d1 = getDistance(frames[ref]->bound_points[EUCLIDEAN][0], frames[prev]->bound_points[EUCLIDEAN][4]);
		d2 = getDistance(frames[ref]->bound_points[EUCLIDEAN][1], frames[prev]->bound_points[EUCLIDEAN][4]);
		d3 = getDistance(frames[ref]->bound_points[EUCLIDEAN][2], frames[prev]->bound_points[EUCLIDEAN][4]);
		d4 = getDistance(frames[ref]->bound_points[EUCLIDEAN][3], frames[prev]->bound_points[EUCLIDEAN][4]);
		if (d1 > d2)
		{
			if (d2 > d4)
			{
				third = frames[ref]->bound_points[EUCLIDEAN][0];
				fourth = frames[ref]->bound_points[EUCLIDEAN][1];
				i3 = 0;
				i4 = 1;
			}
			else if (d1 > d3)
			{
				third = frames[ref]->bound_points[EUCLIDEAN][0];				
				fourth = frames[ref]->bound_points[EUCLIDEAN][3];
				i3 = 0;
				i4 = 3;
			}
			else
			{
				third = frames[ref]->bound_points[EUCLIDEAN][2];				
				fourth = frames[ref]->bound_points[EUCLIDEAN][3];
				i3 = 2;
				i4 = 3;
			}
		}
		else
		{
			if (d4 > d2)
			{
				third = frames[ref]->bound_points[EUCLIDEAN][2];
				fourth = frames[ref]->bound_points[EUCLIDEAN][3];
				i3 = 2;
				i4 = 3;
			}
			else if (d3 > d1)
			{
				third = frames[ref]->bound_points[EUCLIDEAN][1];				
				fourth = frames[ref]->bound_points[EUCLIDEAN][2];
				i3 = 1;
				i4 = 2;
			}
			else
			{
				third = frames[ref]->bound_points[EUCLIDEAN][0];				
				fourth = frames[ref]->bound_points[EUCLIDEAN][1];
				i3 = 0;
				i4 = 1;
			}
		}
		if (ref==0)
		{
			first = third;
			second = fourth;
			i1 = i3;
			i2 = i4;
			ref += n_frames-1;
			prev = ref - 1;
		}
	}
	vector<vector<Point2f> > corner_points(2);
	vector<Point2f> euclidean_points;
	euclidean_points.push_back(first);
	euclidean_points.push_back(second);
	euclidean_points.push_back(third);
	euclidean_points.push_back(fourth);

	vector<Point2f> perspective_points;
	perspective_points.push_back(frames[0]->bound_points[PERSPECTIVE][i1]);
	perspective_points.push_back(frames[0]->bound_points[PERSPECTIVE][i2]);
	perspective_points.push_back(frames[ref]->bound_points[PERSPECTIVE][i3]);
	perspective_points.push_back(frames[ref]->bound_points[PERSPECTIVE][i4]);

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

	// float top = TARGET_HEIGHT, bottom = 0, left = TARGET_WIDTH, right = 0;
	// for (Frame *frame : frames)
	// {
	// 	for (Point2f point : frame->bound_points[PERSPECTIVE])
	// 	{
	// 		if (point.x < left)
	// 			left = point.x;
	// 		if (point.y < top)
	// 			top = point.y;
	// 		if (point.x > right)
	// 			right = point.x;
	// 		if (point.y > bottom)
	// 			bottom = point.y;
	// 	}
	// }

	// centroid.x = (right + left) / 2;
	// centroid.y = (bottom + top) / 2;

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
