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
	// calculate the offset by the location of the bounding rectangle
	vector<float> offset = {-frames[0]->bound_rect.y, 0, -frames[0]->bound_rect.x, 0};
	// translate all sub mosaic
	updateOffset(offset);
}

// See description in header file
void SubMosaic::computeOffset()
{
	float top, bottom, left, right;
	// initial values
	top = frames[0]->bound_points[PERSPECTIVE][0].y;
	bottom = frames[0]->bound_points[PERSPECTIVE][0].y;
	left = frames[0]->bound_points[PERSPECTIVE][0].x;
	right = frames[0]->bound_points[PERSPECTIVE][0].x;
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
	// translate the mosaic based on previous offset
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
	// update transformation to last frame of mosaic
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
		tot_error = max(tot_error, frame->frameDistortion(_ref));
	
	return tot_error;
}

// See description in header file
void SubMosaic::correct()
{
	// get corner points of mosaic, based on first and last frame
	vector<vector<Point2f> > corner_points = getCornerPoints();
	// calculate perspective transform from perspective points to euclidean ones
	Mat correct_H1 = getPerspectiveTransform(corner_points[PERSPECTIVE], corner_points[EUCLIDEAN]);
	float temp_distortion;
	for (Frame *frame : frames)
	{
		// save it in temporal frame object variable
		perspectiveTransform(frame->bound_points[PERSPECTIVE], frame->bound_points[RANSAC], correct_H1);
	}
	// calculate the overall geometric distortion
	temp_distortion = calcDistortion(RANSAC);
	corner_points = getBorderPoints();
	Mat correct_H2 = getPerspectiveTransform(corner_points[PERSPECTIVE], corner_points[EUCLIDEAN]);
	for (Frame *frame : frames)
	{
		// save it in temporal frame object variable
		perspectiveTransform(frame->bound_points[PERSPECTIVE], frame->bound_points[RANSAC], correct_H2);
	}
	Mat correct_H;
	if (temp_distortion < calcDistortion(RANSAC))
	{
		cout << "corners" << endl;
		correct_H = correct_H1;
	}
	else
	{
		cout << "borders" << endl;
		correct_H = correct_H2;
	}


	// update each frame of mosaic
	for (Frame *frame : frames)
		frame->setHReference(correct_H);
	// update transformation to last frame of mosaic
	next_H = correct_H * next_H;
	next_E = correct_H * next_E;
}


vector<vector<Point2f> > SubMosaic::getBorderPoints()
{
	int p1=0, p2=0, p3=0, p4=0;
	int f1=0, f2=0, f3=0, f4=0;
	int pidx=-1, fidx=-1;
	float top, bottom, left, right;
	
	top = frames[0]->bound_points[EUCLIDEAN][4].y;
	bottom = frames[0]->bound_points[EUCLIDEAN][4].y;
	left = frames[0]->bound_points[EUCLIDEAN][4].x;
	right = frames[0]->bound_points[EUCLIDEAN][4].x;
	
	for (Frame *frame: frames)
	{
		fidx++;
		pidx=-1;
		for (Point2f point: frame->bound_points[EUCLIDEAN])
		{
			pidx++;
			if (point.y < top)
			{
				top = point.y;
				p1 = pidx;
				f1 = fidx;
			}
			if (point.y > bottom)
			{
				bottom = point.y;
				p2 = pidx;
				f2 = fidx;
			}
			if (point.x < left)
			{
				left = point.x;
				p3 = pidx;
				f3 = fidx;
			}
			if (point.x > right)
			{
				right = point.x;
				p4 = pidx;
				f4 = fidx;
			}
		}
	}

	vector<vector<Point2f> > border_points(2);
	// save points with bigger distance for each frame
	vector<Point2f> euclidean_points;
	euclidean_points.push_back(frames[f1]->bound_points[EUCLIDEAN][p1]);
	euclidean_points.push_back(frames[f2]->bound_points[EUCLIDEAN][p2]);
	euclidean_points.push_back(frames[f3]->bound_points[EUCLIDEAN][p3]);
	euclidean_points.push_back(frames[f4]->bound_points[EUCLIDEAN][p4]);
	// save points with bigger distance for each frame, using the point index
	vector<Point2f> perspective_points;
	perspective_points.push_back(frames[f1]->bound_points[PERSPECTIVE][p1]);
	perspective_points.push_back(frames[f2]->bound_points[PERSPECTIVE][p2]);
	perspective_points.push_back(frames[f3]->bound_points[PERSPECTIVE][p3]);
	perspective_points.push_back(frames[f4]->bound_points[PERSPECTIVE][p4]);

	border_points[EUCLIDEAN]= euclidean_points;
	border_points[PERSPECTIVE]= perspective_points;

	return border_points;
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
	// loop over all key points position
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
	int width, height, point_size = 5;
	float factor;
	// compute resize factor
	if (scene_size.height < scene_size.width)
	{
		width = 1024;
		factor = (float)width / scene_size.width;
		height = scene_size.height * factor;		
	}
	else
	{
		height = 1024;
		factor = (float)height / scene_size.height;
		width = scene_size.width * factor;
	}
	Mat scene_map = Mat(Size(width, height), CV_8UC3, Scalar(255, 255, 255));
	// check type of map
	if (_type == RECTANGLE)
	{
		// print form of each frame
		vector<vector<Point>> map_points;
		vector<Point> aux_points(4);
		for (Frame *frame : frames)
		{
			aux_points[0] = Point(frame->bound_points[PERSPECTIVE][0].x, 
								  frame->bound_points[PERSPECTIVE][0].y) * factor;
			aux_points[1] = Point(frame->bound_points[PERSPECTIVE][1].x, 
								  frame->bound_points[PERSPECTIVE][1].y) * factor;
			aux_points[2] = Point(frame->bound_points[PERSPECTIVE][2].x, 
								  frame->bound_points[PERSPECTIVE][2].y) * factor;
			aux_points[3] = Point(frame->bound_points[PERSPECTIVE][3].x, 
								  frame->bound_points[PERSPECTIVE][3].y) * factor;
			map_points.push_back(aux_points);
		}
		polylines(scene_map, map_points, true, Scalar(0,0,0), 1);
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
								 frame->bound_points[PERSPECTIVE][4].y) * factor;
			for (Frame *neighbor : frame->neighbors)
			{
				neighbor_point = Point(neighbor->bound_points[PERSPECTIVE][4].x,
								 	   neighbor->bound_points[PERSPECTIVE][4].y) * factor;
				line(scene_map, center_point, neighbor_point, Scalar(190, 190, 190), 1);
			}
		}
		// second print the colored links (good neighbors)		
		for (Frame *frame : frames)
		{
			center_point = Point(frame->bound_points[PERSPECTIVE][4].x,
								 frame->bound_points[PERSPECTIVE][4].y) * factor;
			for (Frame *neighbor : frame->good_neighbors)
			{
				neighbor_point = Point(neighbor->bound_points[PERSPECTIVE][4].x,
								 	   neighbor->bound_points[PERSPECTIVE][4].y) * factor;
				line(scene_map, center_point, neighbor_point, line_color, 1);
			}
		}
		// third print a circle at the center of each frame
		int w=1;
		for (Frame *frame : frames)
		{
			center_point = Point(frame->bound_points[PERSPECTIVE][4].x,
								 frame->bound_points[PERSPECTIVE][4].y) * factor;
			//putText(scene_map, to_string(w++), Point(center_point.x+3, center_point.y+((w==3)?8:2)), FONT_HERSHEY_PLAIN, 2, cvScalar(0, 0, 0), 2);
			circle(scene_map, center_point, point_size, _color, -1);
		}
	}
    	copyMakeBorder(scene_map, scene_map, 40, 50, 100, 30, BORDER_CONSTANT,Scalar(255,255,255));
	// y-axis labels
	rectangle(scene_map, Point(70, 30), Point(width+110, height+50), cvScalar(0, 0, 0), 1);
	putText(scene_map, to_string(0), Point(5, 50), FONT_HERSHEY_PLAIN, 1, cvScalar(0, 0, 0), 1);
	putText(scene_map, to_string((int)(height/factor)/4), Point(5, (height-50) / 4), FONT_HERSHEY_PLAIN, 1, cvScalar(0, 0, 0), 1);
	putText(scene_map, to_string((int)(height/factor)/2), Point(5, (height-50) / 2), FONT_HERSHEY_PLAIN, 1, cvScalar(0, 0, 0), 1);
	putText(scene_map, to_string((int)(height/factor)*3/4), Point(5, (height-50) * 3 / 4), FONT_HERSHEY_PLAIN, 1, cvScalar(0, 0, 0), 1);
	putText(scene_map, to_string((int)(height/factor)), Point(5, height-50), FONT_HERSHEY_PLAIN, 1, cvScalar(0, 0, 0), 1);
	// x-axis labels
	putText(scene_map, to_string(0), Point(65, height+65), FONT_HERSHEY_PLAIN, 1, cvScalar(0, 0, 0), 1);
	putText(scene_map, to_string((int)(width/factor)/4), Point((width+20)/4+60, height+65), FONT_HERSHEY_PLAIN, 1, cvScalar(0, 0, 0), 1);
	putText(scene_map, to_string((int)(width/factor)/2), Point((width+20)/2+60, height+65), FONT_HERSHEY_PLAIN, 1, cvScalar(0, 0, 0), 1);
	putText(scene_map, to_string((int)(width/factor)*3/4), Point((width+20)*3/4+60, height+65), FONT_HERSHEY_PLAIN, 1, cvScalar(0, 0, 0), 1);
	putText(scene_map, to_string((int)(width/factor)), Point(width+80, height+65), FONT_HERSHEY_PLAIN, 1, cvScalar(0, 0, 0), 1);
	putText(scene_map, "Unidades en Pixeles", Point((int)(width/2)-20, 20), FONT_HERSHEY_PLAIN, 1, cvScalar(0, 0, 0), 1);
	return scene_map;
}

// See description in header file
void saveHomographyData(Mat _h, vector<KeyPoint> keypoints[2], vector<DMatch> matches)
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
