/**
 * @file blend.cpp
 * @brief Implementation of Blend class functions 
 * @version 0.2
 * @date 10/02/2018
 * @author Victor Garcia
 */
#include "../include/blend.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;

namespace m2d
{

// See description in header file
void Blender::blendSubMosaic(SubMosaic *_sub_mosaic)
{
	MultiBandBlender multiband(false, bands);

	_sub_mosaic->final_scene.release();

	if (_sub_mosaic->calcDistortion() > 5)
	{
		cout << "mosaic too distorted to blend" << endl;
		return;
	}

	_sub_mosaic->final_scene = Mat(_sub_mosaic->scene_size, CV_8UC3, Scalar(0, 0, 0));
	multiband.prepare(Rect(Point(0, 0), _sub_mosaic->scene_size));

	vector<Frame *> frames = _sub_mosaic->frames;
	//reverse(_sub_mosaic->frames.begin(), _sub_mosaic->frames.end());
	UMat aux_u_img, aux_u_mask;
	Mat aux_img, aux_mask;
	int k = 0;
	correctColor(_sub_mosaic);
	for (Frame *frame : frames)
	{
		aux_img = getWarpImg(frame);
		//aux_img.copyTo(aux_u_img);
		aux_img.convertTo(aux_img, CV_32F);
		aux_u_img = aux_img.getUMat(ACCESS_RW);
		warp_imgs.push_back(aux_u_img.clone());
		aux_mask = getMask(frame);
		// aux_mask.copyTo(aux_u_mask);
		aux_u_mask = aux_mask.getUMat(ACCESS_RW);
		masks.push_back(aux_u_mask.clone());
		bound_rect.push_back(frame->bound_rect);
	}
	Mat result_16s, result_mask;
	// int j = 0;
	// for (int i = 0; i < frames.size(); i++)
	// {
	// 	j = i + 1;
	// 	while (j < frames.size())
	// 	{
	// 		if (checkCollision(frames[j], frames[i]))
	// 			break;
			
	// 		cropMask(j++, i);

	// 	}
	// 	dilate(masks[i], masks[i],Mat(), Point(-1, -1), 1, 1, 1);
	// 	//multiband.feed(warp_imgs[i], masks[i], Point((int)bound_rect[i].x, (int)bound_rect[i].y));
	// }

	vector<Point> pts;
	for (int i = 0; i < frames.size(); i++)
	{
		pts.push_back(Point((int)bound_rect[i].x, (int)bound_rect[i].y));
	}

	GraphCutSeamFinder *seam_finder = new GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR);

	seam_finder->find(warp_imgs, pts, masks);

	for (int i = 0; i < frames.size(); i++)
	{
		warp_imgs[i].copyTo(aux_img);
		aux_img.convertTo(aux_img, CV_8U);

		// imshow("warp", aux_img);
		// imshow("mask", masks[i]);
		// waitKey(0);
	}
	//multiband.blend(result_16s, result_mask);
	//result_16s.convertTo(_sub_mosaic->final_scene, CV_8U);
	warp_imgs.clear();
	masks.clear();
	bound_rect.clear();
}

Mat Blender::getWarpImg(Frame *_frame)
{
	Mat warp_img;

	Mat aux_T = Mat::eye(3, 3, CV_64F);
	aux_T.at<double>(0, 2) = -_frame->bound_rect.x;
	aux_T.at<double>(1, 2) = -_frame->bound_rect.y;

	warpPerspective(_frame->color, warp_img, aux_T * _frame->H, Size(_frame->bound_rect.width, _frame->bound_rect.height));

	return warp_img;
}

void Blender::correctColor(SubMosaic *_sub_mosaic)
{
	vector<Mat> lab_imgs;
	Mat lab_img;
	vector<Scalar> mean, stdev;
	Scalar avg_mean, aux_mean, avg_stdev, aux_stdev;
	int n = 0;


	for (Frame *frame: _sub_mosaic->frames)
	{	
		frame->enhance();
		cvtColor(frame->color, lab_img, CV_BGR2Lab);
		meanStdDev(lab_img, aux_mean, aux_stdev);
		mean.push_back(aux_mean);
		avg_mean += aux_mean;
		stdev.push_back(aux_stdev);
		avg_stdev += aux_stdev;
		lab_imgs.push_back(lab_img.clone());
		n++;
	}
	avg_mean /= n;
	avg_stdev /= n;

	vector<Mat> channels;
	for (int i = 0; i < lab_imgs.size(); i++)
	{
		split(lab_imgs[i], channels);
		for (int j = 0; j<3; j++)
		{
			channels[j] = (avg_stdev.val[j]*(channels[j] - mean[i].val[j]) / stdev[i].val[j])
							+ avg_mean.val[j];
		}
		merge(channels, lab_imgs[i]);
		cvtColor(lab_imgs[i], _sub_mosaic->frames[i]->color, CV_Lab2BGR);
	}

}

Mat Blender::getMask(Frame *_frame)
{
	vector<Point2f> aux_points = _frame->bound_points[FIRST];

	for (Point2f &point : aux_points)
	{
		point.x -= _frame->bound_rect.x;
		point.y -= _frame->bound_rect.y;
	}

	for (int i = 0; i < aux_points.size() - 1; i++)
	{
		// point[4] correspond to center point
		aux_points[i].x += 5 * (aux_points[4].x - aux_points[i].x) / 100;
		aux_points[i].y += 5 * (aux_points[4].y - aux_points[i].y) / 100;
	}

	Point mask_points[4] = {
		aux_points[0],
		aux_points[1],
		aux_points[2],
		aux_points[3],
	};

	Mat mask(_frame->bound_rect.height, _frame->bound_rect.width, CV_8U, Scalar(0));
	fillConvexPoly(mask, mask_points, 4, Scalar(255));

	return mask;
}

bool Blender::checkCollision(Frame *_object, Frame *_scene)
{

	if (_object->bound_rect.x > _scene->bound_rect.x + _scene->bound_rect.width)
		return false;
    
    if (_object->bound_rect.x + _object->bound_rect.width < _scene->bound_rect.x)
		return false;

	if (_object->bound_rect.y > _scene->bound_rect.y + _scene->bound_rect.height)
		return false;

    if (_object->bound_rect.y + _object->bound_rect.height > _scene->bound_rect.y)
        return false;

	return true;
}

void Blender::cropMask(int _object, int _scene)
{
	// Rect overlap_roi;

	// overlap_roi.x = max(bound_rect[_scene].x, bound_rect[_object].x) - bound_rect[_object].x;
	// overlap_roi.y = max(bound_rect[_scene].y, bound_rect[_object].y) - bound_rect[_object].y;

	// overlap_roi.width = min(bound_rect[_scene].x + bound_rect[_scene].width,
	// 						bound_rect[_object].x + bound_rect[_object].width) - 
	// 					max(bound_rect[_scene].x, bound_rect[_object].x);
						

	// overlap_roi.height = min(bound_rect[_scene].y + bound_rect[_scene].height,
	// 						 bound_rect[_object].y + bound_rect[_object].height) - 
	// 					 max(bound_rect[_scene].y, bound_rect[_object].y);
						 

	// Mat object_roi(masks[_object], overlap_roi);

	// overlap_roi.x = max(bound_rect[_object].x - bound_rect[_scene].x, 0.f);
	// overlap_roi.y = max(bound_rect[_object].y - bound_rect[_scene].y, 0.f);

	// Mat scene_roi(masks[_scene], overlap_roi);

	// scene_roi -= object_roi;

}

vector<Point2f> Blender::findLocalStitch(Frame *_object, Frame *_scene)
{
	vector<Point2f> local_stitch;
	vector<BlendPoint> blend_points;

	for (int i = 0; i < _object->grid_points[PREV].size(); i++)
	{
		blend_points.push_back(BlendPoint(i, _object->grid_points[PREV][i],
										  _scene->grid_points[NEXT][i]));
	}

	sort(blend_points.begin(), blend_points.end());

	vector<Point2f> good_points;
	int percentile = 5 * blend_points.size() / 100;
	for (int i = 0; i < blend_points.size(); i++)
	{
		good_points.push_back(blend_points[i].prev);
	}

	convexHull(good_points, local_stitch);

	return good_points;
}
}