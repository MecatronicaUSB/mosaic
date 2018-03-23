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
	_sub_mosaic->computeOffset();
	_sub_mosaic->final_scene.release();
	_sub_mosaic->final_scene = Mat(_sub_mosaic->scene_size, CV_8UC3, Scalar(0, 0, 0));

	MultiBandBlender multiband(false, bands);
	if (bands > 0)
		multiband.prepare(Rect(Point(0, 0), _sub_mosaic->scene_size));
		
	vector<Frame *> frames = _sub_mosaic->frames;
	// correctColor(_sub_mosaic);
	int k = 0;
	vector<Point> corners;
	for (Frame *frame : frames)
	{
		warp_imgs.push_back(getWarpImg(frame));
		masks.push_back(getMask(frame));
		full_masks.push_back(getMask(frame));
		bound_rect.push_back(frame->bound_rect);
		corners.push_back(Point(frame->bound_rect.x, frame->bound_rect.y));
	}

	if (graph_cut)
	{
		GraphCutSeamFinder *seam_finder = new GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR_GRAD);
		seam_finder->find(warp_imgs, corners, masks);
	}
	
	correctColor(_sub_mosaic);

	Mat aux_img;
	Mat final_mask = Mat(_sub_mosaic->final_scene.size(), CV_8U, Scalar(0));
	Mat roi;
	for (int i = 0; i < frames.size(); i++)
	{
		warp_imgs[i].copyTo(aux_img);
		if (bands > 0)
		{
			multiband.feed(aux_img, masks[i], Point((int)bound_rect[i].x, (int)bound_rect[i].y));
		}
		else
		{
			roi = Mat(_sub_mosaic->final_scene, Rect(bound_rect[i].x,
													bound_rect[i].y,
													bound_rect[i].width,
													bound_rect[i].height));
			aux_img.copyTo(roi, masks[i]);
			roi = Mat(final_mask, Rect(bound_rect[i].x,
									bound_rect[i].y,
									bound_rect[i].width,
									bound_rect[i].height));
			masks[i].copyTo(roi, masks[i]);
		}
	}

	// enhanceImage(_sub_mosaic->final_scene, final_mask);
	if (bands > 0)
	{
		Mat result_16s, result_mask;
		multiband.blend(result_16s, result_mask);
		result_16s.convertTo(_sub_mosaic->final_scene, CV_8U);
	}

	warp_imgs.clear();
	masks.clear();
	full_masks.clear();
	bound_rect.clear();
}

UMat Blender::getWarpImg(Frame *_frame)
{
	Mat warp_img;

	Mat aux_T = Mat::eye(3, 3, CV_64F);
	aux_T.at<double>(0, 2) = -_frame->bound_rect.x;
	aux_T.at<double>(1, 2) = -_frame->bound_rect.y;

	warpPerspective(_frame->color, warp_img, aux_T * _frame->H, Size(_frame->bound_rect.width, _frame->bound_rect.height));

	warp_img.convertTo(warp_img, CV_32F);
	UMat warp_uimg = warp_img.getUMat(ACCESS_RW);

	return warp_uimg;
}

UMat Blender::getMask(Frame *_frame)
{
	vector<Point2f> aux_points = _frame->bound_points[PERSPECTIVE];

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

	//mask.convertTo(mask, CV_32F);
	UMat umask = mask.getUMat(ACCESS_RW);

	return umask;
}

void Blender::correctColor(SubMosaic *_sub_mosaic)
{
	Mat lab_img;
	Scalar ob_mean, sc_mean, ob_stdev, sc_stdev;
	vector<Mat> over_masks;
	vector<Mat> channels;
	Mat aux_img;

	for (int i = 0; i < warp_imgs.size() - 1; i++)
	{
		over_masks = getOverlapMasks(i+1, i);
		warp_imgs[i].copyTo(aux_img);
		aux_img.convertTo(aux_img, CV_8U);
		cvtColor(aux_img, lab_img, CV_BGR2Lab);
		meanStdDev(lab_img, sc_mean, sc_stdev, over_masks[0]);
		cvtColor(lab_img, warp_imgs[i], CV_Lab2BGR);

		warp_imgs[i+1].copyTo(aux_img);
		aux_img.convertTo(aux_img, CV_8U);
		cvtColor(aux_img, lab_img, CV_BGR2Lab);
		meanStdDev(lab_img, ob_mean, ob_stdev, over_masks[1]);
		split(lab_img, channels);
		for (int j = 0; j<3; j++)
		{
			channels[j] = (sc_stdev.val[j]*(channels[j] - ob_mean.val[j]) / ob_stdev.val[j])
										+ sc_mean.val[j];
		}
		merge(channels, lab_img);
		cvtColor(lab_img, warp_imgs[i+1], CV_Lab2BGR);

		// imshow("test 0", warp_imgs[i]);
		// imshow("mask 0", over_masks[0]);
		// imshow("mask 1", over_masks[1]);
		// imshow("test 1", warp_imgs[i+1]);
		// waitKey(0);
	}
}

vector<Mat> Blender::getOverlapMasks(int _object, int _scene)
{
	Rect obj_roi, sc_roi;
	Mat obj_mask, sc_mask;

	full_masks[_object].copyTo(obj_mask);
	full_masks[_scene].copyTo(sc_mask);

	obj_roi.x = max(bound_rect[_scene].x - bound_rect[_object].x, 0.f);
	obj_roi.y = max(bound_rect[_scene].y - bound_rect[_object].y, 0.f);

	sc_roi.x = max(bound_rect[_object].x - bound_rect[_scene].x, 0.f);
	sc_roi.y = max(bound_rect[_object].y - bound_rect[_scene].y, 0.f);

	obj_roi.width = min(bound_rect[_scene].x + bound_rect[_scene].width,
											bound_rect[_object].x + bound_rect[_object].width) - 
											max(bound_rect[_scene].x, bound_rect[_object].x);
						
	obj_roi.height = min(bound_rect[_scene].y + bound_rect[_scene].height,
											bound_rect[_object].y + bound_rect[_object].height) - 
											max(bound_rect[_scene].y, bound_rect[_object].y);

	sc_roi.width = obj_roi.width;
	sc_roi.height = obj_roi.height;

	Mat object_overlap(obj_mask, obj_roi);
	Mat scene_overlap(sc_mask, sc_roi);

	Mat overlap_sc_mask(sc_mask.size(), CV_8U, Scalar(0));
	Mat overlap_obj_mask(obj_mask.size(), CV_8U, Scalar(0));

	Mat and_mask;
	
	and_mask = Mat(overlap_sc_mask, sc_roi);
	bitwise_and(scene_overlap, object_overlap, and_mask);
	and_mask = Mat(overlap_obj_mask, obj_roi);
	bitwise_and(scene_overlap, object_overlap, and_mask);

	vector<Mat> overlap_masks;
	overlap_masks.push_back(overlap_sc_mask);
	overlap_masks.push_back(overlap_obj_mask);
	
	return overlap_masks;
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