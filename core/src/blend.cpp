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
	// translate sub mosaic to positive location
	_sub_mosaic->computeOffset();
	_sub_mosaic->final_scene.release();
	// create final mosaic image
	_sub_mosaic->final_scene = Mat(_sub_mosaic->scene_size, CV_8UC3, Scalar(255, 255, 255));

	MultiBandBlender multiband(false, bands);
	// define number of bands for multi-band blender
	if (bands > 0 && graph_cut)
		multiband.prepare(Rect(Point(0, 0), _sub_mosaic->scene_size));
	// save frames in class variable
	vector<Frame *> frames = _sub_mosaic->frames;

	vector<Point> corners;
	// save data in class variables
	for (int i=0; i<frames.size(); i++)
	{
		// warp image, by each transformation
		warp_imgs.push_back(getWarpImg(frames[i]));
		// mask to be cropped
		masks.push_back(getMask(frames[i]));
		// constant filled mask
		full_masks.push_back(masks[i].clone());
		// bounding rectangle
		bound_rect.push_back(frames[i]->bound_rect);
		// corners in array mode
		corners.push_back(Point(frames[i]->bound_rect.x, frames[i]->bound_rect.y));
	}
	Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(ExposureCompensator::GAIN);
	compensator->feed(corners, warp_imgs, masks);

	// for (int i = 0; i < warp_imgs.size() - 1; i++)
	// {
	// 	compensator->apply(i+1, corners[i], warp_imgs[i], masks[i]);
	// }
	//compensator->apply(4, corners[3], warp_imgs[3], masks[3]);

	// for (int i=0 ; i<exp_warp_img.size(); i++)
	// {
	// 	exp_warp_img[i].convertTo(exp_warp_img[i], CV_32F);
	// 	warp_imgs.push_back(exp_warp_img[i].getUMat(ACCESS_RW));
	// 	masks.push_back(exp_mask[i].getUMat(ACCESS_RW));
	// }
	int j=0;
	// apply graph cur algorithm
	if (graph_cut)
	{
		cout<<"\rFinding cut line..."<<yellow<< "\tthis may take some time..."<<reset<<flush;
		GraphCutSeamFinder *seam_finder = new GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR_GRAD);
		seam_finder->find(warp_imgs, corners, masks);
		cout<<"\rFinding cut line\t"<<green<<"OK                          "<<reset<<flush<<endl;
	}
	else
	{
		// for (int i = 0; i < warp_imgs.size(); i++)
		// {
		// 	j = i + 1;
		// 	while (j < warp_imgs.size())
		// 	{
		// 		if (!checkCollision(frames[j], frames[i]))
		// 			break;
				
		// 		cropMask(j++, i);

		// 	}
		// }
	}
	Mat aux_img;
	//imwrite("/home/victor/dataset/Results/cut-line/geo/mask0x.jpg", masks[0]); 
	for (int i = 0; i < warp_imgs.size(); i++)
	{
		warp_imgs[i].copyTo(aux_img);
		aux_img.convertTo(aux_img, CV_8U);
		aux_img.copyTo(warp_imgs[i]);
	}
	// correct color by Reinhard's method or Gain compensation
	if (color_correction)
	{
		correctColor(_sub_mosaic);
		cout << flush << "\rCorrecting color\t"<<green<<"OK"<<reset;
	}
	
	//Mat final_mask = Mat(_sub_mosaic->final_scene.size(), CV_8U, Scalar(0));
	Mat roi;
	cout << endl << "Blending...\t";
	// loop over all frames
	// vector<vector<vector<Point> > > tot_contour;
	// vector<vector<Point> > cont;
	// for (int i = 0; i < masks.size(); i++)
	// {
	// 	findContours(masks[i], cont, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	// 	drawContours(warp_imgs[i], cont, -1, Scalar(0,0,255), 2);		
	// 	tot_contour.push_back(cont);
	// 	cont.clear();
	// }

	for (int i = 0; i < frames.size(); i++)
	{
		warp_imgs[i].copyTo(aux_img);
		// if multi-band blender if selected
		if (bands > 0 && graph_cut)
		{
			multiband.feed(aux_img, masks[i], Point((int)bound_rect[i].x, (int)bound_rect[i].y));
		}
		// else copy images to final image
		else
		{
			// locate region of interest in final image
			roi = Mat(_sub_mosaic->final_scene, Rect(bound_rect[i].x,
													 bound_rect[i].y,
													 bound_rect[i].width,
													 bound_rect[i].height));
			// copy using mask
			aux_img.copyTo(roi, masks[i]);
			// roi = Mat(final_mask, Rect(bound_rect[i].x,
			// 						   bound_rect[i].y,
			// 						   bound_rect[i].width,
			// 						   bound_rect[i].height));
			// copy final image mask
			//masks[i].copyTo(roi, masks[i]);
		}
	}
	// for (int i = 0; i < masks.size(); i++)
	// {
	// 	drawContours(_sub_mosaic->final_scene, tot_contour[i], -1, Scalar(0,0,255), 1);
	// }
	// apply multi-band blender (if selected)	
	if (bands > 0 && graph_cut)
	{
		Mat result_mask, result_16s;
		// blender uses CV_16S data type
		multiband.blend(result_16s, result_mask);
		result_16s.convertTo(_sub_mosaic->final_scene, CV_8U);
		//final_mask = result_mask;
	}
	cout<<flush << "\rBlending\t\t"<<green<<"OK"<<reset;
	// clear used data
	warp_imgs.clear();
	masks.clear();
	full_masks.clear();
	bound_rect.clear();
}

// See description in header file
UMat Blender::getWarpImg(Frame *_frame)
{
	Mat warp_img;
	// get translation matrix, to locate image at default position
	Mat aux_T = Mat::eye(3, 3, CV_64F);
	aux_T.at<double>(0, 2) = -_frame->bound_rect.x;
	aux_T.at<double>(1, 2) = -_frame->bound_rect.y;
	// apply or not SCB to final image
	if (scb)
		enhanceImage(_frame->color);
	// get warp image
	warpPerspective(_frame->color, warp_img, aux_T * _frame->H, Size(_frame->bound_rect.width, _frame->bound_rect.height));
	// cast to UMat and CV_32F type (for graph cut algorithm)
	warp_img.convertTo(warp_img, CV_32F);
	UMat warp_uimg = warp_img.getUMat(ACCESS_RW);

	return warp_uimg;
}

// See description in header file
UMat Blender::getMask(Frame *_frame)
{
	vector<Point2f> aux_points = _frame->bound_points[PERSPECTIVE];
	// get translation offset, to locate points at default position
	for (Point2f &point : aux_points)
	{
		point.x -= _frame->bound_rect.x;
		point.y -= _frame->bound_rect.y;
	}
	// reduce mask to the center point (to avoid small black gaps in final image)
	for (int i = 0; i < aux_points.size() - 1; i++)
	{
		// point[4] correspond to center point
		aux_points[i].x += 5 * (aux_points[4].x - aux_points[i].x) / 100;
		aux_points[i].y += 5 * (aux_points[4].y - aux_points[i].y) / 100;
	}
	// save corner points
	Point mask_points[4] = {
		aux_points[0],
		aux_points[1],
		aux_points[2],
		aux_points[3],
	};
	Mat mask(_frame->bound_rect.height, _frame->bound_rect.width, CV_8U, Scalar(0));
	// fill area between points with white
	fillConvexPoly(mask, mask_points, 4, Scalar(255));
	// cast to UMat type (for graph cut algorithm)
	UMat umask = mask.getUMat(ACCESS_RW);

	return umask;
}

void Blender::cropMask(int _object, int _scene)
{
	Mat obj_mask, sc_mask;
	// cast to Mat type
	masks[_object].copyTo(obj_mask);
	masks[_scene].copyTo(sc_mask);
	Rect overlap_roi;

	overlap_roi.x = max(bound_rect[_scene].x, bound_rect[_object].x) - bound_rect[_object].x;
	overlap_roi.y = max(bound_rect[_scene].y, bound_rect[_object].y) - bound_rect[_object].y;

	overlap_roi.width = min(bound_rect[_scene].x + bound_rect[_scene].width,
							bound_rect[_object].x + bound_rect[_object].width) - 
						max(bound_rect[_scene].x, bound_rect[_object].x);
						

	overlap_roi.height = min(bound_rect[_scene].y + bound_rect[_scene].height,
							 bound_rect[_object].y + bound_rect[_object].height) - 
						 max(bound_rect[_scene].y, bound_rect[_object].y);
						 

	Mat object_roi(obj_mask, overlap_roi);

	overlap_roi.x = max(bound_rect[_object].x - bound_rect[_scene].x, 0.f);
	overlap_roi.y = max(bound_rect[_object].y - bound_rect[_scene].y, 0.f);

	Mat scene_roi(sc_mask, overlap_roi);

	scene_roi -= object_roi;
	obj_mask.copyTo(masks[_object]);
	sc_mask.copyTo(masks[_scene]);
}

// See description in header file
void Blender::correctColor(SubMosaic *_sub_mosaic)
{
	Mat lab_img;
	Scalar temp_ob_mean, temp_sc_mean, temp_ob_stdev, temp_sc_stdev;
	vector<Scalar> ob_mean, sc_mean, ob_stdev, sc_stdev;
	vector<Mat> over_masks;
	vector<Mat> channels;
	Mat aux_img;
	// loop over all warp images
	// for (int i = 0; i < warp_imgs.size() - 1; i++)
	// {
	// 	// get intersection mask, referenced to each frame
	// 	over_masks = getOverlapMasks(i+1, i);
	// 	// cast from UMat to Mat type
	// 	warp_imgs[i].copyTo(aux_img);
	// 	aux_img.convertTo(aux_img, CV_8U);
	// 	// convert to CieLab color space
	// 	cvtColor(aux_img, lab_img, CV_BGR2Lab);
	// 	// get the mean and standard deviation in intersection area
	// 	meanStdDev(lab_img, temp_sc_mean, temp_sc_stdev, over_masks[0]);
	// 	sc_mean.push_back(temp_sc_mean);
	// 	sc_stdev.push_back(temp_sc_stdev);
	// 	// return to BGR color space
	// 	cvtColor(lab_img, warp_imgs[i], CV_Lab2BGR);
	// 	// -- now with the first image, apply the first 3 steps
	// 	warp_imgs[i+1].copyTo(aux_img);
	// 	aux_img.convertTo(aux_img, CV_8U);
	// 	cvtColor(aux_img, lab_img, CV_BGR2Lab);
	// 	meanStdDev(lab_img, temp_ob_mean, temp_ob_stdev, over_masks[1]);
	// 	ob_mean.push_back(temp_ob_mean);
	// 	ob_stdev.push_back(temp_ob_stdev);
	// 	cvtColor(lab_img, warp_imgs[i+1], CV_Lab2BGR);
	// }
	Mat gray_img;
	vector<Scalar> sc_g_mean, ob_g_mean;
	Scalar aux_mean;
	// for (int i = 0; i < warp_imgs.size()-1; i++)
	// {
	// 	warp_imgs[i+1].copyTo(aux_img);
	// 	cvtColor(aux_img, lab_img, CV_BGR2Lab);
	// 	// modify histogram of each channel for second image, based on first one
	// 	// (in CieLab color space)
	// 	split(lab_img, channels);
	// 	Scalar aux_mean1, aux_stdev1, aux_mean2, aux_stdev2;
	// 	for (int j = 0; j<3; j++)
	// 	{
	// 		// getHistogram(channels[j], histogram);
	// 		// printHistogram(histogram, "/home/victor/dataset/Results/Color-Correction/U_hist_"+to_string(j)+".png", color[j]);

	// 		channels[j] = (sc_stdev[i].val[j]*(channels[j] - ob_mean[i].val[j]) / ob_stdev[i].val[j])
	// 									+ sc_mean[i].val[j];

	// 		// getHistogram(channels[j], histogram);
	// 		// printHistogram(histogram, "/home/victor/dataset/Results/Color-Correction/C_hist_"+to_string(j)+".png", color[j]);
	// 	}
	// 	merge(channels, lab_img);
	// 	// return to BGR color space
	// 	cvtColor(lab_img, warp_imgs[i+1], CV_Lab2BGR);
	// }
	for (int i=0; i<warp_imgs.size()-1; i++)
	{
		over_masks = getOverlapMasks(i+1, i);
		
		warp_imgs[i].copyTo(aux_img);
		cvtColor(aux_img, gray_img, CV_BGR2GRAY);
		sc_g_mean.push_back(mean(aux_img , over_masks[0]));

		warp_imgs[i+1].copyTo(aux_img);
		cvtColor(aux_img, gray_img, CV_BGR2GRAY);
		ob_g_mean.push_back(mean(aux_img , over_masks[1]));
	}
	for (int i=0; i<warp_imgs.size()-2; i++)
	{
		warp_imgs[i+1].copyTo(aux_img);
		aux_img *= (sc_g_mean[i].val[0] +ob_g_mean[i+1].val[0]) / (ob_g_mean[i].val[0]+sc_g_mean[i+1].val[0]);
		
		aux_img.copyTo(warp_imgs[i+1]);
	}
	// warp_imgs[0].copyTo(aux_img);
	// aux_img *= (ob_g_mean[0].val[0]) / (sc_g_mean[0].val[0]);
	// aux_img.copyTo(warp_imgs[0]);

	// warp_imgs[warp_imgs.size()-1].copyTo(aux_img);
	// aux_img *= (sc_g_mean[warp_imgs.size()-2].val[0]) / (ob_g_mean[warp_imgs.size()-2].val[0]);
	// aux_img.copyTo(warp_imgs[warp_imgs.size()-1]);
		
}

// See description in header file
vector<Mat> Blender::getOverlapMasks(int _object, int _scene)
{
	Rect obj_roi, sc_roi;
	Mat obj_mask, sc_mask;
	// cast to Mat type
	full_masks[_object].copyTo(obj_mask);
	full_masks[_scene].copyTo(sc_mask);
	// locate intersection roi in the object image
	obj_roi.x = max(bound_rect[_scene].x - bound_rect[_object].x, 0.f);
	obj_roi.y = max(bound_rect[_scene].y - bound_rect[_object].y, 0.f);
	// locate intersection roi in the scene image
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
	// intersection roi of object mask
	Mat object_overlap(obj_mask, obj_roi);
	// intersection roi of scene mask	
	Mat scene_overlap(sc_mask, sc_roi);
	// auxiliar masks to not modify original ones
	Mat overlap_sc_mask(sc_mask.size(), CV_8U, Scalar(0));
	Mat overlap_obj_mask(obj_mask.size(), CV_8U, Scalar(0));

	Mat and_mask;
	// apply and operation on each mask
	and_mask = Mat(overlap_sc_mask, sc_roi);
	bitwise_and(scene_overlap, object_overlap, and_mask);
	and_mask = Mat(overlap_obj_mask, obj_roi);
	bitwise_and(scene_overlap, object_overlap, and_mask);

	vector<Mat> overlap_masks;
	// save masks
	overlap_masks.push_back(overlap_sc_mask);
	overlap_masks.push_back(overlap_obj_mask);
	
	return overlap_masks;
}

// See description in header file
// void Blender::cropMask(int _object, int _scene)
// {
// 	Rect overlap_roi;
// 	// get overlap roi dimensions
// 	overlap_roi.x = max(bound_rect[_scene].x, bound_rect[_object].x) - bound_rect[_object].x;
// 	overlap_roi.y = max(bound_rect[_scene].y, bound_rect[_object].y) - bound_rect[_object].y;
// 	overlap_roi.width = min(bound_rect[_scene].x + bound_rect[_scene].width,
// 							bound_rect[_object].x + bound_rect[_object].width) - 
// 						max(bound_rect[_scene].x, bound_rect[_object].x);
// 	overlap_roi.height = min(bound_rect[_scene].y + bound_rect[_scene].height,
// 							 bound_rect[_object].y + bound_rect[_object].height) - 
// 						 max(bound_rect[_scene].y, bound_rect[_object].y);
	
// 	Mat obj_mask, sc_mask;
// 	// cast to Mat type
// 	masks[_object].copyTo(obj_mask);
// 	masks[_scene].copyTo(sc_mask);
// 	// locate roi in object image masks
// 	Mat object_roi(obj_mask, overlap_roi);
// 	// move roi to scene intersection location
// 	overlap_roi.x = max(bound_rect[_object].x - bound_rect[_scene].x, 0.f);
// 	overlap_roi.y = max(bound_rect[_object].y - bound_rect[_scene].y, 0.f);

// 	// locate roi in object image masks
// 	Mat scene_roi(sc_mask, overlap_roi);
// 	// remove scene portion mask that intersect scene
// 	scene_roi -= object_roi;
// 	obj_mask.copyTo(masks[_object]);
// 	sc_mask.copyTo(masks[_scene]);
// }

// See description in header file
vector<Point2f> Blender::findLocalStitch(Frame *_object, Frame *_scene)
{
	vector<Point2f> local_stitch;
	vector<BlendPoint> blend_points;
	// initialize match points
	for (int i = 0; i < _object->grid_points[PREV].size(); i++)
	{
		blend_points.push_back(BlendPoint(i, _object->grid_points[PREV][i],
										  _scene->grid_points[NEXT][i]));
	}
	// sort by match distance
	sort(blend_points.begin(), blend_points.end());
	// get the matches with less distance (they will describe the local stitch area)
	vector<Point2f> good_points;
	// select a percent of points (first point is the one with less distance)
	int percentile = 5.f * (float)blend_points.size() / 100.f;
	// TODO: use best criteria selection
	for (int i = 0; i < percentile; i++)
	{
		good_points.push_back(blend_points[i].prev);
	}
	// return area covered by best matches (external points of local stitch)
	convexHull(good_points, local_stitch);

	return good_points;
}

bool Blender::checkCollision(Frame *_object, Frame *_scene)
{
	if (_object->bound_rect.x > _scene->bound_rect.x)
		if (_object->bound_rect.x > _scene->bound_rect.x + _scene->bound_rect.width)
			return false;
		
	if (_object->bound_rect.y > _scene->bound_rect.y)
		if (_object->bound_rect.y > _scene->bound_rect.y + _scene->bound_rect.height)
			return false;

	return true;
}

}
