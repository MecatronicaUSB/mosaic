#include "../include/stitch.h"

using namespace std;
using namespace cv;

Rect getBound(Mat H, int width, int height){
	Rect r;
	vector<Point2f> points;
	points.push_back(Point2f(0,0));
	points.push_back(Point2f(width,0));
	points.push_back(Point2f(width,height));
	points.push_back(Point2f(0,height));

	// double w;
	// for(int i=0; i<points.size(); i++){
	// 	w = H.at<double>(2,0)*points[i].x + H.at<double>(2,1)*points[i].y + H.at<double>(2,2);
	// 	points[i].x = (H.at<double>(0,0)*points[i].x + H.at<double>(0,1)*points[i].y + H.at<double>(0,2)) / w;
	// 	points[i].y = (H.at<double>(1,0)*points[i].x + H.at<double>(1,1)*points[i].y + H.at<double>(1,2)) / w;
	// }
	vector<Point2f> finalpts(4);
	perspectiveTransform(points,finalpts, H);

	// line(img, points[0], points[1], Scalar(0,255,0));
	// line(img, points[1], points[2], Scalar(0,255,0));
	// line(img, points[2], points[3], Scalar(0,255,0));
	// line(img, points[3], points[0], Scalar(0,255,0));
	return boundingRect(finalpts);
}

Mat stitch(Mat obj, Mat scene, Mat H){
	Mat result;
	Size dim;
	Size2f offset;

	Rect bound = getBound(H, TARGET_WIDTH, TARGET_HEIGHT);

	bound.x < 0 ? offset.width  = -bound.x : offset.width  = 0;
	bound.y < 0 ? offset.height = -bound.y : offset.height = 0;

	dim.width  = max(TARGET_WIDTH + (int)offset.width,  bound.width) + max(bound.x,0);
	dim.height = max(TARGET_HEIGHT+ (int)offset.height, bound.height) + max(bound.y,0);

	Mat T = Mat::eye(3,3,CV_64F);
	T.at<double>(0,2)= offset.width;
	T.at<double>(1,2)= offset.height;
	// Important: first transform and then translate, not inverse. (T*H)
	H = T*H;
	warpPerspective(obj,result,H,dim);

	cv::Mat half(result,cv::Rect(offset.width, offset.height, 640, 480));
	scene.copyTo(half);

	return result;
}