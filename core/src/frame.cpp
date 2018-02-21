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
Frame::Frame(Mat _img, bool _pre, int _width, int _height){

    bound_points = vector<vector<Point2f> >(3);
    keypoints_pos = vector<vector<Point2f> >(2);

    if (_img.size().width != _width || _img.size().height != _height)
        resize(_img, _img, Size(_width, _height));
    
    color = _img.clone();
    cvtColor(color, gray, CV_BGR2GRAY);
    if (_pre) {
        imgChannelStretch(gray, gray, 1, 99);
    }

    bound_rect = Rect2f(0, 0, _width, _height);
    // corner points
	bound_points[FIRST].push_back(Point2f(0, 0));
	bound_points[FIRST].push_back(Point2f(_width, 0));
	bound_points[FIRST].push_back(Point2f(_width, _height));
	bound_points[FIRST].push_back(Point2f(0, _height));
    // center point
    bound_points[FIRST].push_back(Point2f(_width/2, _height/2));

    H = Mat::eye(3, 3, CV_64F);

}

Frame::~Frame(){
    H.release();
    gray.release();
    color.release();
    descriptors.release();
    keypoints_pos.clear();
    bound_points.clear();
    keypoints.clear();
    neighbors.clear();
}

Frame* Frame::clone(){
    Frame* new_frame;
    new_frame->key = key;
    new_frame->frame_error = frame_error;        
    new_frame->descriptors = descriptors.clone();
    new_frame->color = color.clone();
    new_frame->gray = gray.clone();
    new_frame->H = H.clone();
    new_frame->bound_rect = bound_rect;
    new_frame->bound_points = bound_points;
    new_frame->keypoints_pos = keypoints_pos;
    new_frame->keypoints = keypoints;
    new_frame->neighbors = neighbors;

    return new_frame;
}

// See description in header file
void Frame::resetFrame(){
    
    H = Mat::eye(3, 3, CV_64F);
    key = true;    

    bound_points[FIRST][0] = Point2f(0, 0);
    bound_points[FIRST][1] = Point2f(color.cols, 0);
    bound_points[FIRST][2] = Point2f(color.cols, color.rows);
    bound_points[FIRST][3] = Point2f(0, color.rows);

    bound_points[FIRST][4] = Point2f(color.cols/2, color.rows/2);
    bound_rect = Rect2f(0, 0, color.cols, color.rows);
    neighbors.clear();
}

// See description in header file
bool Frame::isGoodFrame(){
    float deformation, area, keypoints_area;
    float semi_diag[4], ratio[2];
    
    for (int i=0; i<4; i++) {
        // 5th point correspond to center of image
        // Getting the distance between corner points to the center (all semi diagonal distances)
        semi_diag[i] = getDistance(bound_points[FIRST][i], bound_points[FIRST][4]);
    }
    // ratio beween semi diagonals
    ratio[0] = max(semi_diag[0]/semi_diag[2], semi_diag[2]/semi_diag[0]);
    ratio[1] = max(semi_diag[1]/semi_diag[3], semi_diag[3]/semi_diag[1]);

    // Area of distorted images
    area = contourArea(bound_points[FIRST]);

    // enclosing area with good keypoints
    keypoints_area = boundAreaKeypoints();

    // 3 initial threshold value, must be ajusted in future tests 
    if (area > 1.5*color.cols*color.rows)
        return false;
    // 4 initial threshold value, must be ajusted in future tests 
    if (ratio[0]>1.6 || ratio[1]>1.6)
        return false;
    if (keypoints_area < 0.2*color.cols*color.rows)
        return false;
        
    return true;
}

// See description in header file
float Frame::boundAreaKeypoints(){
    vector<Point2f> hull;

    convexHull(keypoints_pos[PREV], hull);

    return contourArea(hull);
}

void Frame::setHReference(Mat _H){
    perspectiveTransform(bound_points[FIRST], bound_points[FIRST], _H);
    bound_rect =  boundingRect(bound_points[FIRST]);
    H = _H * H;
}

bool Frame::haveKeypoints(){
    return keypoints.size() > 0 ? true : false;
}


}