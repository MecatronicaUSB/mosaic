/**
 * @file mosaic.cpp
 * @brief Implementation of stitch class and Mosaic2d Namespace functions 
 * @version 0.2
 * @date 10/02/2018
 * @author Victor Garcia
 */

#include "../include/mosaic.hpp"

using namespace std;
using namespace cv;

namespace m2d //!< mosaic 2d namespace
{

// See description in header file
Frame::Frame(Mat _img, bool _key, int _width, int _height){
    //if(_img.size().width != _width || _img.size().height != _height)
    resize(_img, _img, Size(_width, _height));
    
    color = _img.clone();
    cvtColor(color, gray, CV_BGR2GRAY);

    bound_rect = Rect2f(0, 0, _width, _height);
    // corner points
	bound_points.push_back(Point2f(0, 0));
	bound_points.push_back(Point2f(_width, 0));
	bound_points.push_back(Point2f(_width, _height));
	bound_points.push_back(Point2f(0, _height));
    // center point
    bound_points.push_back(Point2f(_width/2, _height/2));

    H = Mat::eye(3, 3, CV_64F);

    key = _key;
}

// See description in header file
void Frame::trackKeypoints(){
    perspectiveTransform(keypoints_pos[NEXT], keypoints_pos[NEXT], H);
}

// See description in header file
float Frame::getDistance(Point2f _pt1, Point2f _pt2){
    return sqrt(pow((_pt1.x - _pt2.x),2) + pow((_pt1.y - _pt2.y),2));
}

// See description in header file
bool Frame::isGoodFrame(){
    float deformation, area, keypoints_area;
    float semi_diag[4], ratio[2];

    for(int i=0; i<4; i++){
        // 5th point correspond to center of image
        // Getting the distance between corner points to the center (all semi diagonal distances)
        semi_diag[i] = getDistance(bound_points[i], bound_points[4]);
    }
    // ratio beween semi diagonals
    ratio[0] = max(semi_diag[0]/semi_diag[2], semi_diag[2]/semi_diag[0]);
    ratio[1] = max(semi_diag[1]/semi_diag[3], semi_diag[3]/semi_diag[1]);

    // Area of distorted images
    area = contourArea(bound_points);

    // enclosing area with good keypoints
    keypoints_area = boundAreaKeypoints();

    // 3 initial threshold value, must be ajusted in future tests 
    if(area > 3*color.cols*color.rows)
        return false;
    // 4 initial threshold value, must be ajusted in future tests 
    if(ratio[0] > 4 || ratio[1] > 4)
        return false;

    if(keypoints_area < 0.2*color.cols*color.rows)
        return false;

    return true;
}

// See description in header file
float Frame::boundAreaKeypoints(){
    vector<Point2f> hull;

    convexHull(keypoints_pos[PREV], hull);

    return contourArea(hull);
}

// See description in header file
void SubMosaic::setRerenceFrame(Mat _scene){
    frames.push_back(new Frame(_scene.clone(), true));
    n_frames++;
    key_frame = frames[0];
    final_scene = frames[0]->color.clone();
}

// See description in header file
bool SubMosaic::add2Mosaic(Mat _object){
    if(n_frames == 0){
        setRerenceFrame(_object);
        return true;
    }

    frames.push_back(new Frame(_object.clone()));
    n_frames++;

    return stitcher->stitch(frames[n_frames-1], frames[n_frames-2], final_scene);
}


}

