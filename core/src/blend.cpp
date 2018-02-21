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

namespace m2d
{

// See description in header file
void Blender::blendSubMosaic(SubMosaic *_sub_mosaic){
    Mat warp_img;
    _sub_mosaic->final_scene.release();
    _sub_mosaic->final_scene = Mat(_sub_mosaic->scene_size, CV_8UC3, Scalar(0,0,0));

    //reverse(_sub_mosaic->frames.begin(), _sub_mosaic->frames.end());
    for (Frame* frame: _sub_mosaic->frames) {

        Mat aux_T = Mat::eye(3, 3, CV_64F);
        aux_T.at<double>(0,2)= -frame->bound_rect.x;
        aux_T.at<double>(1,2)= -frame->bound_rect.y;
        warpPerspective(frame->color , warp_img, aux_T*frame->H, Size(frame->bound_rect.width,
                                                                      frame->bound_rect.height));

        for(Point2f &pt: frame->bound_points[FIRST]){
            pt.x -= frame->bound_rect.x;
            pt.y -= frame->bound_rect.y;
        }
        vector<Point2f> aux_points = frame->bound_points[FIRST];
        reduceRoi(aux_points);
        Point points_array[4] = {aux_points[0],
                                 aux_points[1],
                                 aux_points[2],
                                 aux_points[3],};

        Mat mask(frame->bound_rect.height, frame->bound_rect.width, CV_8UC3, Scalar(0,0,0));
        fillConvexPoly( mask, points_array, 4, Scalar(255,255,255));
        erode( mask, mask, getStructuringElement( MORPH_RECT, Size(7, 7),Point(-1, -1)));

        // frame->bound_rect.x = max(frame->bound_rect.x,0.f);
        // frame->bound_rect.y = max(frame->bound_rect.y,0.f);
        Mat frame_position(_sub_mosaic->final_scene, cv::Rect(frame->bound_rect.x,
                                                              frame->bound_rect.y,
                                                              frame->bound_rect.width,
                                                              frame->bound_rect.height));

        warp_img.copyTo(frame_position, mask);
        // object_position -= _warp_img;
        // object_position += _warp_img;
        mask.release();
    }
}

void Blender::reduceRoi(vector<Point2f> &_points){
    
    // _points[4] correspond to center point
    for (Point2f &corner: _points) {
        corner.x += 2 * ((corner.x - _points[4].x)>0 ? -1 : 1);
        corner.y += 2 * ((corner.y - _points[4].y)>0 ? -1 : 1);
    }

}

}