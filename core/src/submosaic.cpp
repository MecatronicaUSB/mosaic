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

SubMosaic::SubMosaic(){
    n_frames = 0;
    scene_size = Size2f(TARGET_WIDTH, TARGET_HEIGHT);
    avg_H = Mat::eye(3, 3, CV_64F);
}

SubMosaic::~SubMosaic(){
    for (Frame *frame: frames) {
        delete frame;
    }
    final_scene.release();
    avg_H.release();
    neighbors.clear();
}

// See description in header file
SubMosaic* SubMosaic::clone(){
    SubMosaic *new_sub_mosaic = new SubMosaic();
    new_sub_mosaic->final_scene = final_scene.clone();
    new_sub_mosaic->avg_H = avg_H.clone();
    new_sub_mosaic->scene_size = scene_size;
    //new_sub_mosaic->neighbors = neighbors;
    for (Frame *frame: frames) {
        new_sub_mosaic->addFrame(frame->clone());
    }

    return new_sub_mosaic;
}

// See description in header file
void SubMosaic::addFrame(Frame *_frame){
    frames.push_back(_frame);
    last_frame = _frame;
    n_frames++;
}

void SubMosaic::referenceToZero(){

    vector<float> offset = {-frames[0]->bound_rect.y, 0,
                            -frames[0]->bound_rect.x, 0};

    updateOffset(offset);
}

void SubMosaic::computeOffset(){

    float top=TARGET_HEIGHT, bottom=0, left=TARGET_WIDTH, right=0;
    for (Frame *frame: frames) {
        for (Point2f point: frame->bound_points[FIRST]) {
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
    scene_size.width  = right - left + 5;
    scene_size.height = bottom - top + 5;

    vector<float> offset(4);
    offset[TOP]  = -top;
    offset[LEFT] = -left;

    updateOffset(offset);
}

void SubMosaic::updateOffset(vector<float> _total_offset){

    if ( !_total_offset[TOP] && !_total_offset[LEFT] ) {
        return;
    }

    Mat t = Mat::eye(3, 3, CV_64F);
    t.at<double>(0, 2) = _total_offset[LEFT];
    t.at<double>(1, 2) = _total_offset[TOP]; 

    for (Frame *frame: frames){
        frame->setHReference(t);
    }
    avg_H = t * avg_H;
}

// See description in header file
float SubMosaic::calcKeypointsError(Frame *_first, Frame *_second){
    float error=0;
    for (int i=0; i<_first->keypoints_pos[PREV].size(); i++){
        error += getDistance(_first->keypoints_pos[PREV][i],
                             _second->keypoints_pos[NEXT][i]);
        //error += abs(Mat(_first->keypoints_pos[PREV]) - Mat(_second->keypoints_pos[NEXT]))
    }
    return error;
}

// See description in header file
float SubMosaic::calcDistortion(){
    Point2f line[4][2];
    Point2f v1, v2;
    float side[4];
    float cosine[4];
    float frame_dims = (float)TARGET_WIDTH * (float)TARGET_HEIGHT;
    float ratio_dims = (float)TARGET_HEIGHT / (float)TARGET_WIDTH ;
    float o_sides_error=0;
    float c_sides_error=0;
    float angle_error=0;
    float area_error=0;
    float min_ratio=0;
    float tot_error=0;

    for (Frame *frame: frames) {
        for (int i=0; i<4; i++) {
            
            line[i][0] = frame->bound_points[RANSAC][i];
            line[i][1] = frame->bound_points[RANSAC][i<3?i+1:0];
            
            side[i] = getDistance(line[i][0], line[i][1]);
        }
        for (int i=0; i<4; i++) {
            v1.x = abs(line[i][1].x - line[i][0].x);
            v1.y = abs(line[i][1].y - line[i][0].y);
            v2.x = abs(line[i<3?i+1:0][1].x - line[i<3?i+1:0][0].x);
            v2.y = abs(line[i<3?i+1:0][1].y - line[i<3?i+1:0][0].y);
            
            cosine[i] = (v1.x*v2.x + v1.y*v2.y) / (sqrt(v1.x*v1.x + v1.y*v1.y)*sqrt(v2.x*v2.x + v2.y*v2.y));
        }

        o_sides_error = 2 - 0.5*(min(side[0]/side[2], side[2]/side[0]) +
                                 min(side[1]/side[3], side[3]/side[1]));
        
        min_ratio = min(side[0]/side[1], min(side[1]/side[2], min(side[2]/side[3], side[3]/side[0])));
        c_sides_error = 1 - min(min_ratio/ratio_dims, ratio_dims/min_ratio);

        vector<Point2f> auxpts = {frame->bound_points[RANSAC][0],
                                    frame->bound_points[RANSAC][1],
                                    frame->bound_points[RANSAC][2],
                                    frame->bound_points[RANSAC][3]};

        area_error = 1 - min(contourArea(auxpts)/frame_dims,
                              frame_dims/contourArea(auxpts));

        angle_error = pow(max(cosine[0], max(cosine[1], max(cosine[2], cosine[3]))), 5);

        //tot_error += max(o_sides_error, max(c_sides_error, max(area_error, angle_error)));
        tot_error += o_sides_error + c_sides_error + area_error + angle_error;
    }

    return tot_error;
}

// See description in header file
void SubMosaic::correct(){
    float temp_distortion=0, distortion=0;
    Mat temp_avg_H;
    avg_H = Mat::eye(3, 3, CV_64F);

    for (Frame *frame: frames) {
        frame->gray.release();
        frame->bound_points[SECOND] = frame->bound_points[FIRST];
    }
    
    distortion = calcDistortion();

    cout << distortion << endl << endl;

    for (int i=1; i<frames.size(); i++) {
        temp_avg_H = frames[i]->H.inv();

        for (Frame *frame: frames) {
            perspectiveTransform(frame->bound_points[FIRST], frame->bound_points[SECOND], temp_avg_H*frame->H);
        }

        temp_distortion = calcDistortion();

        if (temp_distortion < distortion) {
            distortion = temp_distortion;
            avg_H = temp_avg_H;
        }
    }

    for (Frame *frame: frames) {
        //frame->H = frame->H * avg_H;
        frame->setHReference(avg_H);
    }
}

// See description in header file
Point2f SubMosaic::getCentroid(){
    Point2f centroid(0, 0);

    int n_points=0;
    for (Frame *frame: frames) {
        for (const Point2f point: frame->bound_points[FIRST]) {
            centroid += point;
            n_points++;
        }
    }

    centroid /= n_points;

    return centroid;
}

// See description in header file
bool SubMosaic::isEmpty(){
    return n_frames==0 ? true : false;
}

// See description in header file
void saveHomographyData(Mat _h, vector<KeyPoint> keypoints[2], std::vector<DMatch> matches){
    ofstream file;
    file.open("homography-data.txt");

    for (int i=0; i<_h.cols; i++) {
        for (int j=0; j<_h.rows; j++) {
            file << _h.at<double>(i, j);
            file << " ";
        }
        file << "\n";
    }
    file << matches.size() << "\n";
    for (auto m: matches) {
        file << keypoints[0][m.queryIdx].pt.x << " ";
        file << keypoints[0][m.queryIdx].pt.y << "\n";
    }
    for (auto m: matches) {
        file << keypoints[1][m.trainIdx].pt.x << " ";
        file << keypoints[1][m.trainIdx].pt.y << "\n";
    }

    file.close();
}

}

