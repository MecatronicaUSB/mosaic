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
    new_sub_mosaic->distortion = distortion;
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
        if (frame->bound_rect.x < left)
            left = frame->bound_rect.x;
        if (frame->bound_rect.y < top)
            top = frame->bound_rect.y;
        if (frame->bound_rect.x + frame->bound_rect.width > right)
            right = frame->bound_rect.x + frame->bound_rect.width;
        if (frame->bound_rect.y + frame->bound_rect.height > bottom)
            bottom = frame->bound_rect.y + frame->bound_rect.height;
    }
    scene_size.width  = right - left;
    scene_size.height = bottom - top;

    vector<float> offset = {max(-top,0.f), 0, max(-left,0.f), 0};
    updateOffset(offset);
}

void SubMosaic::updateOffset(vector<float> _total_offset){

    if ( !_total_offset[TOP] && !_total_offset[LEFT] ) {
        return;
    }

    Mat t = Mat::eye(3, 3, CV_64F);
    t.at<double>(0, 2) = _total_offset[LEFT];
    t.at<double>(1, 2) = _total_offset[TOP]; 

    for (int i=0; i<frames.size(); i++){
        frames[i]->H = t * frames[i]->H;

        perspectiveTransform(frames[i]->bound_points[FIRST], frames[i]->bound_points[FIRST], t);
        // frames[i]->bound_rect = boundingRect(frames[i]->bound_points[FIRST]);
        frames[i]->bound_rect.x += _total_offset[LEFT];
        frames[i]->bound_rect.y += _total_offset[TOP];
        
        if (frames[i]->keypoints_pos[PREV].size())
            perspectiveTransform(frames[i]->keypoints_pos[PREV], frames[i]->keypoints_pos[PREV], t);
        if (frames[i]->keypoints_pos[NEXT].size())        
            perspectiveTransform(frames[i]->keypoints_pos[NEXT], frames[i]->keypoints_pos[NEXT], t); 
    }
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
    float semi_diag[4], ratio[2];
    float distortion=0;
    for (Frame *frame: frames) {
        for (int i=0; i<4; i++) {
            // 5th point correspond to center of image
            // Getting the distance between corner points to the center (all semi diagonal distances)
            semi_diag[i] = getDistance(frame->bound_points[SECOND][i], frame->bound_points[SECOND][4]);
        }
        // ratio beween semi diagonals
        ratio[0] = max(semi_diag[0]/semi_diag[2], semi_diag[2]/semi_diag[0]);
        ratio[1] = max(semi_diag[1]/semi_diag[3], semi_diag[3]/semi_diag[1]);
        distortion += (ratio[0] + ratio[1]);
    }

    return distortion;
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

