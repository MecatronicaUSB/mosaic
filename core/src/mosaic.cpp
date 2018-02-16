/**
 * @file mosaic.cpp
 * @brief Implementation of stitch class and Mosaic2d Namespace functions 
 * @version 0.2
 * @date 10/02/2018
 * @author Victor Garcia
 */

#include "../include/mosaic.hpp"
#include "../include/stitch.hpp"

using namespace std;
using namespace cv;

namespace m2d //!< mosaic 2d namespace
{

// See description in header file
float getDistance(Point2f _pt1, Point2f _pt2){
    return sqrt(pow((_pt1.x - _pt2.x),2) + pow((_pt1.y - _pt2.y),2));
}

// See description in header file
Point2f getMiddlePoint(Point2f _pt1, Point2f _pt2){
    Point2f middle;

    return middle;
}

// See description in header file
void transformationError(){

}

// See description in header file
void getAvHomography(Mat img1, Mat img2){

}

// See description in header file
Frame::Frame(Mat _img, bool _key, int _width, int _height){
    if (_img.size().width != _width || _img.size().height != _height)
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
bool Frame::isGoodFrame(){
    float deformation, area, keypoints_area;
    float semi_diag[4], ratio[2];

    for (int i=0; i<4; i++) {
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
    if (area > 3*color.cols*color.rows)
        return false;
    // 4 initial threshold value, must be ajusted in future tests 
    if (ratio[0]>4 || ratio[1]>4)
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

// See description in header file
void SubMosaic::setRerenceFrame(Frame *_frame){
    key_frame = _frame;
    key_frame->key = true;
    //final_scene = _frame->color.clone();
}

// See description in header file
void SubMosaic::addFrame(Frame *_frame){
    frames.push_back(_frame);
    n_frames++;
    if (n_frames == 1) {
        setRerenceFrame(frames[0]);
        return;
    }
}

void SubMosaic::updateOffset(Size2f _size){
    Mat t = Mat::eye(3, 3, CV_64F);
    t.at<double>(0, 2) = _size.width;
    t.at<double>(1, 2) = _size.height; 

    for (int i=0; i<frames.size(); i++){
        frames[i]->H = t * frames[i]->H;

        perspectiveTransform(frames[i]->bound_points, frames[i]->bound_points, t);
        // frames[i]->bound_rect = boundingRect(frames[i]->bound_points);
        frames[i]->bound_rect.x += _size.width;
        frames[i]->bound_rect.y += _size.height;
        
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
    }
    return error;
}

// See description in header file
void SubMosaic::correct(){
    for (Frame *frame: frames) {
        frame->gray.release();
    }
    // for (int i=0; i<frames.size()-1; i++) {
    //     distortion += calcKeypointsError(frames[n_frames], frames[n_frames+1]);
    // }
    
}

// See description in header file
Mosaic::Mosaic(){
    n_subs = 0;
    tot_frames = 0;
    sub_mosaics.push_back(new SubMosaic());
}

// See description in header file
void Mosaic::addFrame(Mat _object){
    tot_frames++;

    Frame *aux_frame = new Frame(_object.clone());
    if (sub_mosaics[n_subs]->n_frames == 0) {
        sub_mosaics[n_subs]->addFrame(aux_frame);
        return;
    }

    struct StitchStatus status;

    status =  stitcher->stitch(aux_frame,
                               sub_mosaics[n_subs]->frames[sub_mosaics[n_subs]->n_frames-1],
                               sub_mosaics[n_subs]->size);

    if (status.ok && (sub_mosaics[n_subs]->n_frames<5)) {

        sub_mosaics[n_subs]->addFrame(aux_frame);

        sub_mosaics[n_subs]->size.width += status.offset[LEFT] + status.offset[RIGHT];
        sub_mosaics[n_subs]->size.height += status.offset[TOP] + status.offset[BOTTOM];

        if ( status.offset[LEFT] || status.offset[TOP]) {
            sub_mosaics[n_subs]->updateOffset(Size(status.offset[LEFT], status.offset[TOP]));
        }
    } else {
        sub_mosaics[n_subs]->correct();
        sub_mosaics[n_subs]->is_complete = true;
        sub_mosaics[n_subs]->final_scene = Mat(sub_mosaics[n_subs]->size, CV_8UC3, Scalar(0,0,0));

        blender->blendSubMosaic(sub_mosaics[n_subs]);
        imshow("Blend", sub_mosaics[n_subs]->final_scene);
        waitKey(0);
        
        sub_mosaics.push_back(new SubMosaic());
        n_subs++;
        sub_mosaics[n_subs]->addFrame(new Frame(_object.clone()));
    }
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

