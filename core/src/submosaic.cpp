/**
 * @file submosaic.cpp
 * @brief Implementation of stitch class and Mosaic2d Namespace functions 
 * @version 0.2
 * @date 10/02/2018
 * @author Victor Garcia
 */

#include "../include/submosaic.hpp"
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
void transformationError(){

}

// See description in header file
void getAvHomography(Mat img1, Mat img2){

}

// See description in header file
Frame::Frame(Mat _img, bool _pre, int _width, int _height){

    if (_img.size().width != _width || _img.size().height != _height)
        resize(_img, _img, Size(_width, _height));
    
    color = _img.clone();
    cvtColor(color, gray, CV_BGR2GRAY);
    if (_pre) {
        imgChannelStretch(gray, gray, 1, 99);
    }

    bound_rect = Rect2f(0, 0, _width, _height);
    // corner points
	bound_points.push_back(Point2f(0, 0));
	bound_points.push_back(Point2f(_width, 0));
	bound_points.push_back(Point2f(_width, _height));
	bound_points.push_back(Point2f(0, _height));
    // center point
    bound_points.push_back(Point2f(_width/2, _height/2));

    H = Mat::eye(3, 3, CV_64F);

}

// See description in header file
void Frame::resetFrame(){
    
    H = Mat::eye(3, 3, CV_64F);
    key = true;    

    bound_points[0] = Point2f(0, 0);
    bound_points[1] = Point2f(color.cols, 0);
    bound_points[2] = Point2f(color.cols, color.rows);
    bound_points[3] = Point2f(0, color.rows);

    bound_points[4] = Point2f(color.cols/2, color.rows/2);
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
    if (area > 1.5*color.cols*color.rows)
        return false;
    // 4 initial threshold value, must be ajusted in future tests 
    if (ratio[0]>1.6 || ratio[1]>1.6)
        return false;
    if (keypoints_area < 0.2*color.cols*color.rows)
        return false;
    cout << "good Frame" << endl;
    return true;
}

// See description in header file
float Frame::boundAreaKeypoints(){
    vector<Point2f> hull;

    convexHull(keypoints_pos[PREV], hull);

    return contourArea(hull);
}

// See description in header file
void SubMosaic::addFrame(Frame *_frame){
    frames.push_back(_frame);
    last_frame = _frame;
    n_frames++;
    if (n_frames == 1) {
        key_frame = _frame;
        key_frame->key = true;
    }
}

void Frame::setHReference(Mat _avg_H){
    perspectiveTransform(bound_points, bound_points, _avg_H);
    bound_rect =  boundingRect(bound_points);
    H = _avg_H * H;
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

void SubMosaic::updateOffset(vector<float> _offset){

    scene_size.width += _offset[LEFT] + _offset[RIGHT];
    scene_size.height += _offset[TOP] + _offset[BOTTOM];

    if ( !_offset[TOP] && !_offset[LEFT] ) {
        return;
    }

    Mat t = Mat::eye(3, 3, CV_64F);
    t.at<double>(0, 2) = _offset[LEFT];
    t.at<double>(1, 2) = _offset[TOP]; 

    for (int i=0; i<frames.size(); i++){
        frames[i]->H = t * frames[i]->H;

        perspectiveTransform(frames[i]->bound_points, frames[i]->bound_points, t);
        // frames[i]->bound_rect = boundingRect(frames[i]->bound_points);
        frames[i]->bound_rect.x += _offset[LEFT];
        frames[i]->bound_rect.y += _offset[TOP];
        
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
            semi_diag[i] = getDistance(frame->bound_points2[i], frame->bound_points2[4]);
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
        frame->bound_points2 = frame->bound_points;
    }
    
    distortion = calcDistortion();

    cout << distortion << endl << endl;

    for (int i=1; i<frames.size(); i++) {
        temp_avg_H = frames[i]->H.inv();

        for (Frame *frame: frames) {
            perspectiveTransform(frame->bound_points, frame->bound_points2, temp_avg_H*frame->H);
        }

        temp_distortion = calcDistortion();
        cout << temp_distortion << endl << endl;

        if (temp_distortion < distortion) {
            distortion = temp_distortion;
            avg_H = temp_avg_H;
        }
    }

    for (Frame *frame: frames) {
        frame->setHReference(avg_H);
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

