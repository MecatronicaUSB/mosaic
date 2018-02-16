/**
 * @file stitch.cpp
 * @brief Implementation of stitch class and Mosaic2d Namespace functions 
 * @version 0.2
 * @date 10/02/2018
 * @author Victor Garcia
 */

#include "../include/stitch.hpp"
#include "../include/utils.h"

using namespace std;
using namespace cv;
using namespace xfeatures2d;

namespace m2d //!< mosaic 2d namespace
{

// See description in header file
Stitcher::Stitcher(bool _grid, bool _pre, int _detector, int _matcher){
    use_grid = _grid;
    apply_pre = _pre;
    cells_div = 10;

    switch( _detector ) {
    case USE_KAZE:
        detector = KAZE::create();
        break;
    case USE_AKAZE:
        detector = AKAZE::create();
        break;
    case USE_SIFT:
        detector = SIFT::create();
        break;
    case USE_SURF:
        detector = SURF::create();
        break;
    }

    switch( _matcher ) {
    case USE_BRUTE_FORCE:
        matcher = BFMatcher::create();
        break;
    case USE_FLANN:
        matcher = FlannBasedMatcher::create();
    }
}

// See description in header file
void Stitcher::setDetector(int _detector){
    switch( _detector ) {
    case USE_KAZE:
        detector = KAZE::create();
        break;
    case USE_AKAZE:
        detector = AKAZE::create();
    }
}

// See description in header file
void Stitcher::setMatcher(int _matcher){
    switch( _matcher ) {
    case USE_BRUTE_FORCE:
        matcher = BFMatcher::create();
        break;
    case USE_FLANN:
        matcher = FlannBasedMatcher::create();
    }
}

// See description in header file
void Stitcher::setScene(Frame *_frame){
    img[SCENE] = _frame;
}

// See description in header file
struct StitchStatus Stitcher::stitch(Frame *_object, Frame *_scene, Size _scene_dims){
	StitchStatus status;

    _object->neighbors.push_back(_scene);
    _scene->neighbors.push_back(_object);
    img[OBJECT] = _object;
    img[SCENE] = _scene;

    if (apply_pre) {
        imgChannelStretch(img[OBJECT]->gray, img[OBJECT]->gray, 1, 99);
        imgChannelStretch(img[SCENE]->gray, img[SCENE]->gray, 1, 99);
    }

    // Detect the keypoints using desired Detector and compute the descriptorss
    keypoints[OBJECT].clear();
    keypoints[SCENE].clear();
    detector->detectAndCompute( img[OBJECT]->gray, Mat(), keypoints[OBJECT], descriptors[OBJECT] );
    detector->detectAndCompute( img[SCENE]->gray, Mat(), keypoints[SCENE], descriptors[SCENE] );

    if (!keypoints[OBJECT].size() || !keypoints[SCENE].size()) {
        cout << "No Key points Found" <<  endl;
        return status;
    }

    // Match the keypoints using Knn
    matches.clear();
    matcher->knnMatch( descriptors[OBJECT], descriptors[SCENE], matches, 2);
    
    // Discard outliers based on distance between descriptor's vectors
    good_matches.clear();
    getGoodMatches();
    
    // Apply grid detector if flag is activated
    if (use_grid) {
        gridDetector();
    }

    // Convert the keypoints into a vector containing the correspond X,Y position in image
    positionFromKeypoints();

    img[SCENE]->trackKeypoints();

    // TODO: implement own findHomography function
    img[OBJECT]->H = findHomography(Mat(img[OBJECT]->keypoints_pos[PREV]), Mat(img[SCENE]->keypoints_pos[NEXT]), CV_RANSAC);

    if (img[OBJECT]->H.empty()) {
        cout << "not enought keypoints to calculate homography matrix. Exiting..." <<  endl;
        return status;
    }

    if (!img[OBJECT]->isGoodFrame()) {
        cout << "Frame too distorted. Exiting..." <<  endl;
        return status;
    }

    status.offset = getWarpOffet(img[OBJECT]->H, _scene_dims);

    status.ok = true;
    return status;
}

// See description in header file
void Stitcher::getGoodMatches(){

    for (vector<DMatch> match: matches) {
        if ((match[0].distance < 0.5 * (match[1].distance)) &&
            ((int) match.size() <= 2 && (int) match.size() > 0)) {
            // take the first result only if its distance is smaller than 0.5*second_best_dist
            // that means this descriptor is ignored if the second distance is bigger or of similar
            good_matches.push_back(match[0]);
        }
    }
}

// See description in header file
void Stitcher::gridDetector(){
    vector<DMatch> grid_matches;
    DMatch best_match;
    int k=0, best_distance = 100;
    int stepx = img[OBJECT]->color.cols / cells_div;
    int stepy = img[OBJECT]->color.rows / cells_div;
    
    for(int i=0; i<10; i++){
        for(int j=0; j<10; j++){
            k=0;
            best_distance = 100;
            for (DMatch match: good_matches) {
                //-- Get the keypoints from the good matches
                if (keypoints[OBJECT][match.queryIdx].pt.x >= stepx*i && keypoints[OBJECT][match.queryIdx].pt.x < stepx*(i+1) &&
                keypoints[OBJECT][match.queryIdx].pt.y >= stepy*j && keypoints[OBJECT][match.queryIdx].pt.y < stepy*(j+1)) {
                    if (match.distance < best_distance) {
                        best_distance = match.distance;
                        best_match = match;
                    }
                    matches.erase(matches.begin() + k);  
                }
                k++;
            }
            if (best_distance != 100)
                grid_matches.push_back(best_match);
        }
    }
    good_matches = grid_matches;
}

// See description in header file
void  Stitcher::positionFromKeypoints(){
    for (DMatch good: good_matches) {
        //-- Get the keypoints from the good matches
        img[OBJECT]->keypoints_pos[PREV].push_back(keypoints[OBJECT][good.queryIdx].pt);
        img[SCENE]->keypoints_pos[NEXT].push_back(keypoints[SCENE][good.trainIdx].pt);
    }
}

// See description in header file
void Stitcher::drawKeipoints(vector<float> _warp_offset, Mat &_final_scene){
    vector<Point2f> aux_kp[2];

    aux_kp[0] = img[SCENE]->keypoints_pos[NEXT];
    for(Point2f &pt: aux_kp[0]){
        pt.x += _warp_offset[LEFT];
        pt.y += _warp_offset[TOP];
    }

    perspectiveTransform(img[OBJECT]->keypoints_pos[PREV], aux_kp[1], img[OBJECT]->H);
    for(int j=0; j<aux_kp[0].size(); j++){
        circle(_final_scene, aux_kp[0][j], 3, Scalar(255, 0, 0), -1);
        circle(_final_scene, aux_kp[1][j], 3, Scalar(0, 0, 255), -1);
    }

    cout << min(Mat(Point2f(5,5)), abs(Mat(aux_kp[0]) - Mat(aux_kp[1]))) << endl;

}

// See description in header file
vector<float> Stitcher::getWarpOffet(Mat _H, Size _scene_dims){
    vector<float> warp_offset(4);
    Rect2f aux_rect;
    perspectiveTransform(img[OBJECT]->bound_points, img[OBJECT]->bound_points, _H);
    aux_rect = boundingRect(img[OBJECT]->bound_points);

    warp_offset[TOP] = max(0.f,-aux_rect.y);
    warp_offset[BOTTOM] = max(0.f, aux_rect.y + aux_rect.height - _scene_dims.height);
    warp_offset[LEFT] = max(0.f,-aux_rect.x);
    warp_offset[RIGHT] = max(0.f, aux_rect.x + aux_rect.width - _scene_dims.width);
    
    img[OBJECT]->bound_rect = aux_rect;

    return warp_offset;
}

}