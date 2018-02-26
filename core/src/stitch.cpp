/**
 * @file stitch.cpp
 * @brief Implementation of Stitcher class functions 
 * @version 0.2
 * @date 10/02/2018
 * @author Victor Garcia
 */

#include "../include/stitch.hpp"

using namespace std;
using namespace cv;
using namespace xfeatures2d;

namespace m2d //!< mosaic 2d namespace
{

// See description in header file
Stitcher::Stitcher(bool _grid, int _detector, int _matcher){
    use_grid = _grid;
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
        default:
            detector = KAZE::create();
            break; 
    }

    switch( _matcher ) {
        case USE_BRUTE_FORCE:
            matcher = BFMatcher::create();
            break;
        case USE_FLANN:
            matcher = FlannBasedMatcher::create();
            break;
        default:
            matcher = FlannBasedMatcher::create();
            break;
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
int Stitcher::stitch(Frame *_object, Frame *_scene, Size _scene_dims){

    img[OBJECT] = _object;
    img[SCENE] = _scene;

    if (!img[SCENE]->haveKeypoints()) {
        detector->detectAndCompute(img[SCENE]->gray, Mat(), img[SCENE]->keypoints,
                                                            img[SCENE]->descriptors);
    }
    detector->detectAndCompute(img[OBJECT]->gray, Mat(), img[OBJECT]->keypoints,
                                                         img[OBJECT]->descriptors);

    if (!img[SCENE]->haveKeypoints()) {
        cout << "No Key points Found. exiting" <<  endl;
        return NO_KEYPOINTS;
    }

    // Match the keypoints using Knn
    vector<vector<DMatch> > aux_matches;

    matcher->knnMatch( img[OBJECT]->descriptors, img[SCENE]->descriptors, aux_matches, 2);
    matches.push_back(aux_matches);
    
    for (Frame *neighbor: img[SCENE]->neighbors) {
        aux_matches.clear();
        matcher->knnMatch(img[OBJECT]->descriptors, neighbor->descriptors, aux_matches, 2);
        matches.push_back(aux_matches);
    }

    float thresh = 0.8;
    bool good_thresh = true;
    while (good_thresh) {

        // Discard outliers based on distance between descriptor's vectors
        good_matches.clear();
        getGoodMatches(thresh);
        
        // Apply grid detector if flag is activated
        if (use_grid) {
            gridDetector();
        }
        // Convert the keypoints into a vector containing the correspond X,Y position in image
        positionFromKeypoints();

        if (points_pos[OBJECT].rows>4 && points_pos[SCENE].rows>4) {
            good_thresh = false;
        } else {
            thresh += 0.1;
            for (int i=0; i<img[SCENE]->neighbors.size(); i++) {
                good_matches.pop_back();
            }
            neighbors_kp.clear();
            img[OBJECT]->keypoints_pos[PREV].clear();
            img[SCENE]->keypoints_pos[NEXT].clear();
        }
    }

    Mat H = findHomography(points_pos[OBJECT], points_pos[SCENE], CV_RANSAC);
    // Mat H = estimateRigidTransform(points_pos[OBJECT], points_pos[SCENE], true);
    // Mat t = (Mat1d(1,3) << 0, 0, 1);
    // vconcat(H, t, H);
    // H.at<double>(2, 2) = 1;

    if (H.empty()) {
        cout << "Not enought keypoints to calculate homography matrix. Exiting..." <<  endl;
        return NO_HOMOGRAPHY;
    }

    img[OBJECT]->setHReference(H);

    if (!img[OBJECT]->isGoodFrame()) {
        cout << "Frame too distorted. Creating new Sub-Mosaic..." <<  endl;

        cleanNeighborsData();
        return BAD_DISTORTION;
    }

    updateNeighbors();

    return OK;
}

void Stitcher::cleanNeighborsData(){
    for (int i=0; i<img[SCENE]->neighbors.size(); i++) {
        good_matches.pop_back();
    }
    neighbors_kp.clear();
    matches.clear();
}

void Stitcher::updateNeighbors(){
    img[OBJECT]->neighbors.push_back(img[SCENE]);
    for (int i=0; i<img[SCENE]->neighbors.size(); i++) {
        if (good_matches[i+1].size() > 4) {
            img[OBJECT]->neighbors.push_back(img[SCENE]->neighbors[i]);
        }
    }
    cleanNeighborsData();
}

// See description in header file
void Stitcher::getGoodMatches(float _thresh){
    vector<DMatch> aux_matches;
    for (int i=0; i<img[SCENE]->neighbors.size()+1; i++){
        aux_matches.clear();
        for (vector<DMatch> match: matches[i]) {
            if ((match[0].distance < _thresh * (match[1].distance)) &&
                ((int) match.size() <= 2 && (int) match.size() > 0)) {
                // take the first result only if its distance is smaller than threshold*second_best_dist
                // that means this descriptor is ignored if the second distance is bigger or of similar
                aux_matches.push_back(match[0]);
            }
        }
        good_matches.push_back(aux_matches);
    }
}

// See description in header file
void Stitcher::gridDetector(){
    vector<vector<DMatch> > grid_matches;
    DMatch best_match;
    int m=0, best_distance = 100;
    int stepx = img[OBJECT]->color.cols / cells_div;
    int stepy = img[OBJECT]->color.rows / cells_div;
    int index = 0;

    grid_matches = good_matches;
    for (vector<DMatch> &aux: grid_matches) {
        aux.clear();
    }

    for (int i=0; i<10; i++) {
        for (int j=0; j<10; j++) {
            best_distance = 100;
            index=0;
            for (int k=0; k<img[SCENE]->neighbors.size()+1; k++) {
                m=0;
                for (DMatch match: good_matches[k]) {
                    //-- Get the keypoints from the good matches
                    if (img[OBJECT]->keypoints[match.queryIdx].pt.x >= stepx*i && img[OBJECT]->keypoints[match.queryIdx].pt.x < stepx*(i+1) &&
                        img[OBJECT]->keypoints[match.queryIdx].pt.y >= stepy*j && img[OBJECT]->keypoints[match.queryIdx].pt.y < stepy*(j+1)) {
                        if (match.distance < best_distance) {
                            best_distance = match.distance;
                            best_match = match;
                            index = k;
                        }
                        good_matches[k].erase(good_matches[k].begin() + m);
                    }
                    m++;
                }
            }
            if (best_distance != 100)
                grid_matches[index].push_back(best_match);
        }
    }
    good_matches = grid_matches;

}

// See description in header file
void  Stitcher::positionFromKeypoints(){

    for (DMatch good: good_matches[0]) {
        //-- Get the keypoints from the good matches
        img[OBJECT]->keypoints_pos[PREV].push_back(img[OBJECT]->keypoints[good.queryIdx].pt);
        img[SCENE]->keypoints_pos[NEXT].push_back(img[SCENE]->keypoints[good.trainIdx].pt);
    }

    vector<Point2f> aux_points;

    for (int i=0; i<img[SCENE]->neighbors.size(); i++) {
        
        aux_points.clear();
        for (DMatch good: good_matches[i+1]) {
            img[OBJECT]->keypoints_pos[PREV].push_back(img[OBJECT]->keypoints[good.queryIdx].pt);
            aux_points.push_back(img[SCENE]->neighbors[i]->keypoints[good.trainIdx].pt);
        }
        neighbors_kp.push_back(aux_points);
    }

    trackKeypoints();

    points_pos[OBJECT] = Mat(img[OBJECT]->keypoints_pos[PREV]);
    points_pos[SCENE] = Mat(img[SCENE]->keypoints_pos[NEXT]);

    for (int i=0; i<img[SCENE]->neighbors.size(); i++) {
        vconcat(points_pos[SCENE], Mat(neighbors_kp[i]), points_pos[SCENE]);
    }
}

// See description in header file
void Stitcher::trackKeypoints(){
    if (img[SCENE]->keypoints_pos[NEXT].size() >0)
        perspectiveTransform(img[SCENE]->keypoints_pos[NEXT], img[SCENE]->keypoints_pos[NEXT],
                             img[SCENE]->H);

    for (int i=0; i<img[SCENE]->neighbors.size(); i++) {
        if (neighbors_kp[i].size()>0)
            perspectiveTransform(neighbors_kp[i], neighbors_kp[i],
                                 img[SCENE]->neighbors[i]->H);
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

}