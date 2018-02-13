/**
 * @file stitch.cpp
 * @brief Implementation of stitch class and Mosaic2d Namespace functions 
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
void saveHomographyData(Mat H, vector<KeyPoint> keypoints[2], std::vector<DMatch> matches){
    ofstream file;
    file.open("homography-data.txt");

    for(int i=0; i<H.cols; i++){
        for(int j=0; j<H.rows; j++){
            file << H.at<double>(i, j);
            file << " ";
        }
        file << "\n";
    }
    file << matches.size() << "\n";
    for(auto m: matches){
        file << keypoints[0][m.queryIdx].pt.x << " ";
        file << keypoints[0][m.queryIdx].pt.y << "\n";
    }
    for(auto m: matches){
        file << keypoints[1][m.trainIdx].pt.x << " ";
        file << keypoints[1][m.trainIdx].pt.y << "\n";
    }

    file.close();
}

// See description in header file
float getDistance(Point2f pt1, Point2f pt2){
    return sqrt(pow((pt1.x - pt2.x),2) + pow((pt1.y - pt2.y),2));
}

// See description in header file
Stitcher::Stitcher(bool _grid, bool _pre, int _width, int _height, int _detector, int _matcher){
    use_grid = _grid;
    apply_pre = _pre;
    frame_size.width = TARGET_WIDTH;
    frame_size.height = TARGET_HEIGHT;
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

    // corner points
	border_points.push_back(Point2f(0,0));
	border_points.push_back(Point2f(frame_size.width,0));
	border_points.push_back(Point2f(frame_size.width, frame_size.height));
	border_points.push_back(Point2f(0, frame_size.height));
    // center point
    border_points.push_back(Point2f(frame_size.width/2, frame_size.height/2));

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
void Stitcher::setScene(Mat _scene){
    old_scene = false;

    img[SCENE_COLOR] = _scene.clone();
    resize(img[SCENE_COLOR], img[SCENE], Size(frame_size.width, frame_size.height));

    bound_rect = Rect(0, 0, frame_size.width, frame_size.height);
}

// See description in header file
bool Stitcher::stitch(Mat _object){
    matches.clear();
    good_matches.clear();

	Mat warp_img;
	vector<float> warp_offset;
    img[OBJECT_COLOR] = _object.clone();
    if(old_scene){
        img[SCENE] = img[OLD_OBJECT].clone();
    }

    img[OLD_OBJECT] = _object.clone();

    resize(img[OBJECT_COLOR], img[OBJECT_COLOR], Size(frame_size.width, frame_size.height));

    // Conver object image to gray
    cvtColor(img[SCENE], img[SCENE], COLOR_BGR2GRAY);
    cvtColor(img[OBJECT_COLOR], img[OBJECT], COLOR_BGR2GRAY);

    // Apply preprocessing (SCB) if the flag is activated
    if(apply_pre){
        imgChannelStretch(img[OBJECT], img[OBJECT], 1, 99);
        imgChannelStretch(img[SCENE], img[SCENE], 1, 99);
    }

    // Detect the keypoints using desired Detector and compute the descriptors
    detector->detectAndCompute( img[OBJECT], Mat(), keypoints[OBJECT], descriptors[OBJECT] );
    detector->detectAndCompute( img[SCENE], Mat(), keypoints[SCENE], descriptors[SCENE] );

    if(!keypoints[OBJECT].size() || !keypoints[SCENE].size()){
        cout << "No Key points Found" <<  endl;
        return false;
    }

    // Match the keypoints using Knn
    matcher->knnMatch( descriptors[OBJECT], descriptors[SCENE], matches, 2);
    
    // Discard outliers based on distance between descriptor's vectors
    getGoodMatches();
    
    // Apply grid detector if flag is activated
    if(use_grid){
        gridDetector();
    }

    // Convert the keypoints into a vector containing the correspond X,Y position in image
    positionFromKeypoints();

    if(old_scene){
        perspectiveTransform(keypoints_coord[SCENE], keypoints_coord[SCENE], old_H);
    }

    // TODO: implement own findHomography function
    H = findHomography(Mat(keypoints_coord[OBJECT]), Mat(keypoints_coord[SCENE]), CV_RANSAC);

    if(H.empty()){
        cout << "not enought keypoints to calculate homography matrix. Exiting..." <<  endl;
        return false;
    }

    warp_offset = getWarpOffet();

    // if(!goodframe()){
    //     cout << "New frame is too much distorted. Exiting..." <<  endl;
    //     return false;
    // }

	warpPerspective(img[OBJECT_COLOR], warp_img, H, Size(bound_rect.width, bound_rect.height));

    // Create padd in scene based on offset in transformed image. when transformed image moves 
    // to negatives values a new padding is created in unexisting sides to achieve a correct stitch
    copyMakeBorder(img[SCENE_COLOR], img[SCENE_COLOR], warp_offset[TOP], warp_offset[BOTTOM],
                                                       warp_offset[LEFT], warp_offset[RIGHT],
                                                       BORDER_CONSTANT,Scalar(0,0,0));

    blendToScene(warp_img);
    //drawKeipoints();
    old_scene = true;
    cleanData();
    return true;
}

// See description in header file
bool Stitcher::goodframe(){
    float deformation, area, keypoints_area;
    float semi_diag[4], ratio[2];

    for(int i=0; i<4; i++){
        // 5th point correspond to center of image
        // Getting the distance between corner points to the center (all semi diagonal distances)
        semi_diag[i] = getDistance(warp_points[i], warp_points[4]);
    }
    // ratio beween semi diagonals
    ratio[0] = max(semi_diag[0]/semi_diag[2], semi_diag[2]/semi_diag[0]);
    ratio[1] = max(semi_diag[1]/semi_diag[3], semi_diag[3]/semi_diag[1]);

    // Area of distorted images
    area = contourArea(warp_points);

    // enclosing area with good keypoints
    keypoints_area = boundAreaKeypoints();

    // 3 initial threshold value, must be ajusted in future tests 
    if(area > 3*frame_size.width*frame_size.height)
        return false;
    // 4 initial threshold value, must be ajusted in future tests 
    if(ratio[0] > 4 || ratio[1] > 4)
        return false;

    return true;
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
    int stepx = frame_size.width / cells_div;
    int stepy = frame_size.height/ cells_div;
    
    for(int i=0; i<10; i++){
        for(int j=0; j<10; j++){
            k=0;
            best_distance = 100;
            for (DMatch match: good_matches) {
                //-- Get the keypoints from the good matches
                if(keypoints[OBJECT][match.queryIdx].pt.x >= stepx*i && keypoints[OBJECT][match.queryIdx].pt.x < stepx*(i+1) &&
                keypoints[OBJECT][match.queryIdx].pt.y >= stepy*j && keypoints[OBJECT][match.queryIdx].pt.y < stepy*(j+1)){
                    if(match.distance < best_distance){
                        best_distance = match.distance;
                        best_match = match;
                    }
                    matches.erase(matches.begin() + k);  
                }
                k++;
            }
            if(best_distance != 100)
                grid_matches.push_back(best_match);
        }
    }
    good_matches = grid_matches;
}

// See description in header file
float Stitcher::boundAreaKeypoints(){
    vector<Point2f> points, hull;
    for(DMatch match: good_matches){
        points.push_back(keypoints[OBJECT][match.queryIdx].pt);
    }
    convexHull(points, hull);

    return contourArea(hull);
}

// See description in header file
void  Stitcher::positionFromKeypoints(){
    for (DMatch good: good_matches) {
        //-- Get the keypoints from the good matches
        keypoints_coord[OBJECT].push_back(keypoints[OBJECT][good.queryIdx].pt);
        keypoints_coord[SCENE].push_back(keypoints[SCENE][good.trainIdx].pt);
    }
}

// See description in header file
void Stitcher::drawKeipoints(){
    vector<Point2f> aux_kp[2];
    aux_kp[0] = keypoints_coord[SCENE];
    //perspectiveTransform(keypoints_coord[OBJECT], aux_kp[1], old_H);
    img[SCENE_KEYPOINTS] = img[SCENE_COLOR].clone();
    for(int j=0; j<aux_kp[0].size(); j++){
        circle(img[SCENE_KEYPOINTS], aux_kp[0][j], 3, Scalar(255, 0, 0), -1);
        circle(img[SCENE_KEYPOINTS], aux_kp[1][j], 3, Scalar(0, 0, 255), -1);
    }

}

// See description in header file
vector<float> Stitcher::getWarpOffet(){
    vector<float> warp_offset(4);

    perspectiveTransform(border_points, warp_points, H);
    bound_rect = boundingRect(warp_points);

    warp_offset[TOP] = max(0.f,-bound_rect.y);
    warp_offset[BOTTOM] = max(0.f, bound_rect.y + bound_rect.height - img[SCENE_COLOR].rows);
    warp_offset[LEFT] = max(0.f,-bound_rect.x);
    warp_offset[RIGHT] = max(0.f, bound_rect.x + bound_rect.width - img[SCENE_COLOR].cols);

    Mat old_T = Mat::eye(3,3,CV_64F);
    old_T.at<double>(0,2)= warp_offset[LEFT];
	old_T.at<double>(1,2)= warp_offset[TOP];
    old_H = old_T*H;

	Mat T = Mat::eye(3,3,CV_64F);
	T.at<double>(0,2)= -bound_rect.x;
	T.at<double>(1,2)= -bound_rect.y;

	// Important: first transform and then translate, not inverse way. correct form: (T*H)
	H = T*H;

    return warp_offset;
}

// See description in header file
void Stitcher::blendToScene(Mat _warp_img){

    for(Point2f& pt: warp_points){
        pt.x -= bound_rect.x;
        pt.y -= bound_rect.y;
    }

    Point points_array[4] = {warp_points[0], warp_points[1], warp_points[2], warp_points[3]};

    Mat mask(bound_rect.height, bound_rect.width, CV_8UC3, Scalar(0,0,0));
    fillConvexPoly( mask, points_array, 4, Scalar(255,255,255));
    erode( mask, mask, getStructuringElement( MORPH_RECT, Size(7, 7),Point(-1, -1)));
    bound_rect.x = max(bound_rect.x,0.f);
    bound_rect.y = max(bound_rect.y,0.f);
	cv::Mat object_position(img[SCENE_COLOR], cv::Rect(bound_rect.x,
                                                       bound_rect.y,
                                                       bound_rect.width,
                                                       bound_rect.height));

    _warp_img.copyTo(object_position, mask);
    // object_position -= _warp_img;
    // object_position += _warp_img;
    mask.release();
}

// See description in header file
void Stitcher::cleanData(){
        // matches.clear();
        // good_matches.clear();
        img[OBJECT].release();
        img[OBJECT_COLOR].release();
        keypoints[OBJECT].clear();
        keypoints[SCENE].clear();
        keypoints_coord[OBJECT].clear();
        keypoints_coord[SCENE].clear();
        descriptors[OBJECT].release();
        descriptors[SCENE].release();
}

}