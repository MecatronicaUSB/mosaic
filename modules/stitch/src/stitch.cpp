#include "../include/stitch.h"
#include <cmath> 

using namespace std;
using namespace cv;

// See description in header file
vector<Point2f> getBoundPoints(Mat H, int width, int height){
	vector<Point2f> points, final_points;
    Rect bound;

	points.push_back(Point2f(0,0));
	points.push_back(Point2f(width,0));
	points.push_back(Point2f(width,height));
	points.push_back(Point2f(0,height));
    // center point
    points.push_back(Point2f(width/2, height/2));

	perspectiveTransform(points,final_points, H);
	return final_points;
}

// See description in header file
Rect stitch(Mat object, Mat& scene, Mat H){
	Mat warped;
    Size dim;
	Size2f offset;

	vector<Point2f> bound_points = getBoundPoints(H, object.cols, object.rows);
    Rect bound_rect =  boundingRect(bound_points);
	bound_rect.x < 0 ? offset.width  = -bound_rect.x : offset.width  = 0;
	bound_rect.y < 0 ? offset.height = -bound_rect.y : offset.height = 0;

    dim.width  = scene.cols + abs(bound_rect.x);
    dim.height = scene.rows + abs(bound_rect.y);

	Mat T = Mat::eye(3,3,CV_64F);
	T.at<double>(0,2)= -bound_rect.x;
	T.at<double>(1,2)= -bound_rect.y;
	// Important: first transform and then translate, not inverse way. correct form (T*H)
	H = T*H;
	warpPerspective(object, warped, H, Size(bound_rect.width, bound_rect.height));
    //scene = translateImg(scene, offset.width, offset.height);

    copyMakeBorder(scene, scene, offset.height, max(0, bound_rect.height+bound_rect.y-scene.rows),
                                 offset.width,  max(0, bound_rect.width+bound_rect.x-scene.cols),
                                 BORDER_CONSTANT,Scalar(0,0,0));

    for(Point2f& pt: bound_points){
        pt.x -= bound_rect.x;
        pt.y -= bound_rect.y;
    }

    Point pts[4] = {bound_points[0], bound_points[1], bound_points[2], bound_points[3]};

    Mat mask(bound_rect.height, bound_rect.width, CV_8UC3, Scalar(0,0,0));
    fillConvexPoly( mask, pts, 4, Scalar(255,255,255));
    erode( mask, mask, getStructuringElement( MORPH_RECT, Size(7, 7),Point(-1, -1)));

	cv::Mat object_pos(scene, cv::Rect(max(bound_rect.x,0), max(bound_rect.y,0), bound_rect.width, bound_rect.height));
    warped.copyTo(object_pos, mask);
    mask.release();

    // object_pos -= warped*200;
    // object_pos += warped;

    bound_rect.x = max(bound_rect.x,0);
    bound_rect.y = max(bound_rect.y,0);


	return bound_rect;
}

// See description in header file
vector<DMatch> getGoodMatches(int n_matches, vector<vector<DMatch> > matches){
    vector<DMatch> good_matches;

    for (int i = 0; i < std::min(n_matches, (int)matches.size()); i++) {
        if ((matches[i][0].distance < 0.5 * (matches[i][1].distance)) &&
            ((int) matches[i].size() <= 2 && (int) matches[i].size() > 0)) {
            // take the first result only if its distance is smaller than 0.5*second_best_dist
            // that means this descriptor is ignored if the second distance is bigger or of similar
            good_matches.push_back(matches[i][0]);
        }
    }
    return good_matches;
}

// See description in header file
vector<DMatch> gridDetector(vector<KeyPoint> keypoints, vector<DMatch> matches){
    int stepx=TARGET_WIDTH/10, stepy=TARGET_HEIGHT/10;
    vector<DMatch> grid_matches;
    int k=0, best_distance = 100;
    DMatch best_match;
    
    for(int i=0; i<10; i++){
        for(int j=0; j<10; j++){
            k=0;
            best_distance = 100;
            for (DMatch m: matches) {
                //-- Get the keypoints from the good matches
                if(keypoints[m.queryIdx].pt.x >= stepx*i && keypoints[m.queryIdx].pt.x < stepx*(i+1) &&
                keypoints[m.queryIdx].pt.y >= stepy*j && keypoints[m.queryIdx].pt.y < stepy*(j+1)){
                    if(m.distance < best_distance){
                        best_distance = m.distance;
                        best_match = m;
                    }
                    matches.erase(matches.begin() + k);  
                }
                k++;
            }
            if(best_distance != 100)
                grid_matches.push_back(best_match);
        }
    }
    return grid_matches;
}

// See description in header file
Mat translateImg(Mat img, double offsetx, double offsety){
    Mat T = (Mat_<double>(2,3) << 1, 0, offsetx, 0, 1, offsety);
    Mat result;
    warpAffine(img, result, T, cv::Size(img.cols+offsetx, img.rows+offsety));
    return result;
}

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
bool imageDistorted(Mat H, int width, int height){
    float deformation, area, keypoints_area;
    vector<Point2f> bound_points = getBoundPoints(H, width, height);
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

    // 3 initial threshold value, must be ajusted in future tests 
    if(area > 3*width*height)
        return false;
    // 4 initial threshold value, must be ajusted in future tests 
    if(ratio[0] > 4 || ratio[1] > 4)
        return false;
    

    return true;
}

// See description in header file
float getDistance(Point2f pt1, Point2f pt2){
    return sqrt(pow((pt1.x - pt2.x),2) + pow((pt1.y - pt2.y),2));
}

// See description in header file
float boundAreaKeypoints(vector<KeyPoint> keypoints, vector<DMatch> matches){
    vector<Point2f> points, hull;
    for(DMatch m: matches){
        points.push_back(keypoints[m.queryIdx].pt);
    }
    convexHull(points, hull);

    return contourArea(hull);
}