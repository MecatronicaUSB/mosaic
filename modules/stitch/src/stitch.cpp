#include "../include/stitch.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

// See description in header file
struct WarpPoly getBound(Mat H, int width, int height){
	vector<Point2f> points;
    struct WarpPoly bound;

	points.push_back(Point2f(0,0));
	points.push_back(Point2f(width,0));
	points.push_back(Point2f(width,height));
	points.push_back(Point2f(0,height));

	perspectiveTransform(points,bound.points, H);
    bound.rect =  boundingRect(bound.points);
	return bound;
}

// See description in header file
struct WarpPoly stitch(Mat object, Mat& scene, Mat H){
	Mat warped;
    Size dim;
	Size2f offset;

	struct WarpPoly bound = getBound(H, object.cols, object.rows);

	bound.rect.x < 0 ? offset.width  = -bound.rect.x : offset.width  = 0;
	bound.rect.y < 0 ? offset.height = -bound.rect.y : offset.height = 0;

    dim.width  = scene.cols + abs(bound.rect.x);
    dim.height = scene.rows + abs(bound.rect.y);

	Mat T = Mat::eye(3,3,CV_64F);
	T.at<double>(0,2)= -bound.rect.x;
	T.at<double>(1,2)= -bound.rect.y;
	// Important: first transform and then translate, not inverse way. correct form (T*H)
	H = T*H;
	warpPerspective(object, warped, H, cv::Size(bound.rect.width, bound.rect.height));
    //scene = translateImg(scene, offset.width, offset.height);

    copyMakeBorder(scene, scene, offset.height, max(0, bound.rect.height+bound.rect.y-scene.rows),
                                 offset.width,  max(0, bound.rect.width+bound.rect.x-scene.cols),
                                 BORDER_CONSTANT,Scalar(0,0,0));

    for(int i=0; i< bound.points.size(); i++){
        bound.points[i].x -= bound.rect.x;
        bound.points[i].y -= bound.rect.y;
    }

    Point pts[4] = {bound.points[0], bound.points[1], bound.points[2], bound.points[3]};

    Mat mask(bound.rect.height, bound.rect.width, CV_8UC3, Scalar(0,0,0));
    fillConvexPoly( mask, pts, 4, Scalar(255,255,255));
    erode( mask, mask, getStructuringElement( MORPH_RECT, Size(7, 7),Point(-1, -1)));

	cv::Mat object_pos(scene, cv::Rect(max(bound.rect.x,0), max(bound.rect.y,0), bound.rect.width, bound.rect.height));
	warped.copyTo(object_pos, mask);
    mask.release();
    bound.rect.x = max(bound.rect.x,0);
    bound.rect.y = max(bound.rect.y,0);

	return bound;
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
            for (auto m: matches) {
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
        file << keypoints[1][m.queryIdx].pt.x << " ";
        file << keypoints[1][m.queryIdx].pt.y << "\n";
    }

    file.close();
}