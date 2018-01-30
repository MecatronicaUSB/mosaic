#include "../include/stitch.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

struct warpPoly getBound(Mat H, int width, int height){
	Rect r;
	vector<Point2f> points;
    struct warpPoly bound;
	points.push_back(Point2f(0,0));
	points.push_back(Point2f(width,0));
	points.push_back(Point2f(width,height));
	points.push_back(Point2f(0,height));

	perspectiveTransform(points,bound.points, H);
    bound.rect =  boundingRect(bound.points);
	return bound;
}

Mat stitch(Mat object, Mat scene, Mat H){
	Mat result;
	Size2f offset;

	struct warpPoly bound = getBound(H, object.cols, object.rows);

	bound.rect.x < 0 ? offset.width  = -bound.rect.x : offset.width  = 0;
	bound.rect.y < 0 ? offset.height = -bound.rect.y : offset.height = 0;

	Mat T = Mat::eye(3,3,CV_64F);
	T.at<double>(0,2)= offset.width;
	T.at<double>(1,2)= offset.height;
	// Important: first transform and then translate, not inverse. (T*H)
	H = T*H;
	warpPerspective(object,result,H,Size(bound.rect.width, bound.rect.height));
    scene = translateImg(scene, offset.width, offset.height);
    Mat mask(bound.rect.width, bound.rect.height, CV_8UC1, Scalar(0));

    vector<Point2f> hull;
    convexHull(bound.points, hull);
    vector<vector<Point>> contour(1, bound.points);
    drawContours(mask, contour, 0, Scalar(255), CV_FILLED);

	cv::Mat object_warped(result,cv::Rect(offset.width, offset.height, 640, 480));
	object.copyTo(object_warped);

	return result;
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
    warpAffine(img,result,T,Size(img.cols+offsetx, img.rows+offsety));
    return result;
}

// See description in header file
vector<string> read_filenames(const string dir_ent){
    vector<string> file_names;
    DIR *dir;
    struct dirent *ent;

    if ((dir = opendir(dir_ent.c_str())) != NULL) {
        while ((ent = readdir (dir)) != NULL) {
            file_names.push_back(string(ent->d_name));
        }
        closedir (dir);
    } else {
    // If the directory could not be opened
    cout << "Directory could not be opened" <<endl;
    }
    // Sorting the vector of strings so it is alphabetically ordered
    sort(file_names.begin(), file_names.end());
    file_names.erase(file_names.begin(), file_names.begin()+2);

    return file_names;
}