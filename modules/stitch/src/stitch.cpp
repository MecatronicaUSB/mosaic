#include "../include/stitch.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

Rect getBound(Mat H, int width, int height){
	Rect r;
	vector<Point2f> points;
	points.push_back(Point2f(0,0));
	points.push_back(Point2f(width,0));
	points.push_back(Point2f(width,height));
	points.push_back(Point2f(0,height));

	vector<Point2f> finalpts(4);
	perspectiveTransform(points,finalpts, H);

	return boundingRect(finalpts);
}

Mat stitch(Mat object, Mat scene, Mat H){
	Mat result;
	Size dim;
	Size2f offset;

	Rect bound = getBound(H, TARGET_WIDTH, TARGET_HEIGHT);

	bound.x < 0 ? offset.width  = -bound.x : offset.width  = 0;
	bound.y < 0 ? offset.height = -bound.y : offset.height = 0;

	dim.width  = max(TARGET_WIDTH + (int)offset.width,  bound.width) + max(bound.x,0);
	dim.height = max(TARGET_HEIGHT+ (int)offset.height, bound.height) + max(bound.y,0);

	Mat T = Mat::eye(3,3,CV_64F);
	T.at<double>(0,2)= offset.width;
	T.at<double>(1,2)= offset.height;
	// Important: first transform and then translate, not inverse. (T*H)
	H = T*H;
	warpPerspective(object,result,H,dim);

	cv::Mat half(result,cv::Rect(offset.width, offset.height, 640, 480));
	scene.copyTo(half);

	return result;
}

// See description in header file
std::vector<DMatch> getGoodMatches(int n_matches, std::vector<std::vector<cv::DMatch> > matches){
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
std::vector<string> read_filenames(const std::string dir_ent){
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