#include "../include/detector.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

#define TARGET_WIDTH	640   
#define TARGET_HEIGHT	480 

// TODO: Cite proper ref to work where was introduced the idea of image tiling for further extraction of features on each tile
/// Number of columns and rows for image tiling
#define GRID_COLUMNS	10
#define GRID_ROWS		10

// See description in header file
std::vector<DMatch> getGoodMatches(std::vector<std::vector<cv::DMatch> > matches){
    vector<DMatch> good_matches;

    for (int i = 0; i < matches.size(); i++) {
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
std::vector<DMatch> getGoodMatchesBF(std::vector<cv::DMatch>  matches){
    std::vector<DMatch> good_matches;
    double min_distance = 100;
    for (int i = 0; i < matches.size(); i++)
    {
        if (matches[i].distance < min_distance){
            min_distance = matches[i].distance;
        }
    }
    for (int i = 0; i < matches.size(); i++)
    {
        if (matches[i].distance <= max(2*min_distance, 0.02))
        {
            good_matches.push_back(matches[i]);
        }
    }
    return good_matches;
}

// See description in header file
vector<DMatch> gridDetector(vector<KeyPoint> keypoints, vector<DMatch> matches){
	/// TODO: use temporal float variable, and later apply integer cast when employed as image index
	/// check if stepx/stepy can be provided as function arguments
    int stepx=TARGET_WIDTH/GRID_COLUMNS, stepy=TARGET_HEIGHT/GRID_ROWS;
    vector<DMatch> grid_matches;	// vector containing matches obtained with grid/tiling approach
    int best_distance = 10000;	// forced initial value for best_distance (default value may be ignored as it is not employed)
    DMatch best_match;	// current best match
    
    for(int i=0; i<10; i++){
        for(int j=0; j<10; j++){
            best_distance = 10000;
            for (auto m: matches) {	// booom, using 'auto' like a pro...
                //-- Get the keypoints from the good matches
                if(keypoints[m.trainIdx].pt.x >= stepx*i && keypoints[m.trainIdx].pt.x < stepx*(i+1) &&
                keypoints[m.trainIdx].pt.y >= stepy*j && keypoints[m.trainIdx].pt.y < stepy*(j+1)){
                    if(m.distance < best_distance){
                        best_distance = m.distance;
                        best_match = m;
                    }
                    //matches.erase(matches.begin() + m.trainIdx);  
                }
            }
            if(best_distance != 10000)
                grid_matches.push_back(best_match);
        }
    }
    return grid_matches;
}
