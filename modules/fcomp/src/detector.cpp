#include "../include/detector.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

#define TARGET_WIDTH	640   
#define TARGET_HEIGHT	480 

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
    int best_distance = 100;
    DMatch best_match;
    
    for(int i=0; i<10; i++){
        for(int j=0; j<10; j++){
            best_distance = 100;
            for (auto m: matches) {
                //-- Get the keypoints from the good matches
                if(keypoints[m.queryIdx].pt.x >= stepx*i && keypoints[m.queryIdx].pt.x < stepx*(i+1) &&
                keypoints[m.queryIdx].pt.y >= stepy*j && keypoints[m.queryIdx].pt.y < stepy*(j+1)){
                    if(m.distance < best_distance){
                        best_distance = m.distance;
                        best_match = m;
                    }
                    matches.erase(matches.begin() + m.queryIdx);  
                }
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