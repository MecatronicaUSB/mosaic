/**
 * @file mosaic.cpp
 * @brief Implementation of stitch class and Mosaic2d Namespace functions 
 * @version 0.2
 * @date 10/02/2018
 * @author Victor Garcia
 */

#include "../include/mosaic.hpp"

using namespace std;
using namespace cv;

namespace m2d //!< mosaic 2d namespace
{

void SubMosaic::setRerenceFrame(Mat _scene){
    n_frames = 0;
    frames.push_back(new Frame(_scene, true));
    key_frame = frames[0];
    final_scene = _scene.clone();
}

bool SubMosaic::add2Mosaic(Mat _object){
    frames.push_back(new Frame(_object));
    n_frames++;
    cout << frames.size() <<  endl;
    return stitcher->stitch(frames[n_frames], frames[n_frames-1], final_scene);
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
        file << keypoints[1][m.trainIdx].pt.x << " ";
        file << keypoints[1][m.trainIdx].pt.y << "\n";
    }

    file.close();
}

}

