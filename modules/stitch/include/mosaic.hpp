/**
 * @file mosaic.hpp
 * @brief Mosaic2d Namespace and Classes
 * @version 0.2
 * @date 10/02/2018
 * @author Victor Garcia
 */
#ifndef SUB_MOSAIC_
#define SUB_MOSAIC_

#include "../../common/utils.h"
#include "../include/stitch.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/imgproc.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include <dirent.h>
#include <stdlib.h>
#include <iostream>
#include <cmath> 
#include <vector>

using namespace std;
using namespace cv;

namespace m2d
{

class SubMosaic{
    public:
        int n_frames;
        struct Hierarchy{
            SubMosaic* mosaic;
            float overlap;
        };
        vector<struct Hierarchy> neighbors;
        vector<Frame*> frames;
        Frame* key_frame;
        Stitcher* stitcher;
        Mat avH;
        Mat final_scene;

        void setRerenceFrame(Mat _scene);
        bool add2Mosaic(Mat _object);
        void calcAverageH();
        vector<Frame*> findNeighbors(Frame* _frame);

};

//TODO-----
// class Mosaic{
//     public:
//         int n_frames;
//         int n_subs;
//         vector<SubMosaic*> sub_mosaic;
//         Stitcher* stitcher;

//         Mosaic();
//         SubMosaic* addSubMosaic(SubMosaic* _sub_mosaic);
// };

/**
 * @brief Save the homography matrix and heypoints in a txt file
 * @param H OpenCV Matrix containing Homography transformation
 * @param keypoints Vector with OpenCV Keypoints
 * @param matches Vector with OpenCV Matches
 */
void saveHomographyData(cv::Mat H, vector<KeyPoint> keypoints[2], vector<cv::DMatch> matches);

}

#endif
