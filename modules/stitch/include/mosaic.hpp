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

enum FrameImg{
    COLOR,
    GRAY
};

enum FrameRef{
    PREV,
    NEXT
};

struct Hierarchy{
    SubMosaic* sub_m;
    float overlap;
};

class Frame{
    public:
        Mat H;
        Rect bound_rect;
        vector<Mat> img = vector<Mat>(2);
        vector<Point2f> points;
        vector<Frame*> neighbors;
        Frame* key_frame;
        bool key;

        Frame(Mat _img, bool _key = false) : key(_key){};
        void setHReference(Mat _H);
        void calcDistortion();
};

class SubMosaic{
    public:
        int n_frames;
        Mat avH;
        Frame* key_frame;
        vector<Frame*> frames;
        vector<struct Hierarchy> neighbors;
        Stitcher stitcher;

        SubMosaic();
        vector<Frame*> findNeighbors(Frame* _frame);
        void setFrameRerence(Frame* _frame);
        void calcAverageH();

};

class Mosaic{
    public:
        int n_frames;
        int n_subs;
        vector<SubMosaic*> sub_mosaic;

        Mosaic();
        SubMosaic addSubMosaic(SubMosaic _sub_mosaic);
};

}

#endif
