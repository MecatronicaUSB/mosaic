/**
 * @file blend.hpp
 * @brief Description of Blend class
 * @version 0.2
 * @date 10/02/2018
 * @author Victor Garcia
 */

#pragma once
#include "submosaic.hpp"

using namespace std;
using namespace cv;

namespace m2d
{

class Blender{
    public:
        Blender();
        void blendSubMosaic(SubMosaic *_sub_mosaic);
        void reduceRoi(vector<Point2f> &_points);
};

}