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

struct _BlendPoint {
    int index;
    float distance;
    Point2f *points;

    _BlendPoint(int _index, Point2f *_points) : index(_index),
                                                points(_points),
                                                distance(getDistance(_points[PREV], _points[NEXT])){};

    bool operator < (const _BlendPoint& pt) const{
        return (distance < pt.distance);
    }
};

typedef _BlendPoint BlendPoint;

class Blender{
    public:
        /**
         * @brief 
         */
        Blender();
        /**
         * @brief 
         * @param _sub_mosaic 
         */
        void blendSubMosaic(SubMosaic *_sub_mosaic);
        /**
         * @brief 
         * @param _points 
         */
        void reduceRoi(vector<Point2f> &_points);
        /**
         * @brief 
         */
        vector<Point2f> findLocalStitch(Frame * _object, Frame *_scene);
        
};

}