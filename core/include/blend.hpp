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

struct _BlendPoint
{
    int index;
    float distance;
    Point2f prev;
    Point2f next;

    _BlendPoint(int _index, Point2f _prev, Point2f _next) : index(_index),
                                                            prev(_prev),
                                                            next(_next),
                                                            distance(getDistance(_prev, _next)){};

    bool operator<(const _BlendPoint &pt) const
    {
        return (distance < pt.distance);
    }
};

typedef _BlendPoint BlendPoint;

class Blender
{
  public:
    int bands;
    int graph_cut;
    vector<UMat> warp_imgs;
    vector<UMat> masks;
    vector<UMat> full_masks;
    vector<Rect2f> bound_rect;
    /**
         * @brief 
         */
    Blender(int _bands = 5, int _cut_line = 0): bands(_bands), graph_cut(_cut_line){};
    /**
         * @brief 
         * @param _sub_mosaic 
         */
    void blendSubMosaic(SubMosaic *_sub_mosaic);
    /**
         * @brief 
         * @param _points 
         */
    UMat getMask(Frame *_frame);
    /**
     * @brief 
     * @param _sub_mosaic 
     */
    void correctColor(SubMosaic *_sub_mosaic);
    /**
     * @brief 
     */
    void cropMask(int _object_index, int _scene_index);
    /**
     * @brief 
     * @param _object 
     * @param _scene 
     * @return vector<Mat> 
     */
    vector<Mat> getOverlapMasks(int _object, int _scene);
    /**
         * @brief 
         */
    vector<Point2f> findLocalStitch(Frame *_object, Frame *_scene);
    /**
         * @brief 
         * @return Mat 
         */
    UMat getWarpImg(Frame *_frame);
};
}