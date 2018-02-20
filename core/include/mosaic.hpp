/**
 * @file mosaic.hpp
 * @brief Implementation of Frame, SumMosaic and Mosaic classes
 * @version 0.2
 * @date 10/02/2018
 * @author Victor Garcia
 */
#pragma once
#include "submosaic.hpp"

using namespace std;
using namespace cv;

namespace m2d //!< mosaic 2d namespace
{

class Stitcher;
class Blender;
class SubMosaic;

class Mosaic{
    public:
        // ---------- Atributes
        int tot_frames;
        int n_subs;
        vector<SubMosaic *> sub_mosaics;
        bool apply_pre;                         //!< flag to apply or not SCB preprocessing algorithm
        Stitcher *stitcher;
        Blender *blender;
        // ---------- Methods
        Mosaic(bool _pre = true);
        // TODO:
        SubMosaic* addSubMosaics(SubMosaic *_sub_mosaic1, SubMosaic *_sub_mosaic2);
        /**
         * @brief 
         * @param _object 
         */
        void addFrame(Mat _object);
        // TODO:
        void compute();
};

}