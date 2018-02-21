/**
 * @file mosaic.hpp
 * @brief Description for main Mosaic Class
 * @version 0.2
 * @date 10/02/2018
 * @author Victor Garcia
 */

#pragma once
#include "stitch.hpp"
#include "blend.hpp"

using namespace std;
using namespace cv;

namespace m2d //!< mosaic 2d namespace
{
    
class Mosaic{
    public:
        // ---------- Atributes
        int tot_frames;
        int n_subs;
        vector<SubMosaic *> sub_mosaics;
        bool apply_pre;                         //!< flag to apply or not SCB preprocessing algorithm
        Stitcher *stitcher;
        Blender *blender;
        bool test = false;
        // ---------- Methods
        Mosaic(bool _pre = true);
        // TODO:
        SubMosaic* addSubMosaics(SubMosaic *_sub_mosaic1, SubMosaic *_sub_mosaic2);
        /**
         * @brief 
         * @param _object 
         */
        bool addFrame(Mat _object);
        /**
         * @brief 
         * @param n_inter 
         */
        void compute(int n_inter = 200);
        /**
         * @brief 
         * @param _first 
         * @param _second 
         * @param _ref 
         * @return SubMosaic* 
         */
        void getReferencedMosaics(vector<SubMosaic *> _sub_mosaics);
        /**
         * @brief 
         */
        void positionSubMosaics(SubMosaic *_first, SubMosaic *_second);
        // temporal function
        void show();
};

}