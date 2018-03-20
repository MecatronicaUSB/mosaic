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

class Mosaic
{
  public:
    // ---------- Atributes
    int tot_frames;
    int n_subs;
    int mosaic_mode;
    vector<SubMosaic *> sub_mosaics;
    vector<SubMosaic *> final_mosaics;
    bool apply_pre; //!< flag to apply or not SCB preprocessing algorithm
    Stitcher *stitcher;
    Blender *blender;
    // ---------- Methods
    Mosaic(bool _pre = true, int _mode = SIMPLE);
    // TODO:
    SubMosaic *addSubMosaics(SubMosaic *_sub_mosaic1, SubMosaic *_sub_mosaic2);
    /**
         * @brief 
         * @param _object 
         */
    bool addFrame(Mat _object);
    /**
         * @brief 
         * @param n_inter 
         */
    void compute();
    /**
         * @brief 
         * @param _first 
         * @param _second 
         * @param _ref 
         * @return SubMosaic* 
         */
    void getReferencedMosaics(vector<SubMosaic *> &_sub_mosaics);
    /**
         * @brief 
         * @param _sub_mosaics 
         */
    void alignMosaics(vector<SubMosaic *> &_sub_mosaics);
    /**
         * @brief 
         * @param ransac_mosaics 
         * @return Mat 
         */
    Mat getBestModel(vector<SubMosaic *> &_ransac_mosaics, int _niter = 2000);
    /**
     * @brief 
     * @param _object 
     * @param _scece 
     * @return float 
     */
    float getOverlap(SubMosaic *_object, SubMosaic *_scene);
    // temporal function
    void print();
};
}