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
    // ---------- Attributes
    int n_subs;                                   //!< Number of sub mosaics for each mosaic build
    bool apply_pre;                               //!< flag to apply or not SCB preprocessing algorithm on gray image
    vector<Mat> map;                              //!< Track map for each mosaic
    vector<Frame *> frames;                       //!< All frames object to build the mosaic
    vector<SubMosaic *> sub_mosaics;              //!< Vector of all sub mosaics that can be merged
    vector<vector<SubMosaic *> > final_mosaics;   //!< Vector of disjoint mosaics
    Stitcher *stitcher;                           //!< Stitcher class pointer
    Blender *blender;                             //!< Blender class pointer
    // ---------- Methods
    /**
     * @brief Mosaic constructor
     * @param _pre bollean to apply or not SCB on gray scale image (improve feature detection)
     */
    Mosaic(bool _pre = true) : apply_pre(_pre), n_subs(0){};
    /**
     * @brief Add new frame to mosaic, create frame object and count total frames number
     * @param _object OpenCV Matrix containing input image
     */
    void feed(Mat _object);
    /**
     * @brief Build the sub mosaics 
     * @param _euclidean_mode if true, build using only euclidean transformation
     */
    void compute(bool _euclidean_mode = false);
    /**
     * @brief merge all sub mosaics based on overlap area
     * @param _euclidean_mode if true, no global euclidean correction is needed
     */
    void merge(bool _euclidean_mode = false);
    /**
     * @brief get the two sub mosaics with more overlap
     * @param _sub_mosaics all sub mosaics
     * @return vector<SubMosaic *> resulting sub mosaics
     */
    vector<SubMosaic *> getBestOverlap(vector<SubMosaic *> _sub_mosaics);
    /**
     * @brief update sub mosaics order and neighbors, after sub mosaic merge
     * @param _sub_mosaics input sub mosaics
     * @detail If we have:
     *    submosaic 0         submosaic 1         submosaic 2     \n
     *  nothing at left       SM0 at left         SM1 at left     \n
     *   SM1 at right        SM2 at right       nothing at right  \n
     * [(<-X)--0--(1->)] - [(<-0)--1--(2->)] - [(<-1)--2--(X->)]  \n
     *                                                            \n
     * The result will be:                                        \n
     * [(<-X)--0--(2->)] - [(<-0)--2--(X->)]                      \n
     */
    void removeNeighbor(vector<SubMosaic *> &_sub_mosaics);
    /**
     * @brief Reference two sub mosaic by each reference frame
     * @param _sub_mosaics resulting referenced sub mosaics 
     */
    void referenceMosaics(vector<SubMosaic *> &_sub_mosaics);
    /**
     * @brief Translate two input sub mosaic by their centroid and rotate by similarity transformation
     * @param _sub_mosaics Input sub mosaics
     */
    void alignMosaics(vector<SubMosaic *> &_sub_mosaics);
    /**
     * @brief Get the best transformation matrix, which reduce the overall geometrical distortion
     * @param _ransac_mosaics Two input sub mosaics, previously referenced and aligned
     * @param _niter number of iterations of RANSAC algorithm
     * @return Mat Best transformation matrix to reduce distortion (to be applied on first sub mosaic)
     */
    Mat getBestModel(vector<SubMosaic *> _ransac_mosaics, int _niter = 3000);
    /**
     * @brief Get the overlap area between two sub mosaics
     * @param _object input sub mosaic (bust be next to _scene sub mosaic)
     * @param _scene input sub mosaic
     * @return float normalized overlap area
     */
    float getOverlap(SubMosaic *_object, SubMosaic *_scene);
    /**
     * @brief Update overlap area on each input element
     * @param _sub_mosaic vector containing almost two sub mosaics
     */
    void updateOverlap(vector<SubMosaic *> &_sub_mosaic);
    /**
     * @brief save the final mosaic
     * @param _dir path and filename of resulting mosaic
     */
    void save(string _dir);
    /**
     * @brief show the final mosaic (reduced version)
     */
    void show();
};
}