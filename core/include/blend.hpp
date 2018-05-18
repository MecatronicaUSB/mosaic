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

/// Structure to classify match points by distance, useful to determine local stitch area
typedef struct _BlendPoint
{
    int index;          //!< index of image corresponding Point
    float distance;     //!< distance between same point in different images
    Point2f prev;       //!< Point in previous image
    Point2f next;       //!< Point in next image

    // Initialize parameters when object is created
    _BlendPoint(int _index, Point2f _prev, Point2f _next) : index(_index),
                                                            prev(_prev),
                                                            next(_next),
                                                            distance(getDistance(_prev, _next)){};
    // overload of "<" operator to sort the points by distance
    bool operator<(const _BlendPoint &pt) const
    {
        return (distance < pt.distance);
    }
} BlendPoint;

// Class used to blend one sub mosaic
class Blender
{
  public:
    int bands;                      //!< number of bands for multi-band blender
    bool graph_cut;                 //!< boolean to use or nor graph-cut algorithm
    bool color_correction;
    bool scb;                       //!< boolean to use or not simple color balance in final image
    vector<UMat> warp_imgs;         //!< Vector to store all warped images
    vector<UMat> masks;             //!< Vector to store correspond warped masks (to be cropped by seam finder algorithm)
    vector<UMat> full_masks;        //!< Vector to store completely filled masks
    vector<Rect2f> bound_rect;      //!< Vector to store the minimum bounding rectangle of each image
    /**
     * @brief Blender constructor
     */
    Blender(int _bands = 5, bool _color = true, bool _cut_line = false, int _scb = false): bands(_bands),
                                                                       color_correction(_color),
                                                                       graph_cut(_cut_line),
                                                                       scb(_scb){};
    /**
     * @brief Main function to blend on sub mosaic
     * @param _sub_mosaic Sub Mosaic to be blended
     */
    void blendSubMosaic(SubMosaic *_sub_mosaic);
    /**
     * @brief Get the filled mask of input frame, using the bounding points
     * @param _frame Input Frame pointer
     * @return UMat OpenCV Matrix (OpenCL) containing the resulting mask
     */
    UMat getMask(Frame *_frame);
    /**
     * @brief Get the warped image by the correspond transformation matrix
     * @return Mat OpenCV Matrix (OpenCL) containing the warp image (with top left corner in 0,0)
     */
    UMat getWarpImg(Frame *_frame);
    /**
     * @brief Get the intersection mask between two frames
     * @param _object Index of object mask in global masks vector
     * @param _scene Index of scene mask in global masks vector
     * @return vector<Mat> Intersection mask referenced to each frame
     * @detail The first element will have the intersection mask between two frames
     * in the location if first image, and vice versa.
     */
    vector<Mat> getOverlapMasks(int _object, int _scene);
    /**
     * @brief Correct color on sub mosaic using Reinhard's method
     * @param _sub_mosaic Input Sub Mosaic pointer
     */
    void correctColor(SubMosaic *_sub_mosaic);
    /**
     * @brief (Currently unused) Crop the scene mask with the object one 
     * @param _object_index Index of object mask in global masks vector
     * @param _scene_index Index of scene mask in global masks vector
     */
    void cropMask(int _object_index, int _scene_index);
    /**
     * @brief (Currently unused) find the area where the matches are below a threshold value
     */
    vector<Point2f> findLocalStitch(Frame *_object, Frame *_scene);
    /**
     * @brief 
     * @param _object 
     * @param _scene 
     * @return true 
     * @return false 
     */
    bool checkCollision(Frame *_object, Frame *_scene);
};
}