/**
 * @file submosaic.hpp
 * @brief Description for SubMosaic Class
 * @version 0.2
 * @date 10/02/2018
 * @author Victor Garcia
 */

#pragma once
#include "frame.hpp"

using namespace std;
using namespace cv;

namespace m2d //!< mosaic 2d namespace
{

class SubMosaic;

typedef struct
{                      //!< Struct to relate two SubMosaics
    SubMosaic *mosaic; //!< Pointer to SubMosaic
    float overlap;     //!< Overlap area between this sub-mosaic and pointed one
} Hierarchy;

struct _CornerPoint
{
    int index;
    float distance;
    Point2f point;

    _CornerPoint(Point2f _point, int _idx) : point(_point), index(_idx){};
    bool operator<(const _CornerPoint &point) const
    {
        return (distance < point.distance);
    }
};

typedef _CornerPoint CornerPoint;

/**
 * @brief 
 */
class SubMosaic
{
  public:
    // ---------- Atributes
    int n_frames;    //!< Number of frames in sub-mosaic
    Mat final_scene; //!< Image containing all blended images (the sub-mosaic)
    Mat avg_H;       //!< Average Homography matrix (Matrix that reduces the dostortion error)
    Mat avg_E;       //!< Average Homography matrix (Matrix that reduces the dostortion error)
    Frame *last_frame;
    Size2f scene_size;
    vector<Frame *> frames;      //!< Vector containing all the frames (Pointers) in sub-mosaic
    vector<Hierarchy> neighbors; //!< Vector with all the neighbors SubMosaics (spatially close)
    // ---------- Methods
    /**
         * @brief Default constructor
         */
    SubMosaic();
    /**
         * @brief 
         */
    ~SubMosaic();
    /**
         * @brief 
         * @return SubMosaic 
         */
    SubMosaic *clone();
    /**
         * @brief Using the Stitcher class, add the object image to the current sub-mosaic
         * @param _object OpenCV Matrix containig the BGR image to add in the sub-mosaic
         * @return true If the stitch was sucesussfull
         * @return false If the stitch wasn't sucesussfull
         */
    void addFrame(Frame *_frame);
    /**
         * @brief 
         * @param _object 
         * @param _scene 
         * @return float 
         */
    float calcKeypointsError(Frame *_first, Frame *_second);
    /**
         * @brief 
         * @return float 
         */
    float calcDistortion();
    /**
         * @brief 
         */
    void computeOffset();
    /**
         * @brief 
         */
    void referenceToZero();
    /**
         * @brief 
         * @return Point2f 
         */
    Point2f getCentroid();
    /**
         * @brief 
         * @param _frames 
         */
    void updateOffset(vector<float> _offset);
    /**
         * @brief 
         */
    void correct();
    /**
     * @brief 
     * @return vector<vector<Point2f> > 
     */
    vector<vector<Point2f> > getCornerPoints();

    vector<vector<Point2f> > getCornerPoints2();
    /**
         * @brief Calculate the Homography matrix that reduce the distortion error in the sub-mosaic
         * (Not yet implemented)
         */
    void calcAverageH();
    /**
         * @brief 
         * @return true 
         * @return false 
         */
    bool isEmpty();
};
}