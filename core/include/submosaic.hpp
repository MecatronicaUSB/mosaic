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

/// options to print the resulting map
enum MapType
{
    RECTANGLE,
    CIRCLE
};

class SubMosaic;

/// Struct to link two SubMosaics
typedef struct _Hierarchy   
{                           
    SubMosaic *mosaic;      //!< Pointer to SubMosaic
    float overlap;          //!< Overlap area between this sub-mosaic and pointed one
} Hierarchy;

/// Struct to find the points for global euclidean correction
typedef struct _CornerPoint
{
    int index;              //!< index of point (0 top-left, 1 top-right, 2 bottom-right, 3 bottom-left)
    float distance;         //!< distance of point and Point at center of opposite frame
    Point2f point;          //!< OpenCV float Point coordinate

    /// Object contructor
    _CornerPoint(Point2f _point, int _idx) : point(_point), index(_idx){};
    // overload of "<" operator to sort the points by distance
    bool operator<(const _CornerPoint &point) const
    {
        return (distance < point.distance);
    }
} CornerPoint;

/**
 * @brief 
 */
class SubMosaic
{
  public:
    // ---------- Atributes
    int n_frames;                   //!< Number of frames in sub-mosaic
    Mat final_scene;                //!< Image containing all blended images (the sub-mosaic)
    Mat next_H;                     //!< Perspective transformation to place a new frame to last position
    Mat next_E;                     //!< Euclidean transformation to place a new frame to last position
    Frame *last_frame;              //!< Pointer to last frame
    Size2f scene_size;              //!< Size of mosaic image
    vector<Frame *> frames;         //!< Vector containing all the frames (Pointers) in sub-mosaic
    vector<Hierarchy> neighbors;    //!< Vector with all the neighbors SubMosaics (spatially close)
    // ---------- Methods
    /**
     * @brief Default constructor
     */
    SubMosaic();
    /**
     * @brief Default destructor
     */
    ~SubMosaic();
    /**
     * @brief Clone all the data
     * @return SubMosaic New sub mosaic with the same data
     */
    SubMosaic *clone();
    /**
     * @brief Add new frame object to sub mosaic and update sub mosaic information
     * @param _frame new frame to be added
     */
    void addFrame(Frame *_frame);
    /**
     * @brief Calculate the error based on distance of each keypoint match
     * @param _object first frame
     * @param _scene second frame
     * @return float resulting error
     */
    float calcKeypointsError(Frame *_object, Frame *_scene);
    /**
     * @brief Calculate the geometric distortion of sub mosaic
     * @return float resulting distortion
     */
    float calcDistortion(int _ref = RANSAC);
    /**
     * @brief Translate the sub mosaic to positives coordinates (top-left corner at 0,0)
     */
    void computeOffset();
    /**
     * @brief translate the first frame to default position (top-left corner at 0,0)
     */
    void referenceToZero();
    /**
     * @brief Get the centroid point, based on keypoints position
     * @return Point2f resulting centroid Point
     */
    Point2f getCentroid();
    /**
     * @brief Translate the sub mosaic based on input data
     * @param _offset Top and left offset
     */
    void updateOffset(vector<float> _offset);
    /**
     * @brief Apply global euclidean correction based on position of first and last frame
     */
    void correct();
    /**
     * @brief Get the corner points of sub mosic (first and last frame)
     * @return vector<vector<Point2f> > border points of euclidean and perspective transformation
     */
    vector<vector<Point2f> > getCornerPoints();
    /**
     * @brief Build the trajectory map of sub mosaic
     * @param type type of map, border lines or circle points and neighbor links
     * @param _color color of objects in map
     * @return Mat resulting map
     */
    Mat buildMap(int type = RECTANGLE, Scalar _color = Scalar(255, 0, 0));
};
}