/**
 * @file mosaic.cpp
 * @brief Implementation of stitch class and Mosaic2d Namespace functions 
 * @version 0.2
 * @date 10/02/2018
 * @author Victor Garcia
 */

#include "../include/mosaic.hpp"

using namespace std;
using namespace cv;

namespace m2d //!< mosaic 2d namespace
{

Frame::Frame(Mat _img, bool _key){
    img[COLOR] = _img.clone();
    cvtColor(img[COLOR], img[GRAY], CV_BGR2GRAY);
}

SubMosaic::SubMosaic(){

}

Mosaic::Mosaic(){
    
}

}

