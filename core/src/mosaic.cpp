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
// See description in header file
Mosaic::Mosaic(bool _pre){
    apply_pre = _pre;
    n_subs = 0;
    tot_frames = 0;
    sub_mosaics.push_back(new SubMosaic());
}

// See description in header file
bool Mosaic::addFrame(Mat _object){
    tot_frames++;

    cout <<"Sub Mosaic # "<<n_subs+1<<" # Frames: "<<sub_mosaics[n_subs]->n_frames+1 << endl;

    Frame *new_frame = new Frame(_object.clone(), apply_pre);

    if (sub_mosaics[n_subs]->isEmpty()) {
        sub_mosaics[n_subs]->addFrame(new_frame);
        return true;
    }

    int status = stitcher->stitch(new_frame,
                                  sub_mosaics[n_subs]->last_frame,
                                  sub_mosaics[n_subs]->scene_size);

    switch( status ) {
        case OK: {
            sub_mosaics[n_subs]->addFrame(new_frame);
            sub_mosaics[n_subs]->computeOffset();
            return true;
        }
        case BAD_DISTORTION:{
        
            // sub_mosaics[n_subs]->correct();
            sub_mosaics[n_subs]->computeOffset();
            
            sub_mosaics.push_back(new SubMosaic());
            n_subs++;

            new_frame->resetFrame();
            sub_mosaics[n_subs]->addFrame(new_frame);
            test = true;
            return true;
        }
        case BAD_KEYPOINTS: {
            // TODO: evaluate this case
            return false;
        }
        case BAD_HOMOGRAPHY: {
            // TODO: evaluate this case
            return false;
        }
        default: {
            return false;
        }
    }
}

void Mosaic::compute(){

}

void Mosaic::show(){
    if (test) {
        if (!sub_mosaics[n_subs-1]->final_scene.data) {
            blender->blendSubMosaic(sub_mosaics[n_subs-1]);
            imshow("Blend", sub_mosaics[n_subs-1]->final_scene);
            imwrite("/home/victor/dataset/output/neighbor-"+to_string(n_subs)+".jpg", sub_mosaics[n_subs-1]->final_scene);
            waitKey(0);
            test = false;
        }

    }
}

}