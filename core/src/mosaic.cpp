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
void Mosaic::addFrame(Mat _object){
    tot_frames++;

    cout <<"Sub Mosaic # "<<n_subs+1<<" # Frames: "<<sub_mosaics[n_subs]->n_frames+1 << endl;

    Frame *new_frame = new Frame(_object.clone(), apply_pre);
    if (sub_mosaics[n_subs]->n_frames == 0) {
        sub_mosaics[n_subs]->addFrame(new_frame);
        return;
    }

    struct StitchStatus status;

    status = stitcher->stitch(new_frame,
                               sub_mosaics[n_subs]->last_frame,
                               sub_mosaics[n_subs]->scene_size);

    if (status.ok) {

        sub_mosaics[n_subs]->addFrame(new_frame);
        sub_mosaics[n_subs]->computeOffset();
        // sub_mosaics[n_subs]->updateOffset(status.offset);

    } else {
        sub_mosaics[n_subs]->is_complete = true;
        
        sub_mosaics[n_subs]->correct();
        sub_mosaics[n_subs]->computeOffset();

        blender->blendSubMosaic(sub_mosaics[n_subs]);
        imshow("Blend", sub_mosaics[n_subs]->final_scene);
        imwrite("/home/victor/dataset/output/neighbor-"+to_string(n_subs)+".jpg", sub_mosaics[n_subs]->final_scene);
        waitKey(0);
        
        sub_mosaics.push_back(new SubMosaic());
        n_subs++;

        new_frame->resetFrame();
        sub_mosaics[n_subs]->addFrame(new_frame);
    }
}

void Mosaic::compute(){

}

}