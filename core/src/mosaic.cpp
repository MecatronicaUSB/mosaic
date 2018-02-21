/**
 * @file mosaic.cpp
 * @brief Implementation of Mosaic class functions
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

    cout <<"Sub Mosaic # "<<n_subs+1<<" # Frames: "<<sub_mosaics[n_subs]->n_frames+1;
    cout << flush << "\r";
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
            sub_mosaics[n_subs]->avg_H = new_frame->H.clone();

            //sub_mosaics[n_subs]->computeOffset();
            sub_mosaics.push_back(new SubMosaic());
            n_subs++;

            new_frame->resetFrame();
            sub_mosaics[n_subs]->addFrame(new_frame);
            test = true;

            if (n_subs>1) {
                compute();
            }

            return true;
        }
        case NO_KEYPOINTS: {
            // TODO: evaluate this case
            return false;
        }
        case NO_HOMOGRAPHY: {
            // TODO: evaluate this case
            return false;
        }
        default: {
            return false;
        }
    }
}

void Mosaic::compute(int n_iter){

    vector<SubMosaic *> ransac_mosaics;
    vector<vector<Point2f> > points(2);
    vector<Point2f> mid_points(4);
    vector<Frame *> frames(4);
    float distortion = 100;
    float temp_distortion;
    int rnd_frame, rnd_point;
    int n_ponts;
    Mat temp_H, best_H;


    vector<SubMosaic *> scene;
    ransac_mosaics.push_back(sub_mosaics[0]);
    ransac_mosaics.push_back(sub_mosaics[1]);

    getReferencedMosaics(ransac_mosaics);

    ransac_mosaics[0]->computeOffset();
    ransac_mosaics[1]->computeOffset();

    // blender->blendSubMosaic(ransac_mosaics[0]);
    // imshow("Blend-Ransac0", ransac_mosaics[0]->final_scene);
    // blender->blendSubMosaic(ransac_mosaics[1]);
    // imshow("Blend-Ransac1", ransac_mosaics[1]->final_scene);
    // waitKey(0);


    blender->blendSubMosaic(ransac_mosaics[0]);
    for (Frame *frame: ransac_mosaics[0]->frames) {
        for(int j=0; j<frame->keypoints_pos[PREV].size(); j++){
            circle(ransac_mosaics[0]->final_scene, frame->keypoints_pos[PREV][j], 3, Scalar(255, 0, 0), -1);
        }
    }
    imshow("Blend-Ransac1", ransac_mosaics[0]->final_scene);

    blender->blendSubMosaic(ransac_mosaics[1]);
    for (Frame *frame: ransac_mosaics[1]->frames) {
        for(int j=0; j<frame->keypoints_pos[PREV].size(); j++){
            circle(ransac_mosaics[1]->final_scene, frame->keypoints_pos[PREV][j], 3, Scalar(255, 0, 0), -1);
        }
    }
    imshow("Blend-Ransac1", ransac_mosaics[1]->final_scene);

    waitKey(0);

    srand((uint32_t)getTickCount());

    points[0] = vector<Point2f>(4);
    points[1] = vector<Point2f>(4);

    for (int i=0; i<n_iter; i++) {
        for (int j=0; j<4; j++) {
            rnd_frame = rand() % ransac_mosaics[0]->n_frames;
            rnd_point = rand() % ransac_mosaics[0]->frames[rnd_frame]->bound_points[FIRST].size();

            points[0][j] = ransac_mosaics[0]->frames[rnd_frame]->bound_points[FIRST][rnd_point];
            points[1][j] = ransac_mosaics[1]->frames[rnd_frame]->bound_points[FIRST][rnd_point];
        }
        for (int j=0; j<4; j++) {
            mid_points[j] = getMidPoint(points[0][j], points[1][j]);
        }

        temp_H = getPerspectiveTransform(points[0], mid_points);

        for (Frame *frame: ransac_mosaics[0]->frames) {
            perspectiveTransform(frame->bound_points[FIRST],
                                 frame->bound_points[RANSAC], temp_H*frame->H);
        }

        temp_distortion = ransac_mosaics[0]->calcDistortion();

        if (temp_distortion < distortion) {
            distortion = temp_distortion;
            best_H = temp_H;
        }
    }

    for (Frame *frame: ransac_mosaics[0]->frames) {
        perspectiveTransform(frame->bound_points[FIRST],
                             frame->bound_points[RANSAC], best_H*frame->H);
    }

    for (Frame *frame: ransac_mosaics[0]->frames) {
        frame->setHReference(best_H);
    }


}

void Mosaic::getReferencedMosaics(vector<SubMosaic *> &_sub_mosaics){

    //SubMosaic *aux_mosaic1 = _sub_mosaics[1];

    _sub_mosaics[1]->referenceToZero();

    for (Frame *frame:  _sub_mosaics[1]->frames) {
        frame->setHReference(_sub_mosaics[0]->avg_H);
        _sub_mosaics[0]->addFrame(frame);
    }

    _sub_mosaics[1] = _sub_mosaics[0]->clone();

    for (Frame *frame: _sub_mosaics[1]->frames) {
        frame->setHReference(_sub_mosaics[0]->avg_H.inv());
    }

    _sub_mosaics[1]->avg_H = _sub_mosaics[0]->avg_H.inv();
}

void Mosaic::positionSubMosaics(SubMosaic *_first, SubMosaic *_second){

}

void Mosaic::show(){
    if (test) {
        if (!sub_mosaics[n_subs-1]->final_scene.data) {
            // blender->blendSubMosaic(sub_mosaics[n_subs-1]);
            // imshow("Blend", sub_mosaics[n_subs-1]->final_scene);
            // imwrite("/home/victor/dataset/output/neighbor-"+to_string(n_subs)+".jpg", sub_mosaics[n_subs-1]->final_scene);
            // waitKey(0);
            test = false;
        }

    }
}

}