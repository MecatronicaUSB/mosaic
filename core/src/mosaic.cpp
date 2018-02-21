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
            sub_mosaics[n_subs]->computeOffset();
            
            sub_mosaics.push_back(new SubMosaic());
            n_subs++;

            new_frame->resetFrame();
            sub_mosaics[n_subs]->addFrame(new_frame);
            test = true;
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

    vector<SubMosaic *> ransac_mosaic(2);
    vector<vector<Point2f> > points(2);
    vector<Point2f> mid_points(4);
    vector<Frame *> frames(4);
    float distortion = 100;
    float temp_distortion;
    int rnd_frame, rnd_point;
    int n_ponts;
    Mat temp_H, best_H;

    SubMosaic *scene = sub_mosaics[0];
    SubMosaic *object = sub_mosaics[1];

    ransac_mosaic[0] = getReferencedMosaic(scene, object, FIRST);
    ransac_mosaic[1] = getReferencedMosaic(scene, object, SECOND);

    ransac_mosaic[0]->computeOffset();
    ransac_mosaic[1]->computeOffset();

    int n_frames = ransac_mosaic[0]->n_frames;
    srand((uint32_t)getTickCount());


    for (int i=0; i<n_iter; i++) {
        for (int j=0; j<4; j++) {
            rnd_frame = rand() % n_frames;
            rnd_point = ransac_mosaic[0]->frames[rand()%n_frames]->bound_points[FIRST].size();
            points[0].push_back(ransac_mosaic[0]->frames[rand()%n_frames]->bound_points[FIRST][rnd_point]);
            points[1].push_back(ransac_mosaic[1]->frames[rand()%n_frames]->bound_points[FIRST][rnd_point]);
        }
        for (int j=0; j<4; j++) {
            mid_points[j] = getMidPoint(points[0][j], points[1][j]);
        }

        temp_H = getPerspectiveTransform(points[0], mid_points);

        for (Frame *frame: ransac_mosaic[0]->frames) {
            perspectiveTransform(frame->bound_points[FIRST],
                                 frame->bound_points[RANSAC], temp_H*frame->H);
        }

        temp_distortion = ransac_mosaic[0]->calcDistortion();

        if (temp_distortion < distortion) {
            distortion = temp_distortion;
            best_H = temp_H;
        }
    }

    for (Frame *frame: ransac_mosaic[0]->frames) {
        perspectiveTransform(frame->bound_points[FIRST],
                             frame->bound_points[RANSAC], best_H*frame->H);
    }


}

SubMosaic* Mosaic::getReferencedMosaic(SubMosaic *_first, SubMosaic *_second, int _ref){

}

void Mosaic::positionSubMosaics(SubMosaic *_first, SubMosaic *_second){

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