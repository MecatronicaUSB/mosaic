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

            if (n_subs>4) {
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

// See description in header file
void Mosaic::compute(){

    vector<SubMosaic *> ransac_mosaics(2);

    for (int i=0; i<sub_mosaics.size(); i++) {

        ransac_mosaics[0] = sub_mosaics[0];
        ransac_mosaics[1] = sub_mosaics[i+1];

        getReferencedMosaics(ransac_mosaics);

        ransac_mosaics[0]->computeOffset();
        ransac_mosaics[1]->computeOffset();

        //alignMosaics(ransac_mosaics);

        // for (Frame *frame: ransac_mosaics[0]->frames) {
        //     for(int j=0; j<frame->keypoints_pos[PREV].size(); j++){
        //         circle(ransac_mosaics[0]->final_scene, frame->keypoints_pos[PREV][j], 3, Scalar(255, 0, 0), -1);
        //     }
        // }

        Mat best_H = getBestModel(ransac_mosaics, 2000);
        for (Frame *frame: ransac_mosaics[0]->frames) {
            frame->setHReference( best_H);
        }
        ransac_mosaics[0]->avg_H = best_H * ransac_mosaics[0]->avg_H;

        ransac_mosaics[0]->computeOffset();

        blender->blendSubMosaic(ransac_mosaics[0]);
        imshow("Blend-Ransac2", ransac_mosaics[0]->final_scene);
        imwrite("/home/victor/dataset/output/ransac-00.jpg", ransac_mosaics[0]->final_scene);
        waitKey(0);

        sub_mosaics.erase(sub_mosaics.begin()+1);
    }
}

// See description in header file
void Mosaic::getReferencedMosaics(vector<SubMosaic *> &_sub_mosaics){

    Mat ref_H = _sub_mosaics[0]->avg_H.clone();

    _sub_mosaics[1]->referenceToZero();
    Mat new_ref_H = _sub_mosaics[1]->avg_H.clone();

    for (Frame *frame:  _sub_mosaics[1]->frames) {
        frame->setHReference(ref_H);
        _sub_mosaics[0]->addFrame(frame);
    }
    new_ref_H = ref_H * new_ref_H;

    _sub_mosaics[1] = _sub_mosaics[0]->clone();
    for (Frame *frame: _sub_mosaics[1]->frames) {
        frame->setHReference(ref_H.inv());
    }

    _sub_mosaics[0]->avg_H = new_ref_H.clone();
}

// See description in header file
Mat Mosaic::getBestModel(vector<SubMosaic *> _ransac_mosaics, int _niter){
    int rnd_frame, rnd_point;
    Mat temp_H, best_H;
    float distortion = 1000;
    float temp_distortion;
    vector<vector<Point2f> > points(2);
    vector<Point2f> mid_points(4);
    vector<vector<Point2f> > aux_points(2);
    vector<Point2f> aux_points2(2);
    srand((uint32_t)getTickCount());

    aux_points[0] = vector<Point2f>(4);
    aux_points[1] = vector<Point2f>(4);

    points[0] = vector<Point2f>(4);
    points[1] = vector<Point2f>(4);

    for (int i=0; i<_niter; i++) {
        for (int j=0; j<4; j++) {
            rnd_frame = rand() % _ransac_mosaics[0]->n_frames;
            rnd_point = rand() % _ransac_mosaics[0]->frames[rnd_frame]->keypoints_pos[!rnd_frame].size();

            points[0][j] = _ransac_mosaics[0]->frames[rnd_frame]->keypoints_pos[!rnd_frame][rnd_point];
            points[1][j] = _ransac_mosaics[1]->frames[rnd_frame]->keypoints_pos[!rnd_frame][rnd_point];

            mid_points[j] = getMidPoint(points[0][j], points[1][j]);
        }

        temp_H = getPerspectiveTransform(points[0], mid_points);

        for (Frame *frame: _ransac_mosaics[0]->frames) {
            perspectiveTransform(frame->bound_points[FIRST], frame->bound_points[RANSAC], temp_H*frame->H);
        }

        temp_distortion = _ransac_mosaics[0]->calcDistortion();

        if (temp_distortion < distortion) {
            distortion = temp_distortion;
            aux_points = points;
            aux_points2 = mid_points;
            best_H = temp_H;
        }
    }

    delete _ransac_mosaics[1];

    return best_H;
}
// See description in header file
void Mosaic::alignMosaics(vector<SubMosaic *> &_sub_mosaics){
    Point2f centroid_0 = _sub_mosaics[0]->getCentroid();
    Point2f centroid_1 = _sub_mosaics[1]->getCentroid();

    vector<float> offset(4);

    offset[TOP]  = max(centroid_1.y - centroid_0.y, 0.f);
    offset[LEFT] = max(centroid_1.x - centroid_0.x, 0.f);
    _sub_mosaics[0]->updateOffset(offset);

    offset[TOP]  = max(centroid_0.y - centroid_1.y, 0.f);
    offset[LEFT] = max(centroid_0.x - centroid_1.x, 0.f);
    _sub_mosaics[1]->updateOffset(offset);

    Mat points[2];

    for (int i=0; i<_sub_mosaics.size(); i++) {
        points[i] = Mat(_sub_mosaics[i]->frames[1]->keypoints_pos[NEXT]);
        for (int j=1; j<_sub_mosaics[i]->frames.size(); j++) {
            vconcat(points[i], Mat(_sub_mosaics[i]->frames[1]->keypoints_pos[PREV]), points[i]);
        }
    }

    Mat M = estimateRigidTransform(points[0], points[1], false);
    double sx = sign(M.at<double>(0,0))*sqrt(pow(M.at<double>(0,0), 2) + pow(M.at<double>(0,1), 2));
    double sy = sign(M.at<double>(1,1))*sqrt(pow(M.at<double>(1,0), 2) + pow(M.at<double>(1,1), 2));

    M.at<double>(0,0) /= sx;
    M.at<double>(0,1) /= sx;
    M.at<double>(1,0) /= sy;
    M.at<double>(1,1) /= sy;
    M.at<double>(2,0) = 0;
    M.at<double>(2,1) = 0;

    Mat t = (Mat1d(1,3) << 0.0, 0.0, 1.0);
    vconcat(M, t, M);

    for (Frame *frame: _sub_mosaics[0]->frames) {
        frame->setHReference(M);
    }
    _sub_mosaics[0]->avg_H = M * _sub_mosaics[1]->avg_H ;
}

// See description in header file
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