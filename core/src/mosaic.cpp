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
Mosaic::Mosaic(bool _pre)
{
	apply_pre = _pre;
	n_subs = 0;
	n_mosaics = 0;
	tot_frames = 0;
}

void Mosaic::feed(Mat _img)
{
	Frame *new_frame = new Frame(_img.clone(), apply_pre);
	if (stitcher->detectFeatures(new_frame))
		frames.push_back(new_frame);
	cout << flush << "\rTotal images: " << frames.size();
}

// See description in header file
void Mosaic::compute(int _mode)
{
	int best_frame, k;
	Mat H;
	float distortion, best_distortion;
	vector<Point2f> best_grid_points;
	sub_mosaics.push_back(new SubMosaic());
	sub_mosaics[0]->addFrame(frames[0]);
	for (int i = 0; i<frames.size()-1; i++)
	{
		best_distortion = 100;
		distortion = 100;
		best_frame = i+2;
		for (k =i+1; k < frames.size() && k < i+3; k++)
		{
			H = stitcher->stitch(frames[k], frames[i]);
			if (!H.empty())
			{
				frames[k]->setHReference(H, PERSPECTIVE);
				distortion = frames[k]->frameDistortion(PERSPECTIVE);
				if (distortion < best_distortion)
				{
					best_frame = k;
					best_distortion = distortion;
					best_grid_points = frames[i]->grid_points[NEXT];
				}
				else
					frames[k]->resetFrame();
				
			}
		}
		for (int j = best_frame-1; j>i; j--)
		{
			delete frames[j];
			frames.erase(frames.begin() + j);
		}
		if (best_distortion == 100)
		{
			final_mosaics.push_back(sub_mosaics);
			n_mosaics++;
			sub_mosaics.clear();
			n_subs = 0;
			if (i < frames.size()-2)
				sub_mosaics[n_subs]->addFrame(frames[i+1]);
			else
				break;
		}
		else
		{
			frames[i]->grid_points[NEXT] = best_grid_points;
			best_grid_points.clear();
			if (frames[i + 1]->isGoodFrame())
			{
				sub_mosaics[n_subs]->addFrame(frames[i + 1]);
			}
			else
			{
				sub_mosaics[n_subs]->avg_H.release();
				sub_mosaics[n_subs]->avg_H = frames[i+1]->H.clone();
				sub_mosaics.push_back(new SubMosaic());
				n_subs++;
				frames[i+1]->resetFrame();
				sub_mosaics[n_subs]->addFrame(frames[i+1]);
			}
		}
	}
	if (sub_mosaics.size() > 1 || sub_mosaics[n_subs]->n_frames > 1)
	{
		final_mosaics.push_back(sub_mosaics);
		n_mosaics++;
		sub_mosaics.clear();
	}

	float overlap;
	for (vector<SubMosaic *> final_mosaic : final_mosaics)
	{
		for (SubMosaic *sub_mosaic : final_mosaic)
		{
			Hierarchy aux_1, aux_2;
			overlap = getOverlap(sub_mosaics[n_subs-1], sub_mosaics[n_subs]);
			aux_1.overlap = overlap;
			aux_1.mosaic = sub_mosaics[n_subs-1];
			aux_2.overlap = overlap;
			aux_2.mosaic = sub_mosaics[n_subs];
			sub_mosaics[n_subs]->neighbors.push_back(aux_1);
			sub_mosaics[n_subs-1]->neighbors.push_back(aux_2);
		}
	}
	merge();
}

// See description in header file
void Mosaic::merge()
{
	vector<SubMosaic *> ransac_mosaics(2);

	float best_overlap = 0;
	
	for (vector<SubMosaic *> final_mosaic : final_mosaics)
	{
		while(final_mosaic.size() > 2)
		{
			best_overlap = -1;
			for (SubMosaic *sub_mosaic : final_mosaic)
			{
				for (Hierarchy neighbor: sub_mosaic->neighbors)
					if (neighbor.overlap > best_overlap)
					{
						best_overlap = neighbor.overlap;
						ransac_mosaics[0] = sub_mosaic;
						ransac_mosaics[1] = neighbor.mosaic;
					}
			}
			ransac_mosaics[0] = sub_mosaics[0];
			ransac_mosaics[1] = sub_mosaics[1];
			getReferencedMosaics(ransac_mosaics);
			alignMosaics(ransac_mosaics);
			Mat best_H = getBestModel(ransac_mosaics, 4000);

			for (Frame *frame : ransac_mosaics[0]->frames)
				frame->setHReference(best_H);
			ransac_mosaics[0]->avg_H = best_H * ransac_mosaics[0]->avg_H;

			for (int i=0; i<ransac_mosaics[0]->neighbors.size(); i++)
				if (ransac_mosaics[1] == ransac_mosaics[0]->neighbors[i].mosaic)
				{
					ransac_mosaics[0]->neighbors.erase(ransac_mosaics[0]->neighbors.begin()+i);
					for (int j=0; j<ransac_mosaics[1]->neighbors.size(); j++)
						if (ransac_mosaics[1]->neighbors[j].mosaic != ransac_mosaics[0])
							ransac_mosaics[0]->neighbors.push_back(ransac_mosaics[1]->neighbors[j]);
				}
			
			for (int i=0; i<final_mosaic.size(); i++)
				if (ransac_mosaics[1] == final_mosaic[i])
					final_mosaic.erase(final_mosaic.begin() + i);

			delete ransac_mosaics[1];
		}
	}
}

// See description in header file
void Mosaic::getReferencedMosaics(vector<SubMosaic *> &_sub_mosaics)
{
	Mat ref_H = _sub_mosaics[0]->avg_H.clone();
	Mat ref_E = _sub_mosaics[0]->avg_E.clone();

	_sub_mosaics[1]->referenceToZero();
	Mat new_ref_H = ref_H * _sub_mosaics[1]->avg_H.clone();
	Mat new_ref_E = ref_E * _sub_mosaics[1]->avg_E.clone();

	for (Frame *frame : _sub_mosaics[1]->frames)
	{
		frame->setHReference(ref_H, PERSPECTIVE);
		frame->setHReference(ref_E, EUCLIDEAN);
		_sub_mosaics[0]->addFrame(frame);
	}
	new_ref_H = ref_H * new_ref_H;
	new_ref_E = ref_E * new_ref_E;

	_sub_mosaics[1] = _sub_mosaics[0]->clone();
	for (Frame *frame : _sub_mosaics[1]->frames)
	{
		frame->setHReference(ref_H.inv(), PERSPECTIVE);
		frame->setHReference(ref_E.inv(), EUCLIDEAN);
	}

	_sub_mosaics[0]->avg_H = new_ref_H.clone();
	_sub_mosaics[0]->avg_E = new_ref_E.clone();
}

// See description in header file
Mat Mosaic::getBestModel(vector<SubMosaic *> &_ransac_mosaics, int _niter)
{
	int rnd_frame, rnd_point;
	Mat temp_H, best_H;
	float distortion = 100;
	float temp_distortion;
	vector<vector<Point2f>> points(2);
	vector<Point2f> mid_points(4);

	srand((uint32_t)getTickCount());

	points[0] = vector<Point2f>(4);
	points[1] = vector<Point2f>(4);

	for (int i = 0; i < _niter; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			rnd_frame = rand() % _ransac_mosaics[0]->n_frames;
			rnd_point = rand() % _ransac_mosaics[0]->frames[rnd_frame]->grid_points[!rnd_frame].size();

			points[0][j] = _ransac_mosaics[0]->frames[rnd_frame]->grid_points[!rnd_frame][rnd_point];
			points[1][j] = _ransac_mosaics[1]->frames[rnd_frame]->grid_points[!rnd_frame][rnd_point];

			mid_points[j] = getMidPoint(points[0][j], points[1][j]);
		}

		temp_H = getPerspectiveTransform(points[0], mid_points);

		for (Frame *frame : _ransac_mosaics[0]->frames)
		{
			perspectiveTransform(frame->bound_points[PERSPECTIVE],
								 frame->bound_points[RANSAC], temp_H * frame->H);
		}

		temp_distortion = _ransac_mosaics[0]->calcDistortion();

		if (temp_distortion < distortion)
		{
			distortion = temp_distortion;
			best_H = temp_H;
		}
	}

	//delete _ransac_mosaics[0];

	return best_H;
}
// See description in header file
void Mosaic::alignMosaics(vector<SubMosaic *> &_sub_mosaics)
{
	Point2f centroid_0 = _sub_mosaics[0]->getCentroid();
	Point2f centroid_1 = _sub_mosaics[1]->getCentroid();

	vector<float> offset(4);

	offset[TOP] = max(centroid_1.y - centroid_0.y, 0.f);
	offset[LEFT] = max(centroid_1.x - centroid_0.x, 0.f);
	_sub_mosaics[0]->updateOffset(offset);

	offset[TOP] = max(centroid_0.y - centroid_1.y, 0.f);
	offset[LEFT] = max(centroid_0.x - centroid_1.x, 0.f);
	_sub_mosaics[1]->updateOffset(offset);
	
	Mat points[2];

	for (int i = 0; i < _sub_mosaics.size(); i++)
	{
		points[i] = Mat(_sub_mosaics[i]->frames[0]->grid_points[NEXT]);
		for (int j = 1; j < _sub_mosaics[i]->frames.size(); j++)
		{
			vconcat(points[i], Mat(_sub_mosaics[i]->frames[j]->grid_points[PREV]), points[i]);
		}
	}

	Mat R = estimateRigidTransform(points[1], points[0], false);
	Mat M = Mat::eye(3, 3, CV_64F);
	R.copyTo(M(Rect(0, 0, 3, 2)));
	removeScale(M);

	for (Frame *frame : _sub_mosaics[1]->frames)
	{
		frame->setHReference(M);
	}
	_sub_mosaics[1]->avg_H = M * _sub_mosaics[0]->avg_H;

}

float Mosaic::getOverlap(SubMosaic *_object, SubMosaic *_scene)
{

}

// See description in header file
void Mosaic::print()
{
	int n=0;
	for (SubMosaic *sub_mosaic: sub_mosaics)
	{
		blender->blendSubMosaic(sub_mosaic);
		imwrite("/home/victor/dataset/output/final-000"+to_string(n++)+".jpg", sub_mosaic->final_scene);
		resize(sub_mosaic->final_scene, sub_mosaic->final_scene, Size(1066, 800));
		imshow("Blend-Ransac-Final", sub_mosaic->final_scene);
		waitKey(0);
	}
}
}