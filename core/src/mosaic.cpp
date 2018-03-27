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

void Mosaic::feed(Mat _img)
{
	Frame *new_frame = new Frame(_img.clone(), apply_pre);
	frames.push_back(new_frame);
	cout << flush << "\rTotal images:\t\t[" <<green<< frames.size()<<reset<<"]";
}

// See description in header file
void Mosaic::compute(bool _euclidean_mode)
{
	int best_frame, k;
	float distortion, best_distortion;
	vector<Point2f> best_grid_points;
	vector<Mat> transform(2);

	stitcher->detectFeatures(frames);
	sub_mosaics.push_back(new SubMosaic());
	sub_mosaics[0]->addFrame(frames[0]);
	for (int i = 0; i<frames.size()-1; i++)
	{
		best_distortion = 100;
		distortion = 100;
		best_frame = i+2;
		for (k =i+1; k < frames.size() && k < i+3; k++)
		{
			transform = stitcher->stitch(frames[k], frames[i]);
			if (!transform[PERSPECTIVE].empty() && !transform[EUCLIDEAN].empty())
			{
				if (_euclidean_mode)
					frames[k]->setHReference(transform[EUCLIDEAN], PERSPECTIVE);
				else
					frames[k]->setHReference(transform[PERSPECTIVE], PERSPECTIVE);
				
				frames[k]->setHReference(transform[EUCLIDEAN], EUCLIDEAN);	
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
				sub_mosaics[n_subs]->next_H.release();
				sub_mosaics[n_subs]->next_H = frames[i+1]->H.clone();
				sub_mosaics[n_subs]->next_E.release();
				sub_mosaics[n_subs]->next_E = frames[i+1]->E.clone();
				
				sub_mosaics.push_back(new SubMosaic());
				n_subs++;
				frames[i+1]->resetFrame();
				sub_mosaics[n_subs]->addFrame(frames[i+1]);
			}
		}
		cout<<flush<<"\rBuilding sub-mosaics:\t[" <<green<<((i+2)*100)/frames.size()<<reset<<"%]";
	}
	cout<<endl;
	if (sub_mosaics.size() > 1 || sub_mosaics[n_subs]->n_frames > 1)
	{
		final_mosaics.push_back(sub_mosaics);
		sub_mosaics.clear();
	}

	for (vector<SubMosaic *> &final_mosaic : final_mosaics)
		updateOverlap(final_mosaic);
	
	merge(_euclidean_mode);
}

// See description in header file
void Mosaic::merge(bool _euclidean_mode)
{
	float best_overlap = 0;
	vector<SubMosaic *> ransac_mosaics(2);
	cout<<flush<<"\rMerging sub-mosaics:\t[" <<green<<0<<reset<<"%]";
	for (int n=0; n<final_mosaics.size(); n++)
	{
		while(final_mosaics[n].size() > 1)
		{
			ransac_mosaics = getBestOverlap(final_mosaics[n]);

			removeNeighbor(ransac_mosaics);

			for (int i=0; i<final_mosaics[n].size(); i++)
				if (ransac_mosaics[1] == final_mosaics[n][i])
					final_mosaics[n].erase(final_mosaics[n].begin() + i);

			referenceMosaics(ransac_mosaics);
			alignMosaics(ransac_mosaics);
			Mat best_H = getBestModel(ransac_mosaics, 4000);

			for (Frame *frame : ransac_mosaics[0]->frames)
				frame->setHReference(best_H);
			ransac_mosaics[0]->next_H = best_H * ransac_mosaics[0]->next_H;

			delete ransac_mosaics[1];

			updateOverlap(final_mosaics[n]);
		}
		cout<<flush<<"\rMerging sub-mosaics:\t["<<green<<((n+1)*100)/final_mosaics.size()<<reset<<"%]";
		if (!_euclidean_mode)
			final_mosaics[n][0]->correct();
	}
	cout<<endl;
}

vector<SubMosaic *> Mosaic::getBestOverlap(vector<SubMosaic *> _sub_mosaics)
{
	best_overlap = 0;
	vector<SubMosaic *> ransac_mosaics(2);
	for (SubMosaic *sub_mosaic : _sub_mosaic)
	{
		for (Hierarchy neighbor: sub_mosaic->neighbors)
		{
			if (neighbor.overlap > best_overlap)
			{
				best_overlap = neighbor.overlap;
				ransac_mosaics[0] = sub_mosaic;
				ransac_mosaics[1] = neighbor.mosaic;
			}
		}
	}
	return ransac_mosaics;
}

void Mosaic::removeNeighbor(vector<SubMosaic *> &_sub_mosaics){
	for (int i=0; i<_sub_mosaics[0]->neighbors.size(); i++)
	{
		if (_sub_mosaics[1] == _sub_mosaics[0]->neighbors[i].mosaic)
		{
			_sub_mosaics[0]->neighbors.erase(_sub_mosaics[0]->neighbors.begin()+i);
			for (int j=0; j<_sub_mosaics[1]->neighbors.size(); j++)
			{
				if (_sub_mosaics[1]->neighbors[j].mosaic != _sub_mosaics[0])
				{
					_sub_mosaics[0]->neighbors.push_back(_sub_mosaics[1]->neighbors[j]);
					for (int k=0; k<_sub_mosaics[1]->neighbors[j].mosaic->neighbors.size(); k++)
						if (_sub_mosaics[1]->neighbors[j].mosaic->neighbors[k].mosaic == _sub_mosaics[1])
							_sub_mosaics[1]->neighbors[j].mosaic->neighbors[k].mosaic = _sub_mosaics[0];
				}
			}
		}
	}
}

// See description in header file
void Mosaic::referenceMosaics(vector<SubMosaic *> &_sub_mosaics)
{
	Mat key_H = _sub_mosaics[1]->frames[0]->getResetTransform(PERSPECTIVE);
	Mat key_E = _sub_mosaics[1]->frames[0]->getResetTransform(EUCLIDEAN);

	for (Frame *frame : _sub_mosaics[1]->frames)
	{
		frame->setHReference(key_H, PERSPECTIVE);
		frame->setHReference(key_E, EUCLIDEAN);
	}
	_sub_mosaics[1]->next_H = key_H * _sub_mosaics[1]->next_H;
	_sub_mosaics[1]->next_E = key_E * _sub_mosaics[1]->next_E;
	
	Mat ref_H = _sub_mosaics[0]->next_H.clone();
	Mat ref_E = _sub_mosaics[0]->next_E.clone();

	for (Frame *frame : _sub_mosaics[1]->frames)
	{
		frame->setHReference(ref_H, PERSPECTIVE);
		frame->setHReference(ref_E, EUCLIDEAN);
		_sub_mosaics[0]->addFrame(frame);
	}
	_sub_mosaics[0]->next_H = ref_H * _sub_mosaics[1]->next_H;
	_sub_mosaics[0]->next_E = ref_E * _sub_mosaics[1]->next_E;

	_sub_mosaics[1] = _sub_mosaics[0]->clone();

	for (Frame *frame : _sub_mosaics[1]->frames)
	{
		frame->setHReference(ref_H.inv(), PERSPECTIVE);
		frame->setHReference(key_H.inv(), PERSPECTIVE);
		frame->setHReference(ref_E.inv(), EUCLIDEAN);
		frame->setHReference(key_E.inv(), EUCLIDEAN);
	}
}

// See description in header file
void Mosaic::alignMosaics(vector<SubMosaic *> &_sub_mosaics)
{
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
	if (!R.empty())
	{
		Mat M = Mat::eye(3, 3, CV_64F);
		R(Rect(0, 0, 2, 2)).copyTo(M(Rect(0, 0, 2, 2)));
		removeScale(M);

		for (Frame *frame : _sub_mosaics[1]->frames)
		{
			frame->setHReference(M, PERSPECTIVE);
		}
		_sub_mosaics[1]->next_H = M * _sub_mosaics[1]->next_H;
	}

	Point2f centroid_0 = _sub_mosaics[0]->getCentroid();
	Point2f centroid_1 = _sub_mosaics[1]->getCentroid();

	vector<float> offset(4);

	offset[TOP] = max(centroid_1.y - centroid_0.y, 0.f);
	offset[LEFT] = max(centroid_1.x - centroid_0.x, 0.f);
	_sub_mosaics[0]->updateOffset(offset);

	offset[TOP] = max(centroid_0.y - centroid_1.y, 0.f);
	offset[LEFT] = max(centroid_0.x - centroid_1.x, 0.f);
	_sub_mosaics[1]->updateOffset(offset);
}

// See description in header file
Mat Mosaic::getBestModel(vector<SubMosaic *> _ransac_mosaics, int _niter)
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

		temp_distortion = _ransac_mosaics[0]->calcDistortion(RANSAC);

		if (temp_distortion < distortion)
		{
			distortion = temp_distortion;
			best_H.release();
			best_H = temp_H.clone();
		}
	}

	return best_H;
}

void Mosaic::updateOverlap(vector<SubMosaic *> &_sub_mosaic)
{
	float overlap;
	for (int i=0; i<_sub_mosaic.size()-1;i++)
	{
		Hierarchy aux_1, aux_2;
		overlap = getOverlap(_sub_mosaic[i+1], _sub_mosaic[i]);
		aux_1.overlap = overlap;
		aux_1.mosaic = _sub_mosaic[i+1];
		aux_2.overlap = overlap;
		aux_2.mosaic = _sub_mosaic[i];
		_sub_mosaic[i]->neighbors.push_back(aux_1);
		_sub_mosaic[i+1]->neighbors.push_back(aux_2);
	}
}

float Mosaic::getOverlap(SubMosaic *_object, SubMosaic *_scene)
{
	vector<Point2f> object_points;

	perspectiveTransform(_object->frames[0]->bound_points[PERSPECTIVE], object_points, _scene->next_H);
	Rect2f object_bound_rect = boundingRectFloat(object_points);

	float width = min(_scene->last_frame->bound_rect.x + _scene->last_frame->bound_rect.width,
				  object_bound_rect.x + object_bound_rect.width) - 
				  max(_scene->last_frame->bound_rect.x, object_bound_rect.x);

	float height = min(_scene->last_frame->bound_rect.y + _scene->last_frame->bound_rect.height,
				   object_bound_rect.y + object_bound_rect.height) - 
				   max(_scene->last_frame->bound_rect.y, object_bound_rect.y);

	float overlap_area = width * height;
	float object_area = object_bound_rect.width * object_bound_rect.height;
	float scene_area = _scene->last_frame->bound_rect.width * _scene->last_frame->bound_rect.height;
	
	float overlap_norm = overlap_area / (object_area + scene_area - overlap_area);

	return overlap_norm;
}

// See description in header file
void Mosaic::save(string _dir)
{
	int n=0;
	for (vector<SubMosaic *> final_mosaic: final_mosaics)
	{
		blender->blendSubMosaic(final_mosaic[0]);
		imwrite(_dir+"-000.jpg", final_mosaic[0]->final_scene);
		Mat map = final_mosaic[0]->buildMap(CIRCLE);
		imwrite(_dir+"-MAP.jpg", map);
	}
}

void Mosaic::show()
{
	int n=0;
	int height=800;
	float resize_factor;
	for (vector<SubMosaic *> final_mosaic: final_mosaics)
	{
		resize_factor = (float)height / final_mosaic[0]->final_scene.rows;
		resize(final_mosaic[0]->final_scene, final_mosaic[0]->final_scene,
				Size(round(final_mosaic[0]->final_scene.cols * resize_factor),
				round(final_mosaic[0]->final_scene.rows * resize_factor)));

		imshow("Blend-Ransac-Final", final_mosaic[0]->final_scene);
		waitKey(0);
	}
}

}