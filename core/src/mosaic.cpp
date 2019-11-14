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
void Mosaic::feed(Mat _img)
{
	if (this->to_calibrate) {
		Mat temp = _img.clone();
		//cout<< this->distortion_coeff << endl;
		undistort(temp, _img, this->camera_matrix, this->distortion_coeff);
	}
	//cout<< this->distortion_coeff << endl;
	// create and store new frame object
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
	// detect and compute features for all images
	stitcher->detectFeatures(frames);
	//create initial sub mosaic and add it the first frame
	cout << "Creating initial submosaic" << endl;
	sub_mosaics.push_back(new SubMosaic());
	sub_mosaics[0]->addFrame(frames[0]);
	// loop over all frames to build sub mosaics
	cout << "Looping through all the frames" << endl;
	for (int i = 0; i<frames.size()-1; i++)
	{
		best_distortion = 100;
		distortion = 100;
		best_frame = i+2;
		// search in k-windows for best frame (frame with less distortion)
		// stop if k-frame or last one is reached
		for (k =i+1; k < frames.size() && k < i+2; k++)
		{
			// find perspective and best euclidean transformations
			transform = stitcher->stitch(frames[k], frames[i]);
			cout << "[mosaic] Stitched frames ["<<k<<" + "<< i << "]" << endl;
			// check if transformations are valid
			if (!transform[PERSPECTIVE].empty() && !transform[EUCLIDEAN].empty())
			{
				// if euclidean mode is selected, assign euclidean matrix as default transformation
				if (_euclidean_mode)
					frames[k]->setHReference(transform[EUCLIDEAN], PERSPECTIVE);
				else
					frames[k]->setHReference(transform[PERSPECTIVE], PERSPECTIVE);
				// save euclidean transformation (for global euclidean correction)
				frames[k]->setHReference(transform[EUCLIDEAN], EUCLIDEAN);
				// compute frame distortion (based on perspective transformation)
				distortion = frames[k]->frameDistortion(PERSPECTIVE);
				if (distortion < best_distortion)
				{
					// save frame index, and points from scene to object
					best_frame = k;
					best_distortion = distortion;
					best_grid_points = frames[i]->grid_points[NEXT];
				}
				// if the frame is not the best, reset to default values
				else
					frames[k]->resetFrame();
				
			}
		}
		// delete frames previous to best one (if best is next to scene+1)
		for (int j = best_frame-1; j>i; j--)
		{
			delete frames[j];
			frames.erase(frames.begin() + j);
		}
		// if couldn't reach a best frame, cut the sub mosaic build.
		// and start a new mosaic. (Separate mosaics)
		if (best_distortion == 100)
		{
			//break;
			// save all sub mosaics to be merged together
			if (sub_mosaics.size() > 0)
				final_mosaics.push_back(sub_mosaics);
			sub_mosaics.clear();
			sub_mosaics.push_back(new SubMosaic());
			n_subs = 0;
			// if remains enough frames, continue. else break loop
			if (i < frames.size()-2)
				sub_mosaics[n_subs]->addFrame(frames[i+1]);
			else
				break;
		}
		// save best frame
		else
		{
			// save points to best frame in scene.
			frames[i]->grid_points[NEXT] = best_grid_points;
			best_grid_points.clear();
			// if the new frame isn't too distorted, save it in the current sub mosaic
			if (frames[i + 1]->isGoodFrame())
			{
				sub_mosaics[n_subs]->addFrame(frames[i + 1]);
			}
			// else create a new one, and assign it as reference frame
			else
			{
				// save transformations after last frame in current sub mosaic
				sub_mosaics[n_subs]->next_H.release();
				sub_mosaics[n_subs]->next_H = frames[i+1]->H.clone();
				sub_mosaics[n_subs]->next_E.release();
				sub_mosaics[n_subs]->next_E = frames[i+1]->E.clone();
				// create sub mosaic, reset frame, and save it.
				sub_mosaics.push_back(new SubMosaic());
				n_subs++;
				frames[i+1]->resetFrame();
				sub_mosaics[n_subs]->addFrame(frames[i+1]);
			}
		}
		cout<<"\r[mosaic] Building sub-mosaics:\t[" <<green<<((i+2)*100)/frames.size()<<reset<<"%]"<<flush;
	}
	cout << endl;
	cout << "[mosaic] Saving resulting submosaics before merge" << endl;
	// save resulting sub mosaics to be merged together
	if (sub_mosaics[n_subs]->n_frames > 0 )
	{
		final_mosaics.push_back(sub_mosaics);
		sub_mosaics.clear();
	}

	cout << "[mosaic] Updating overlap for final mosaic" << endl;
	// compute the overlap between sub mosaics (hierarchy to merge)
	for (vector<SubMosaic *> &final_mosaic : final_mosaics)
		updateOverlap(final_mosaic);
	// merge sub mosaics
	cout << "[mosaic] Merging" << endl;
}

// See description in header file
void Mosaic::merge(bool _euclidean_correction)
{
	float best_overlap = 0;
	vector<SubMosaic *> ransac_mosaics(2);
	cout<<flush<<"\rMerging sub-mosaics:\t[" <<green<<0<<reset<<"%]"<<flush;
	// loop over all mosaics
	// a mosaic is a vector with all sub mosaics that can be merged
	for (int n=0; n<final_mosaics.size(); n++)
	{
		int w=0;
		// loop over sub mosaics of same mosaic

		while(final_mosaics[n].size() > 1)
		{
			cout << "\n****************************************\n\t----> final_mosaics[n].size(): " << final_mosaics[n].size() << endl;
			cout << "getBestOverlap" << endl;
			// get the sub mosaic pair with higher overlap
			ransac_mosaics = getBestOverlap(final_mosaics[n]);
			// remove second sub mosaic, and update neighbors
			cout << "removeNeighbor" << endl;
			cout << "ransac_mosaics.size(): " << ransac_mosaics.size() << endl;
			//cout << "ransac_mosaics[0]->size(): " << ransac_mosaics[0]->neighbors.size() << endl;
			// second sub mosaic will be merged and save it in the first one

			cout << "ransac_mosaics[0].neighbors.size(): " << ransac_mosaics[0]->neighbors.size() << endl;

			removeNeighbor(ransac_mosaics);
			// remove second sub mosaic from array
//			cout << "final_mosaics[n].erase" << endl;
			for (int i=0; i<final_mosaics[n].size(); i++)
				{
				cout << "for: i/n" << i << " / " << final_mosaics[n].size() << endl;
				if (ransac_mosaics[1] == final_mosaics[n][i])
					final_mosaics[n].erase(final_mosaics[n].begin() + i);
				}
			// join two sub mosaics based on each reference transformation
//blender->blendSubMosaic(ransac_mosaics[0]);
//blender->blendSubMosaic(ransac_mosaics[1]);
//imwrite("/home/ros/dataset/output/temp/SR-Unref0-"+to_string(w)+".png", ransac_mosaics[0]->final_scene);
//imwrite("/home/ros/dataset/output/temp/SR-Unref1-"+to_string(w)+".png", ransac_mosaics[1]->final_scene);
			cout << "referenceMosaics" << endl;
			referenceMosaics(ransac_mosaics);
//blender->blendSubMosaic(ransac_mosaics[0]);
//blender->blendSubMosaic(ransac_mosaics[1]);
//imwrite("/home/ros/dataset/output/temp/SR-Ref0-"+to_string(w)+".png", ransac_mosaics[0]->final_scene);
//imwrite("/home/ros/dataset/output/temp/SR-Ref1-"+to_string(w)+".png", ransac_mosaics[1]->final_scene);
//ransac_mosaics[0]->final_scene.release();
//ransac_mosaics[1]->final_scene.release();
			// align sub mosaics (translate and rotate)
			cout << "alignMosaics" << endl;
			alignMosaics(ransac_mosaics);
			// find transformation which minimize overall geometric distortion
			Mat best_H = getBestModel(ransac_mosaics, 4000);
			// apply best transformation on first sub mosaic
			for (Frame *frame : ransac_mosaics[0]->frames)
				frame->setHReference(best_H);
			ransac_mosaics[0]->next_H = best_H * ransac_mosaics[0]->next_H;
//blender->blendSubMosaic(ransac_mosaics[0]);
//imwrite("/home/ros/dataset/output/temp/SR-avg-"+to_string(w++)+".png", ransac_mosaics[0]->final_scene);
			// delete second sub mosaic
			delete ransac_mosaics[1];
			// update overlap between remains sub mosaics
			updateOverlap(final_mosaics[n]);
		}
		//blender->blendSubMosaic(ransac_mosaics[0]);
		//imshow("avg", ransac_mosaics[0]->final_scene);
		//waitKey(0);
		cout<<"\rMerging sub-mosaics:\t["<<green<<((n+1)*100)/final_mosaics.size()<<reset<<"%]"<<flush;
//blender->blendSubMosaic(final_mosaics[n][0]);
//imwrite("/home/ros/dataset/output/0233-closure-simple_UnCorrect.png", final_mosaics[n][0]->final_scene);
		// apply global euclidean correction
		if (_euclidean_correction)
			final_mosaics[n][0]->correct();
	}
	cout << endl;
}

// See description in header file
vector<SubMosaic *> Mosaic::getBestOverlap(vector<SubMosaic *> _sub_mosaics)
{
	cout << "[getBestOverlap]" << endl;

	float best_overlap = 0;
	vector<SubMosaic *> ransac_mosaics(2);
	// loop over all sub mosaics
	for (SubMosaic *sub_mosaic : _sub_mosaics)
	{
//		cout << "[getBestOverlap] loop sub_mosaic" << endl;
		// loop over all neighbors
		for (Hierarchy neighbor: sub_mosaic->neighbors)
		{
//			cout << "[getBestOverlap] loop neighbor" << endl;
			// find the neighbor with higher overlap
			if (neighbor.overlap > best_overlap)
			{
//				cout << "[getBestOverlap] find best overlap" << endl;
				best_overlap = neighbor.overlap;
				// save the neighbor and the sub mosaic
				ransac_mosaics[0] = sub_mosaic->clone();	//TODO: ERROR: SEGFAULT IF NOT CLONED
				ransac_mosaics[1] = neighbor.mosaic;
			}
		}
	}
//	cout << "[getBestOverlap] END" << endl;
	cout << "[getBestOverlap] ransac_mosaics[0].neighbors.size()" << ransac_mosaics[0]->neighbors.size() << endl;

	return ransac_mosaics;
}

// See description in header file
void Mosaic::removeNeighbor(vector<SubMosaic *> &_sub_mosaics){
	// loop over SM0 neighbors
//	cout << "[mosaic][removeNeighbor] start..." << endl;
	cout << "[mosaic][removeNeighbor] _sub_mosaics.size(): " << _sub_mosaics.size() << endl;	
	cout << "[mosaic][removeNeighbor] _sub_mosaics[0]->neighbors.size(): " << _sub_mosaics[0]->neighbors.size() << endl;
	for (int i=0; i<_sub_mosaics[0]->neighbors.size(); i++)
	{
		// find the neighbor link to SM1
		if (_sub_mosaics[1] == _sub_mosaics[0]->neighbors[i].mosaic)
		{
			// erase it
			_sub_mosaics[0]->neighbors.erase(_sub_mosaics[0]->neighbors.begin()+i);
			// loop over SM1 neighbors
			for (int j=0; j<_sub_mosaics[1]->neighbors.size(); j++)
			{
				// find the neighbors different to SM0
				if (_sub_mosaics[1]->neighbors[j].mosaic != _sub_mosaics[0])
				{
					// for this neighbor, update pointer of SM2. from SM1 to SM0
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
	// locate first frame at default position
	Mat key_H = _sub_mosaics[1]->frames[0]->getResetTransform(PERSPECTIVE);
	Mat key_E = _sub_mosaics[1]->frames[0]->getResetTransform(EUCLIDEAN);
	// apply transformation
	for (Frame *frame : _sub_mosaics[1]->frames)
	{
		frame->setHReference(key_H, PERSPECTIVE);
		frame->setHReference(key_E, EUCLIDEAN);
	}
	_sub_mosaics[1]->next_H = key_H * _sub_mosaics[1]->next_H;
	_sub_mosaics[1]->next_E = key_E * _sub_mosaics[1]->next_E;
	// ref_H transform a frame from default position to last position on sub mosaic (as next frame)
	Mat ref_H = _sub_mosaics[0]->next_H.clone();
	Mat ref_E = _sub_mosaics[0]->next_E.clone();
	// update good neighbors (this data was deleted in reset frame)
	// TODO: create new linker, such as: sub mosaic neighbors
	_sub_mosaics[0]->last_frame->good_neighbors.push_back(_sub_mosaics[1]->frames[0]);
	_sub_mosaics[1]->frames[0]->good_neighbors.push_back(_sub_mosaics[0]->last_frame);
	// add all SM1 frames to SM0, by next_H of SM0
	for (Frame *frame : _sub_mosaics[1]->frames)
	{
		frame->setHReference(ref_H, PERSPECTIVE);
		frame->setHReference(ref_E, EUCLIDEAN);
		_sub_mosaics[0]->addFrame(frame);
	}
	// now, the new "next_H" will be the next_H from SM1 (now SM1 have the last frame of tot SM)
	_sub_mosaics[0]->next_H = ref_H * _sub_mosaics[1]->next_H;
	_sub_mosaics[0]->next_E = ref_E * _sub_mosaics[1]->next_E;
	// clone SM0 into SM1
	_sub_mosaics[1] = _sub_mosaics[0]->clone();
	// align SM1 by it's reference transformation (first frame of SM1 as reference)
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
	// save points from two sub mosaics
	for (int i = 0; i < _sub_mosaics.size(); i++)
	{
		points[i] = Mat(_sub_mosaics[i]->frames[0]->grid_points[NEXT]);
		for (int j = 1; j < _sub_mosaics[i]->frames.size(); j++)
		{
			vconcat(points[i], Mat(_sub_mosaics[i]->frames[j]->grid_points[PREV]), points[i]);
		}
	}
	// find best similarity transformation (rotation and translation). from SM1 to SM0
	Mat R = estimateRigidTransform(points[1], points[0], false);
	if (!R.empty())
	{
		Mat M = Mat::eye(3, 3, CV_64F);
		// copy only rotate matrix
		R(Rect(0, 0, 2, 2)).copyTo(M(Rect(0, 0, 2, 2)));
		// remove scale factor
		removeScale(M);
		// apply rotation matrix to SM1
		for (Frame *frame : _sub_mosaics[1]->frames)
		{
			frame->setHReference(M, PERSPECTIVE);
		}
		_sub_mosaics[1]->next_H = M * _sub_mosaics[1]->next_H;
	}
	// get the centroid of each SM. (average position of key points)
	Point2f centroid_0 = _sub_mosaics[0]->getCentroid();
	Point2f centroid_1 = _sub_mosaics[1]->getCentroid();

	vector<float> offset(4);
	// translate SM0
	offset[TOP] = max(centroid_1.y - centroid_0.y, 0.f);
	offset[LEFT] = max(centroid_1.x - centroid_0.x, 0.f);
	_sub_mosaics[0]->updateOffset(offset);
	// translate SM1
	offset[TOP] = max(centroid_0.y - centroid_1.y, 0.f);
	offset[LEFT] = max(centroid_0.x - centroid_1.x, 0.f);
	_sub_mosaics[1]->updateOffset(offset);
}

// See description in header file
Mat Mosaic::getBestModel(vector<SubMosaic *> _ransac_mosaics, int _niter)
{
	int rnd_frame, rnd_point;
	float distortion = 100;
	float temp_distortion;
	Mat temp_H, best_H;
	vector<vector<Point2f>> points(2);
	vector<Point2f> mid_points(4);
	// initialize seed for random variable
	srand((uint32_t)getTickCount());
	points[0] = vector<Point2f>(4);
	points[1] = vector<Point2f>(4);
	// start iteration
	for (int i = 0; i < _niter; i++)
	{
		// find four corresponding points
		for (int j = 0; j < 4; j++)
		{
			// select a random frame
			rnd_frame = rand() % _ransac_mosaics[0]->n_frames;
			// select a random point number
			// grid_points[0=PREV] to frames different to first, will be select match points to prev frame
			// grid_points[1=NEXT] (!rnd_frame) because first frame not have points to prev frame
			rnd_point = rand() % _ransac_mosaics[0]->frames[rnd_frame]->grid_points[!rnd_frame].size();
			// save selected point
			points[0][j] = _ransac_mosaics[0]->frames[rnd_frame]->grid_points[!rnd_frame][rnd_point];
			points[1][j] = _ransac_mosaics[1]->frames[rnd_frame]->grid_points[!rnd_frame][rnd_point];
			// get the mid point from correspondence points
			mid_points[j] = getMidPoint(points[0][j], points[1][j]);
		}
		// find perspective transformation from SM0  points to midpoints
		temp_H = getPerspectiveTransform(points[0], mid_points);
if (i==0)
{
	temp_H = Mat::eye(3, 3, CV_64F);
}
		// apply transformation to corner points of each frame
		for (Frame *frame : _ransac_mosaics[0]->frames)
		{
			// save it in temporal frame object variable
			perspectiveTransform(frame->bound_points[PERSPECTIVE],
								 frame->bound_points[RANSAC], temp_H);// old: temp_Hxframe->H *******************
		}
		// calculate the overall geometric distortion
		temp_distortion = _ransac_mosaics[0]->calcDistortion(RANSAC);
		if (temp_distortion < distortion)
		{
			// save best model
			distortion = temp_distortion;
			best_H.release();
			best_H = temp_H.clone();
		}
	}
	return best_H;
}

// See description in header file
void Mosaic::updateOverlap(vector<SubMosaic *> &_sub_mosaic)
{
	float overlap;
	for (SubMosaic *sub : _sub_mosaic)
		sub->neighbors.clear();
	
	for (int i=0; i<_sub_mosaic.size()-1;i++)
	{
		// calculate overlap between each SM pair
		Hierarchy link_1, link_2;
		overlap = getOverlap(_sub_mosaic[i+1], _sub_mosaic[i]);
		link_1.overlap = overlap;
		link_1.mosaic = _sub_mosaic[i+1];
		link_2.overlap = overlap;
		link_2.mosaic = _sub_mosaic[i];
		_sub_mosaic[i]->neighbors.push_back(link_1);
		_sub_mosaic[i+1]->neighbors.push_back(link_2);
	}
}

// See description in header file
float Mosaic::getOverlap(SubMosaic *_object, SubMosaic *_scene)
{
	vector<Point2f> object_points;
	Mat reset_H = _object->frames[0]->getResetTransform(PERSPECTIVE);
	// locate corner (of _object first frame) points to default position
	perspectiveTransform(_object->frames[0]->bound_points[PERSPECTIVE], object_points, reset_H);
	// locate corner (of _object first frame) points next to last frame of scene	
	perspectiveTransform(object_points, object_points, _scene->next_H);
	// get the bounding rectangle of points
	Rect2f object_bound_rect = boundingRectFloat(object_points);
	// calculate the width of overlap rectangle
	float width = min(_scene->last_frame->bound_rect.x + _scene->last_frame->bound_rect.width,
				  object_bound_rect.x + object_bound_rect.width) - 
				  max(_scene->last_frame->bound_rect.x, object_bound_rect.x);
	// calculate the height of overlap rectangle
	float height = min(_scene->last_frame->bound_rect.y + _scene->last_frame->bound_rect.height,
				   object_bound_rect.y + object_bound_rect.height) - 
				   max(_scene->last_frame->bound_rect.y, object_bound_rect.y);
	// calculate the normalized overlap area
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
	cout << "[mosaic] Saving mosaics: " << final_mosaics.size() << endl;
	for (vector<SubMosaic *> final_mosaic: final_mosaics)
	{
		// blend mosaic and save it, using provided filename
		blender->blendSubMosaic(final_mosaic[n]);
		imwrite(_dir+"-000"+to_string(n)+".jpg", final_mosaic[n]->final_scene);
		// create track map and save it
		map.push_back(final_mosaic[n]->buildMap(CIRCLE));
		imwrite(_dir+"-"+to_string(n)+"-MAP.jpg", map[n]);
		n++;
	}
}

// See description in header file
void Mosaic::show()
{
	int n=0;
	int height=800;
	float resize_factor;
	for (vector<SubMosaic *> final_mosaic: final_mosaics)
	{
		// resize the preview image (mosaic will be blend)
		resize_factor = (float)height / final_mosaic[n]->final_scene.rows;
		resize(final_mosaic[n]->final_scene, final_mosaic[n]->final_scene,
				Size(round(final_mosaic[n]->final_scene.cols * resize_factor),
				round(final_mosaic[n]->final_scene.rows * resize_factor)));
		imshow("Final Mosaic - "+to_string(n), final_mosaic[n]->final_scene);
		imshow("Final Map - "+to_string(n), map[n]);
				
		waitKey(0);
	}
}

void Mosaic::SetCameraMatrix(cv::Mat _camera_matrix, cv::Mat _distortion_coeff)
{
	this->camera_matrix = _camera_matrix.clone();
	this->distortion_coeff = _distortion_coeff.clone();

	if(this->camera_matrix.empty()){
		this->to_calibrate = false;
	}
	else {
		this->to_calibrate = true;
	}
}

}
