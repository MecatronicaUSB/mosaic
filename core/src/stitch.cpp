/**
 * @file stitch.cpp
 * @brief Implementation of Stitcher class functions 
 * @version 0.2
 * @date 10/02/2018
 * @author Victor Garcia
 */

#include "../include/stitch.hpp"

using namespace std;
using namespace cv;
using namespace xfeatures2d;

namespace m2d //!< mosaic 2d namespace
{

// See description in header file
Stitcher::Stitcher(bool _grid, int _detector, int _matcher)
{
	// save configuration
	use_grid = _grid;
	cells_div = CELLS_DIV;
	//select input detector
	switch (_detector)
	{
	case USE_KAZE:
	{
		detector = KAZE::create();
		break;
	}
	case USE_AKAZE:
	{
		detector = AKAZE::create();
		break;
	}
	case USE_SIFT:
	{
		detector = SIFT::create();
		break;
	}
	case USE_SURF:
	{
		detector = SURF::create(400);
		break;
	}
	default:
	{
		detector = KAZE::create();
		break;
	}
	}
	// select input matcher
	switch (_matcher)
	{
	case USE_BRUTE_FORCE:
	{
		matcher = BFMatcher::create();
		break;
	}
	case USE_FLANN:
	{
		matcher = FlannBasedMatcher::create();
		break;
	}
	default:
	{
		matcher = FlannBasedMatcher::create();
		break;
	}
	}
}

// See description in header file
void Stitcher::detectFeatures(vector<Frame *> &_frames)
{
	int i=0;
	cout<<endl;
	// loop over all frames
	for (Frame *frame : _frames)
	{
		// detect features
		detector->detectAndCompute(frame->gray, Mat(), frame->keypoints, frame->descriptors);
		// release gray image since wont be used
		frame->gray.release();
		// if not enough key poitns are detected the frame is deleted
		if (!frame->haveKeypoints())
		{
			delete frame;
			_frames.erase(_frames.begin() + i);
		}
		// update status
		cout<<"\rDetecting Features:\t["<<green<<((++i)*100)/_frames.size()<<reset<<"%]"<<flush;
	}
	cout<<endl;

}

// See description in header file
vector<Mat> Stitcher::stitch(Frame *_object, Frame *_scene)
{
	img[OBJECT] = _object;
	img[SCENE] = _scene;

	// to store matches temporarily
	vector<vector<DMatch>> aux_matches;
	// match using desired matcher
	matcher->knnMatch(img[OBJECT]->descriptors, img[SCENE]->descriptors, aux_matches, 2);
	// save in global class variable
	matches.push_back(aux_matches);
	// if scene frame have neighbors, try to match key points with them
	for (Frame *neighbor : img[SCENE]->neighbors)
	{
		aux_matches.clear();
		matcher->knnMatch(img[OBJECT]->descriptors, neighbor->descriptors, aux_matches, 2);
		matches.push_back(aux_matches);
	}
	// initial threshold value for matches outliers selection
	float thresh = 0.8;
	bool good_thresh = true;
	// loop until enough key points are found (>4)
	while (good_thresh)
	{
		// Discard outliers based on euclidean distance between descriptor's vectors
		getGoodMatches(thresh);
		// Apply grid detector if flag is activated
		if (use_grid)
			gridDetector();
		// Update good neighbors (neighbors who have more than 3 strong matches)
		img[OBJECT]->good_neighbors.push_back(img[SCENE]);
		for (int j=1; j<good_matches.size(); j++)
			if (good_matches[j].size() > 3)
				img[OBJECT]->good_neighbors.push_back(img[SCENE]->neighbors[j-1]);
		// Convert the key points into a vector containing the correspond X,Y position in image
		// and track the key points of scene frame and it's neighbors by correspond homography
		positionFromKeypoints();
		// if enough matches are found, stop loop
		if (object_points.size() > 4 && scene_points.size() > 4)
			good_thresh = false;
		else
		{
			// else, update threshold (allow less strong matches) and clean used data
			thresh += 0.1;
			good_matches.clear();
			neighbors_kp.clear();
			img[OBJECT]->good_neighbors.clear();
		}
	}
	// find perspective transformation from object to scene
	Mat H = findHomography(Mat(object_points), Mat(scene_points), CV_RANSAC);
	// if possible, force bottom-right element to 1
	if (!H.empty())
		H.at<double>(2, 2) = 1;
	// find best euclidean transformation from object to scene
	Mat R = estimateRigidTransform(Mat(object_points), Mat(scene_points), false);
	Mat E = Mat::eye(3, 3, CV_64F);
	// Since R is 3x2 matrix, we copy in a 3x3 forcing last row to (0, 0, 1)
	if (!R.empty())
	{
		R.copyTo(E(Rect(0, 0, 3, 2)));
		// remove scale factor from rotation matrix
		removeScale(E);
		// correct perspective transformation based on best euclidean
		correctHomography(H, E);
	}
	// find best euclidean transformation from the euclidean model (all frames tracked by euclidean transformation)
	R = estimateRigidTransform(Mat(object_points), Mat(euclidean_points), false);
	if (!R.empty())
	{
		R.copyTo(E(Rect(0, 0, 3, 2)));
		// remove scale factor from rotation matrix
		removeScale(E);
	}
	// R is empty, we return it
	else
		E = R;
	// save the resulting transformations
	vector<Mat> transform(2);
	transform[PERSPECTIVE] = H;
	transform[EUCLIDEAN] = E;
	// clean used data
	cleanData();
	return transform;
}

// See description in header file
void Stitcher::getGoodMatches(float _thresh)
{
	vector<DMatch> aux_matches;
	// loop over scene frame and it's neighbors
	for (int i = 0; i < img[SCENE]->neighbors.size() + 1; i++)
	{
		// loop over matches of object-scene and object-(scene-neighbors)
		// matches[0] correspond to object-scene matches
		for (vector<DMatch> match : matches[i])
		{
			if ((match[0].distance < _thresh * (match[1].distance)) &&
				((int)match.size() <= 2 && (int)match.size() > 0))
			{
				// take the first result only if its distance is smaller than threshold*second_best_dist
				// that means this descriptor is ignored if the second distance is bigger or of similar
				aux_matches.push_back(match[0]);
			}
		}
		// save the inliers in class variable
		good_matches.push_back(aux_matches);
		// clear to reuse
		aux_matches.clear();
	}

	img[SCENE]->good_points[NEXT].clear();
	img[OBJECT]->good_points[PREV].clear();
	// get the key points from the good matches
	// useful data to calculate local stitching (better than grid_points)
	for (DMatch good : good_matches[0])
	{
		// store the points in each frame object
		img[OBJECT]->good_points[PREV].push_back(img[OBJECT]->keypoints[good.queryIdx].pt);
		img[SCENE]->good_points[NEXT].push_back(img[SCENE]->keypoints[good.trainIdx].pt);
	}
}

// See description in header file
void Stitcher::gridDetector()
{
	// fix cells dimensions
	int stepx = img[OBJECT]->color.cols / cells_div;
	int stepy = img[OBJECT]->color.rows / cells_div;
	int best_distance = 100;
	int index = 0, m = 0;
	vector<vector<DMatch>> grid_matches = vector<vector<DMatch>>(good_matches.size());
	DMatch best_match;
	// loop over horizontal cells
	for (int i = 0; i < cells_div; i++)
	{
		// loop over vertical cells
		for (int j = 0; j < cells_div; j++)
		{
			best_distance = 100;
			index = 0;
			// loop over scene and it's neighbors matches
			for (int k = 0; k < img[SCENE]->neighbors.size() + 1; k++)
			{
				m = 0;
				// good_matches[0] correspond to object-scene good matches (inliers)
				for (DMatch match : good_matches[k])
				{
					// get the (X,Y) points from the good matches
					// if the key point is inside the cell
					if (img[OBJECT]->keypoints[match.queryIdx].pt.x >= stepx * i && img[OBJECT]->keypoints[match.queryIdx].pt.x < stepx * (i + 1) &&
						img[OBJECT]->keypoints[match.queryIdx].pt.y >= stepy * j && img[OBJECT]->keypoints[match.queryIdx].pt.y < stepy * (j + 1))
					{
						// look for strongest match in this cell
						if (match.distance < best_distance)
						{
							// save the match data
							best_distance = match.distance;
							best_match = match;
							// save the index of the frame (for neighbors)
							index = k;
						}
						// if the points is in the box we erase it, even if is not the best.
						// to reduce search space in next loop.
						good_matches[k].erase(good_matches[k].begin() + m);
					}
					m++;
				}
			}
			// check if for this cell we don't have a match.
			if (best_distance != 100)
				grid_matches[index].push_back(best_match);
		}
	}
	// good_matches[k+1] correspond to the matches of k neighbors of scene frame.
	good_matches = grid_matches;
}

// See description in header file
void Stitcher::positionFromKeypoints()
{
	img[OBJECT]->grid_points[PREV].clear();
	img[SCENE]->grid_points[NEXT].clear();
	// get the key points from the good object-scene matches (after grid detector)
	for (DMatch good : good_matches[0])
	{
		// get the key points from the good matches, after grid detector
		img[OBJECT]->grid_points[PREV].push_back(img[OBJECT]->keypoints[good.queryIdx].pt);
		img[SCENE]->grid_points[NEXT].push_back(img[SCENE]->keypoints[good.trainIdx].pt);
	}

	vector<Point2f> aux_points;
	// loop over scene neighbors
	for (int i = 0; i < img[SCENE]->neighbors.size(); i++)
	{
		// loop over matches
		for (DMatch good : good_matches[i + 1])
		{
			img[OBJECT]->grid_points[PREV].push_back(img[OBJECT]->keypoints[good.queryIdx].pt);
			aux_points.push_back(img[SCENE]->neighbors[i]->keypoints[good.trainIdx].pt);
		}
		// save X,Y points of good matches
		neighbors_kp.push_back(aux_points);
		aux_points.clear();
	}
	// track key points by each transformation matrix
	trackKeypoints();
	// save data in class variable
	object_points = img[OBJECT]->grid_points[PREV];
	scene_points = img[SCENE]->grid_points[NEXT];
	// save data from neighbors
	for (int i = 0; i < img[SCENE]->neighbors.size(); i++)
		scene_points.insert(scene_points.end(), neighbors_kp[i].begin(), neighbors_kp[i].end());

}

// See description in header file
void Stitcher::trackKeypoints()
{
	vector<vector<Point2f> > all_points(0);
	euclidean_points.clear();
	// multiply points by scene transformation matrix
	if (img[SCENE]->grid_points[NEXT].size() > 0)
	{
		perspectiveTransform(img[SCENE]->grid_points[NEXT], euclidean_points, img[SCENE]->E);
		perspectiveTransform(img[SCENE]->grid_points[NEXT], img[SCENE]->grid_points[NEXT], img[SCENE]->H);
		all_points.push_back(euclidean_points);		 
	}
	// loop over neighbors
	for (int i = 0; i < img[SCENE]->neighbors.size(); i++)
	{
		euclidean_points.clear();
		// multiply points by each transformation matrix
		if (neighbors_kp[i].size() > 0)
		{
			perspectiveTransform(neighbors_kp[i], euclidean_points, img[SCENE]->neighbors[i]->E);
			perspectiveTransform(neighbors_kp[i], neighbors_kp[i], img[SCENE]->neighbors[i]->H);
			all_points.push_back(euclidean_points);
		}
	}
	euclidean_points.clear();
	// save points in class variable
	for (vector<Point2f> points : all_points)
		euclidean_points.insert(euclidean_points.end(), points.begin(), points.end());
}

// See description in header file
void Stitcher::correctHomography(Mat &_H, Mat _E)
{
	// corner points for object frame (default points)
	vector<Point2f> h_points = {
		img[OBJECT]->bound_points[PERSPECTIVE][0],
		img[OBJECT]->bound_points[PERSPECTIVE][1],  
		img[OBJECT]->bound_points[PERSPECTIVE][2],
		img[OBJECT]->bound_points[PERSPECTIVE][3]
	};
	vector<Point2f> e_points = h_points;
	// multiply points by perspective transformation
	perspectiveTransform(h_points, h_points, _H);
	// multiply points by euclidean transformation	
	perspectiveTransform(e_points, e_points, _E);
	// get the mid points between perspective and euclidean points
	vector<Point2f> mid_points = {
		// getMidPoint(getMidPoint(h_points[0], e_points[0]), h_points[0]),
		// getMidPoint(getMidPoint(h_points[1], e_points[1]), h_points[1]),
		// getMidPoint(getMidPoint(h_points[2], e_points[2]), h_points[2]),
		// getMidPoint(getMidPoint(h_points[3], e_points[3]), h_points[3])
		getMidPoint(h_points[0], e_points[0]),
		getMidPoint(h_points[1], e_points[1]),
		getMidPoint(h_points[2], e_points[2]),
		getMidPoint(h_points[3], e_points[3])
	};
	// get the perspective transformation between perspective points and calculated mid points
	Mat correct_H = getPerspectiveTransform(h_points, mid_points);
	// apply correction to original perspective transformation
	_H = correct_H * _H;
}

// See description in header file
void Stitcher::cleanData()
{
	// clean used data. this variables will be used again
	good_matches.clear();
	neighbors_kp.clear();
	matches.clear(); 
	scene_points.clear();
	object_points.clear();
	euclidean_points.clear();
}
}