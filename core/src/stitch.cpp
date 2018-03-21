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
Stitcher::Stitcher(bool _grid, int _detector, int _matcher, int _mode)
{
	use_grid = _grid;
	cells_div = 10;

	stitch_mode = _mode;
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
		detector = SURF::create();
		break;
	}
	default:
	{
		detector = KAZE::create();
		break;
	}
	}

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

void Stitcher::detectFeatures(vector<Frame *> _frames)
{
	for (int i = 0; i < _frames.size(); i++)
	{
		detector->detectAndCompute(_frames[i]->gray, Mat(),
									_frames[i]->keypoints,
									_frames[i]->descriptors);
		if (!_frames[i]->haveKeypoints())
		{
			delete _frames[i];
			_frames.erase(_frames.begin() + i);
		}
		_frames[i]->gray.release();
	}
}

// See description in header file
void Stitcher::stitch(Frame *_object, Frame *_scene)
{
	img[OBJECT] = _object;
	img[SCENE] = _scene;

	// Match the keypoints using Knn
	vector<vector<DMatch>> aux_matches;

	matcher->knnMatch(img[OBJECT]->descriptors, img[SCENE]->descriptors, aux_matches, 2);
	matches.push_back(aux_matches);

	for (Frame *neighbor : img[SCENE]->neighbors)
	{
		aux_matches.clear();
		matcher->knnMatch(img[OBJECT]->descriptors, neighbor->descriptors, aux_matches, 2);
		matches.push_back(aux_matches);
	}

	float thresh = 0.8;
	bool good_thresh = true;
	while (good_thresh)
	{

		// Discard outliers based on distance between descriptor's vectors
		good_matches.clear();
		getGoodMatches(thresh);

		// Apply grid detector if flag is activated
		if (use_grid)
		{
			gridDetector();
		}
		// Convert the keypoints into a vector containing the correspond X,Y position in image
		positionFromKeypoints();

		if (object_points.rows > 4 && scene_points[PERSPECTIVE].rows > 4)
		{
			good_thresh = false;
		}
		else
		{
			thresh += 0.1;
			for (int i = 0; i < img[SCENE]->neighbors.size(); i++)
			{
				good_matches.pop_back();
			}
			neighbors_kp.clear();
			img[OBJECT]->grid_points[PREV].clear();
			img[SCENE]->grid_points[NEXT].clear();
		}
	}

	Mat R = estimateRigidTransform(object_points, scene_points[PERSPECTIVE], false);
	if (!R.empty())
	{
		R.copyTo(img[OBJECT]->E(Rect(0, 0, 3, 2)));
		removeScale(img[OBJECT]->E);
	}

	Mat H = findHomography(object_points, scene_points[PERSPECTIVE], CV_RANSAC);
	if (!H.empty())
		H.at<double>(2, 2) = 1;
	img[OBJECT]->H = H;

	cleanNeighborsData();
}

void Stitcher::cleanNeighborsData()
{
	for (int i = 0; i < img[SCENE]->neighbors.size(); i++)
	{
		good_matches.pop_back();
	}
	neighbors_kp.clear();
	matches.clear();
}

void Stitcher::updateNeighbors()
{
	img[OBJECT]->neighbors.push_back(img[SCENE]);
	for (Frame *neighbor: img[SCENE]->neighbors)
	{
		if (img[OBJECT]->checkCollision(neighbor))
		{
			img[OBJECT]->neighbors.push_back(neighbor);
		}
	}
	cleanNeighborsData();
}

// See description in header file
void Stitcher::getGoodMatches(float _thresh)
{
	vector<DMatch> aux_matches;
	for (int i = 0; i < img[SCENE]->neighbors.size() + 1; i++)
	{
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
		good_matches.push_back(aux_matches);
		aux_matches.clear();
	}

	for (DMatch good : good_matches[0])
	{
		//-- Get the keypoints from the good matches
		img[OBJECT]->good_points[PREV].push_back(img[OBJECT]->keypoints[good.queryIdx].pt);
		img[SCENE]->good_points[NEXT].push_back(img[SCENE]->keypoints[good.trainIdx].pt);
	}
	
}

// See description in header file
void Stitcher::gridDetector()
{
	vector<vector<DMatch>> grid_matches;
	DMatch best_match;
	int stepx = img[OBJECT]->color.cols / cells_div;
	int stepy = img[OBJECT]->color.rows / cells_div;
	int best_distance = 100;
	int index = 0, m = 0;

	grid_matches = vector<vector<DMatch>>(good_matches.size());

	for (int i = 0; i < cells_div; i++)
	{
		for (int j = 0; j < cells_div; j++)
		{
			best_distance = 100;
			index = 0;
			for (int k = 0; k < img[SCENE]->neighbors.size() + 1; k++)
			{
				m = 0;
				for (DMatch match : good_matches[k])
				{
					//-- Get the keypoints from the good matches
					if (img[OBJECT]->keypoints[match.queryIdx].pt.x >= stepx * i && img[OBJECT]->keypoints[match.queryIdx].pt.x < stepx * (i + 1) &&
						img[OBJECT]->keypoints[match.queryIdx].pt.y >= stepy * j && img[OBJECT]->keypoints[match.queryIdx].pt.y < stepy * (j + 1))
					{
						if (match.distance < best_distance)
						{
							best_distance = match.distance;
							best_match = match;
							index = k;
						}
						good_matches[k].erase(good_matches[k].begin() + m);
					}
					m++;
				}
			}
			if (best_distance != 100)
			{
				grid_matches[index].push_back(best_match);
			}
		}
	}
	good_matches = grid_matches;
}

// See description in header file
void Stitcher::positionFromKeypoints()
{

	for (DMatch good : good_matches[0])
	{
		//-- Get the keypoints from the good matches
		img[OBJECT]->grid_points[PREV].push_back(img[OBJECT]->keypoints[good.queryIdx].pt);
		img[SCENE]->grid_points[NEXT].push_back(img[SCENE]->keypoints[good.trainIdx].pt);
	}

	vector<Point2f> aux_points;

	for (int i = 0; i < img[SCENE]->neighbors.size(); i++)
	{

		for (DMatch good : good_matches[i + 1])
		{
			img[OBJECT]->grid_points[PREV].push_back(img[OBJECT]->keypoints[good.queryIdx].pt);
			aux_points.push_back(img[SCENE]->neighbors[i]->keypoints[good.trainIdx].pt);
		}
		neighbors_kp.push_back(aux_points);
		aux_points.clear();
	}

	vector<Point2f> euclidean_points = trackKeypoints();

	object_points = Mat(img[OBJECT]->grid_points[PREV]);

	for (int i = 0; i < img[SCENE]->neighbors.size(); i++)
	{
		img[SCENE]->grid_points[NEXT].insert(img[SCENE]->grid_points[NEXT].end(),
											   neighbors_kp[i].begin(), neighbors_kp[i].end());
	}
	scene_points[PERSPECTIVE] = Mat(img[SCENE]->grid_points[NEXT]);
	scene_points[EUCLIDEAN] = Mat(euclidean_points);
}

// See description in header file
vector<Point2f> Stitcher::trackKeypoints()
{
	vector<vector<Point2f> > euclidean_points;
	vector<Point2f> aux_points;
	if (img[SCENE]->grid_points[NEXT].size() > 0)
	{
		perspectiveTransform(img[SCENE]->grid_points[NEXT], aux_points, img[SCENE]->E);
		perspectiveTransform(img[SCENE]->grid_points[NEXT], img[SCENE]->grid_points[NEXT], img[SCENE]->H);
		euclidean_points.push_back(aux_points);		 
	}

	for (int i = 0; i < img[SCENE]->neighbors.size(); i++)
	{
		if (neighbors_kp[i].size() > 0)
		{
			perspectiveTransform(neighbors_kp[i], aux_points, img[SCENE]->neighbors[i]->E);
			perspectiveTransform(neighbors_kp[i], neighbors_kp[i], img[SCENE]->neighbors[i]->H);
			euclidean_points.push_back(aux_points);
		}
	}
	aux_points.clear();
	for (int i = 0; i < euclidean_points.size(); i++)
	{
		aux_points.insert(aux_points.end(), euclidean_points[i].begin(), euclidean_points[i].end());
	}

	return aux_points;
}

// See description in header file
void Stitcher::drawKeipoints(vector<float> _warp_offset, Mat &_final_scene)
{
	vector<Point2f> aux_kp[2];

	aux_kp[0] = img[SCENE]->grid_points[NEXT];
	for (Point2f &pt : aux_kp[0])
	{
		pt.x += _warp_offset[LEFT];
		pt.y += _warp_offset[TOP];
	}

	perspectiveTransform(img[OBJECT]->grid_points[PREV], aux_kp[1], img[OBJECT]->H);
	for (int j = 0; j < aux_kp[0].size(); j++)
	{
		circle(_final_scene, aux_kp[0][j], 3, Scalar(255, 0, 0), -1);
		circle(_final_scene, aux_kp[1][j], 3, Scalar(0, 0, 255), -1);
	}

	cout << min(Mat(Point2f(5, 5)), abs(Mat(aux_kp[0]) - Mat(aux_kp[1]))) << endl;
}
}