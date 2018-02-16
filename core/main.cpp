// STITCH MODULE
/**
 * @file main.cpp
 * @brief Contais the main code for image stitching module
 * @version 0.2
 * @date 10/02/2018
 * @author Victor Garcia
 * @title Main code
 */

#include "include/options.h"
#include "include/stitch.hpp"
#include "include/mosaic.hpp"

const std::string green("\033[1;32m");
const std::string reset("\033[0m");

/// User namespaces
using namespace std;
using namespace cv::xfeatures2d;

/*
 * @function main
 * @brief Main function
 */
int main( int argc, char** argv ) {

    double t;
    int n_matches=0, n_good=0;
    int i=0, n_img=0;
    cv::Mat img;
    vector<string> file_names;

    parser.Prog(argv[0]);
  
    try{
        parser.ParseCLI(argc, argv);
    }
    catch (args::Help) {
        std::cout << parser;
        return 1;
    }
    catch (args::ParseError e) {
        std::cerr << e.what() << std::endl;
        std::cerr << "Use -h, --help command to see usage" << std::endl;
        //std::cout << parser; // to relaunch parser options in console
        return 1;
    }
    catch (args::ValidationError e) {
        std::cerr << "Bad input commands" << std::endl;
        std::cerr << "Use -h, --help command to see usage" << std::endl;
        //std::cout << parser; // to relaunch parser options in console
        return 1;
    }


    // Veobose section -----
    cout << "Built with OpenCV " << CV_VERSION << endl;
    op_img ?   // this flag is activated from argument parser
    cout << "\tTwo images as input\t" << endl:
    cout << "\tDirectory as input\t" << endl;
    op_surf ? // this flag is activated from argument parser
    cout << "\tFeature extractor:\t" << "SURF" << endl:
    cout << "\tFeature extractor:\t" << "KAZE\t(Default)" << endl;
    op_flann ? // this flag is activated from argument parser
    cout << "\tFeature Matcher:\t" << "FLANN" << endl:
    cout << "\tFeature Matcher:\t" << "BRUTE FORCE (Default)" << endl;
    cout << boolalpha;
    cout << "\tApply preprocessing:\t"<< op_pre << endl;
    cout << "\tUse grid detection:\t"<< op_grid << endl;

    m2d::Mosaic mosaic;
    mosaic.stitcher = new m2d::Stitcher(
        op_grid,                                            // use grid
        op_pre,                                             // apply histsretch algorithm
        op_surf ? m2d::USE_SURF : m2d::USE_KAZE,          // select feature extractor
        op_flann ? m2d::USE_FLANN : m2d::USE_BRUTE_FORCE    // select feature matcher
    );

    // Two images as input
    if (op_img) {
        for (const string img_name: args::get(op_img)) {
            file_names.push_back(img_name);
        }
    }

    // // // // Mat img11 = imread(file_names[0],1);
    // // // // Mat img22 = imread(file_names[1],1);
    // // // // Mat img1, img2;

    // // // // vector<Point2f> bound_points1, bound_points2;
	// // // // bound_points1.push_back(Point2f(0, 0));
	// // // // bound_points1.push_back(Point2f(640, 0));
	// // // // bound_points1.push_back(Point2f(640, 480));
	// // // // bound_points1.push_back(Point2f(0, 480));


    // // // // vector<Point2f> keypoints_pos[2];
    // // // // vector<KeyPoint> keypoints[2];
    // // // // vector<vector<cv::DMatch> > matches;
    // // // // vector<cv::DMatch> good_matches;
    // // // // Mat descriptors[2];

    // // // // cvtColor(img11, img1, CV_BGR2GRAY);
    // // // // cvtColor(img22, img2, CV_BGR2GRAY);

    // // // // imgChannelStretch(img1, img1, 1, 99);
    // // // // imgChannelStretch(img2, img2, 1, 99);

    // // // // Ptr<KAZE> detector = KAZE::create();
    // // // // detector->detectAndCompute( img1, Mat(), keypoints[0], descriptors[0] );
    // // // // detector->detectAndCompute( img2, Mat(), keypoints[1], descriptors[1] );

    // // // // Ptr<DescriptorMatcher> matcher = FlannBasedMatcher::create();
    // // // // matcher->knnMatch( descriptors[0], descriptors[1], matches, 2);

    // // // // for (vector<DMatch> match: matches) {
    // // // //     if ((match[0].distance < 0.5 * (match[1].distance)) &&
    // // // //         ((int) match.size() <= 2 && (int) match.size() > 0)) {
    // // // //         // take the first result only if its distance is smaller than 0.5*second_best_dist
    // // // //         // that means this descriptor is ignored if the second distance is bigger or of similar
    // // // //         good_matches.push_back(match[0]);
    // // // //     }
    // // // // }

    // // // // vector<float> warp_offset1(4);
    // // // // vector<float> warp_offset2(4);

    // // // // for (DMatch good: good_matches) {
    // // // //     //-- Get the keypoints from the good matches
    // // // //     keypoints_pos[0].push_back(keypoints[0][good.queryIdx].pt);
    // // // //     keypoints_pos[1].push_back(keypoints[1][good.trainIdx].pt);
    // // // // }
    // // // // vector<Mat> avgH(2);
    // // // // Mat avg_pts = Mat(keypoints_pos[0]) - Mat(keypoints_pos[1]);

    // // // // avgH[0] = findHomography(Mat(keypoints_pos[0]), Mat(keypoints_pos[0]) + avg_pts, CV_RANSAC);
    // // // // avgH[1] = findHomography(Mat(keypoints_pos[1]), Mat(keypoints_pos[1]) + avg_pts, CV_RANSAC);

    // // // // perspectiveTransform(bound_points1, bound_points2, avgH[1]);
    // // // // perspectiveTransform(bound_points1, bound_points1, avgH[0]);

    // // // // Rect2f aux_rect1 = boundingRect(bound_points1);
    // // // // Rect2f aux_rect2 = boundingRect(bound_points2);

    // // // // warp_offset1[0] = max(0.f,-aux_rect1.y);
    // // // // warp_offset1[1] = max(0.f, aux_rect1.y + aux_rect1.height - 480);
    // // // // warp_offset1[2] = max(0.f,-aux_rect1.x);
    // // // // warp_offset1[3] = max(0.f, aux_rect1.x + aux_rect1.width - 640);

    // // // // Mat offset_T1 = Mat::eye(3,3,CV_64F);
    // // // // offset_T1.at<double>(0,2)= -aux_rect1.x;
	// // // // offset_T1.at<double>(1,2)= -aux_rect1.y;
    // // // // avgH[0] = offset_T1*avgH[0];

    // // // // warp_offset2[1] = max(0.f, aux_rect2.y + aux_rect2.height - 480);
    // // // // warp_offset2[2] = max(0.f,-aux_rect2.x);
    // // // // warp_offset2[3] = max(0.f, aux_rect2.x + aux_rect2.width - 640);
    // // // // warp_offset2[0] = max(0.f,-aux_rect2.y);

    // // // // Mat offset_T2 = Mat::eye(3,3,CV_64F);
    // // // // offset_T2.at<double>(0,2)= -aux_rect2.x;
	// // // // offset_T2.at<double>(1,2)= -aux_rect2.y;
    // // // // avgH[1] = offset_T2*avgH[1];

    // // // // Mat warp_img1, warp_img2;
	// // // // warpPerspective(img11, warp_img1, avgH[0], Size(aux_rect1.width, aux_rect1.height));
	// // // // warpPerspective(img22, warp_img2, avgH[1], Size(aux_rect2.width, aux_rect2.height));
    // // // // cout << aux_rect1 << endl;
    // // // // copyMakeBorder(warp_img1, warp_img1, warp_offset2[0], warp_offset2[1],
    // // // //                                      warp_offset2[2], warp_offset2[3],
    // // // //                                      BORDER_CONSTANT,Scalar(0,0,0));
    // // // // // aux_rect1.x = 0;
    // // // // // aux_rect1.y = 0;
    
    // // // // cv::Mat object_position(warp_img1, aux_rect2);

    // // // // object_position -= warp_img2;
    // // // // object_position += warp_img2;
    
    // // // // imshow("test1", warp_img1);
    // // // // imshow("test2", warp_img2);
    // // // // waitKey(0);

    // Path as input
    string dir_ent;
    if (op_dir) {
        dir_ent = args::get(op_dir);
        file_names = read_filenames(dir_ent);
    }

    t = (double) getTickCount();

    for (string img_name: file_names) {

        img = imread(img_name, IMREAD_COLOR);
        if (!img.data) {
            cout<< " --(!) Error reading image "<< i << endl;
            return -1;
        }
        mosaic.addFrame(img);
        if(i > 0){
            n_matches = mosaic.stitcher->matches.size();
            n_good = mosaic.stitcher->good_matches.size();

            cout << "Pair  "<< n_img++ <<" -- -- -- -- -- -- -- -- -- --"  << endl;
            cout << "-- Possible matches  ["<< n_matches <<"]"  << endl;
            cout << "-- Good Matches      ["<<green<<n_good<<reset<<"]"  << endl;
            t = 1000 * ((double) getTickCount() - t) / getTickFrequency();        
            cout << "   Execution time: " << t << " ms" <<endl;
            t = (double) getTickCount();
        }
        if (op_out) {
            imshow("Sub-Mosaic "+to_string(mosaic.n_subs), mosaic.sub_mosaics[mosaic.n_subs]->final_scene);
            imwrite("/home/victor/dataset/output/sub-Mosaic "+to_string(mosaic.n_subs)+".jpg", mosaic.sub_mosaics[mosaic.n_subs]->final_scene);
            waitKey(0);
        }
    }

    return 0;
}