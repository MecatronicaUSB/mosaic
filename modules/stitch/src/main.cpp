// STITCH MODULE
/**
 * @file main.h
 * @brief Contais the main code for image stitching module
 * @version 1.0
 * @date 20/01/2018
 * @author Victor Garcia
 * @title Main code
 */

#include "../../common/utils.h"
#include "../include/options.h"
#include "../include/stitch.h"

/// Dimensions to resize images
#define TARGET_WIDTH	640   
#define TARGET_HEIGHT	480 

const std::string green("\033[1;32m");
const std::string reset("\033[0m");

/// User namespaces
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

/*
 * @function main
 * @brief Main function
 */
int main( int argc, char** argv ) {

    double tot_matches=0, tot_good =0;
    double t;
    int n_matches=0, n_good=0;
    int i=0, n_img=0;
    int n_iter = 0, step_iter = 0;

    parser.Prog(argv[0]);

    vector<string> file_names;
    vector<vector<DMatch> > matches;
    vector<DMatch>  good_matches;
    vector<KeyPoint> keypoints[2];
    Mat result;
    Mat descriptors[2];
    Mat img[2], img_ori[2];
    struct WarpPoly bound;
    try{
        parser.ParseCLI(argc, argv);
    }
    catch (args::Help){
        std::cout << parser;
        return 1;
    }
    catch (args::ParseError e){
        std::cerr << e.what() << std::endl;
        std::cerr << "Use -h, --help command to see usage" << std::endl;
        //std::cerr << parser;
        return 1;
    }
    catch (args::ValidationError e){
        std::cerr << "Bad imput commands" << std::endl;
        std::cerr << "Use -h, --help command to see usage" << std::endl;
        return 1;
    }
   
    int minHessian = 400;
    Ptr<KAZE> detector = KAZE::create();
    Ptr<DescriptorMatcher> matcher;

    matcher = FlannBasedMatcher::create();

    // Two images as imput
    if (op_img){
        n_iter = 1;
        t = (double) getTickCount();
        // Check for two image flags and patchs (-i imageName)
        for(const auto img_name: args::get(op_img)){
            img[i++] = imread(img_name, IMREAD_UNCHANGED);
            if( !img[i-1].data){
                cout<< " --(!) Error reading image "<< i << endl; 
                cerr << parser;
                return -1;
            }
            cout<< " -- Loaded image "<< img_name << endl;
            // Two images loaded successfully
        }
        if(i<2){
            cout<< " -- Insuficient imput data " << endl;
            std::cerr << "Use -h, --help command to see usage" << std::endl;
            return -1;
        }
    }
    string dir_ent;
    if(op_dir){
        dir_ent = args::get(op_dir);
        file_names = read_filenames(dir_ent);
        n_iter = file_names.size()-3;
        img[0] = imread(dir_ent+"/"+file_names[0],IMREAD_COLOR);
        img[1] = imread(dir_ent+"/"+file_names[1],IMREAD_COLOR);
    }   
    Rect detectRoi(0, 0, TARGET_WIDTH, TARGET_HEIGHT);
    // Resize the images to 640 x 480
    resize(img[0], img[0], Size(TARGET_WIDTH, TARGET_HEIGHT), 0, 0, CV_INTER_LINEAR);
    resize(img[1], img[1], Size(TARGET_WIDTH, TARGET_HEIGHT), 0, 0, CV_INTER_LINEAR);

    img_ori[0] = img[0].clone();
    img_ori[1] = img[1].clone();
    t = (double) getTickCount();
    for(i=0; i<n_iter; i++){
        if(op_dir && i>0){
            img[0] = imread(dir_ent+"/"+file_names[i+1],IMREAD_COLOR);
            img[1] = img_ori[1].clone();
            resize(img[0], img[0], Size(TARGET_WIDTH, TARGET_HEIGHT), 0, 0, CV_INTER_LINEAR);
            img_ori[0] = img[0].clone();
        }
        // Conver images to gray
        cvtColor(img[0],img[0],COLOR_BGR2GRAY);
        cvtColor(img[1],img[1],COLOR_BGR2GRAY);

        // Apply pre-processing algorithm if selected (histogram stretch)
        if(op_pre){
            imgChannelStretch(img[0], img[0], 1, 99);
            imgChannelStretch(img[1], img[1], 1, 99);
        }

        // Detect the keypoints using desired Detector and compute the descriptors
        detector->detectAndCompute( img[0], Mat(), keypoints[0], descriptors[0] );
        detector->detectAndCompute( img[1](detectRoi), Mat(), keypoints[1], descriptors[1] );

        if(!keypoints[0].size() || !keypoints[1].size()){
            cout << "No Key points Found" <<  endl;
            return -1;
        }

        // Match the keypoints for input images
        matcher->knnMatch( descriptors[0], descriptors[1], matches, 2);
        n_matches = descriptors[0].rows;
        // Discard the bad matches (outliers)
        good_matches = getGoodMatches(n_matches - 1, matches);
        if(op_grid){
            good_matches = gridDetector(keypoints[0], good_matches);
        }

        n_good = good_matches.size();
        tot_matches+=n_matches;
        tot_good+=n_good;

        cout << "Pair  "<< n_img++ <<" -- -- -- -- -- -- -- -- -- --"  << endl;
        cout << "-- Possible matches  ["<< n_matches <<"]"  << endl;
        cout << "-- Good Matches      ["<<green<<n_good<<reset<<"]"  << endl;

        vector<Point2f> img0, img1;
        for (int i = 0; i < good_matches.size(); i ++) {
            //-- Get the keypoints from the good matches
            img0.push_back(keypoints[0][good_matches[i].queryIdx].pt);
            img1.push_back(keypoints[1][good_matches[i].trainIdx].pt);
        }

        Mat H = findHomography(Mat(img0), Mat(img1), CV_RANSAC);
        if(H.empty()){
            cout << "not enought keypoints to calculate homography matrix. Exiting..." <<  endl;
            break;
        }
        //saveHomographyData(H, keypoints, good_matches);
        bound = stitch(img_ori[0], img_ori[1], H);
        detectRoi = bound.rect;

        if(op_out && !op_img){
            imshow("STITCH",img_ori[1]);
            t = 1000 * ((double) getTickCount() - t) / getTickFrequency();        
            cout << "   Execution time: " << t << " ms" <<endl;
            waitKey(0);
            t = (double) getTickCount();
        }
        matches.clear();
        img[0].release();
        img[1].release();
        img_ori[0].release();
        keypoints[0].clear();
        keypoints[1].clear();
        descriptors[0].release();
        descriptors[1].release();
    }
    imshow("STITCH",img_ori[1]);
    t = 1000 * ((double) getTickCount() - t) / getTickFrequency();        
    cout << "   Execution time: " << t << " ms" <<endl;
    waitKey(0);
    img_ori[1].release();
    return 0;
}