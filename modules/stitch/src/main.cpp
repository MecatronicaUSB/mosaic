/**
 * @file main.h
 * @brief Contais the main code for feature extraction and match comparison
 * @version 1.0
 * @date 20/01/2018
 * @author Victor Garcia
 * @title Main code
 */

#include "../include/options.h"
#include "../include/detector.h"
#include "../include/stitch.h"
#include "../include/preprocessing.h"

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
 * @detail Compute the feature extraction and match using the desired algorithm.
 * Can perform the algorithm with:
 * - Two images
 * - Video
 * - A set of frames
 * Using the following extractors: 
 * - Sift
 * - Surf
 * - Orb
 * - Kaze
 * And the folowing matchers: 
 * - Brute force
 * - Flann
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
    Mat descriptors[2];
    Mat img[2], img_ori[2];

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
    if(op_flann){
        matcher = FlannBasedMatcher::create();
    }else{
        matcher = BFMatcher::create();
    }

    // Two images as imput
    if (op_img){
        n_iter = 1;
        t = (double) getTickCount();
        // Check for two image flags and patchs (-i imageName)
        for(const auto img_name: args::get(op_img)){
            img[i++] = imread(img_name, IMREAD_COLOR);
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
    if(0){
        //dir_ent = args::get(op_dir);
        //file_names = read_filenames(dir_ent);
        //n_iter = file_names.size()-1;
    }
    t = (double) getTickCount();
    for(i=0; i<n_iter; i++){
        if(0){
            img[0] = imread(dir_ent+"/"+file_names[i++],IMREAD_COLOR);
            img[1] = imread(dir_ent+"/"+file_names[i],IMREAD_COLOR);
        }
        // Resize the images to 640 x 480
        resize(img[0], img[0], Size(TARGET_WIDTH, TARGET_HEIGHT), 0, 0, CV_INTER_LINEAR);
        resize(img[1], img[1], Size(TARGET_WIDTH, TARGET_HEIGHT), 0, 0, CV_INTER_LINEAR);
        img_ori[0] = img[0].clone();
        img_ori[1] = img[1].clone();
        // Apply pre-processing algorithm if selected
        if(op_pre){
            colorChannelStretch(img[0], img[0], 1, 99);
            colorChannelStretch(img[1], img[1], 1, 99);
        }
        // Conver images to gray
        cvtColor(img[0],img[0],COLOR_BGR2GRAY);
        cvtColor(img[1],img[1],COLOR_BGR2GRAY);

        // Detect the keypoints using desired Detector and compute the descriptors
        keypoints[0].clear();
        keypoints[1].clear();
        detector->detectAndCompute( img[0], Mat(), keypoints[0], descriptors[0] );
        detector->detectAndCompute( img[1], Mat(), keypoints[1], descriptors[1] );

        if(!keypoints[0].size() || !keypoints[1].size()){
            cout << "No Key points Found" <<  endl;
            return -1;
        }

        // Match the keypoints for input images
        matches.clear();
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
        // for output command ( -o )

        vector<Point2f> img0, img1;
        for (int i = 0; i < good_matches.size(); i ++) {
            //-- Get the keypoints from the good matches
            img0.push_back(keypoints[0][good_matches[i].queryIdx].pt);
            img1.push_back(keypoints[1][good_matches[i].trainIdx].pt);
        }

        Mat H = findHomography(Mat(img0), Mat(img1), CV_RANSAC);
        
        Mat result = stitch(img_ori[0], img_ori[1], H);
        imshow("STITCH",result);
        waitKey(0);

        if(op_out){
            Mat img_matches;
            // Draw only "good" matches
            drawMatches( img[0], keypoints[0], img[1], keypoints[1],
                        good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                        vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
            // Show matches
            namedWindow("Good Matches", WINDOW_NORMAL);
            imshow( "Good Matches", img_matches );
            waitKey(0);
        }
        img[0].release();
        img[1].release();
        descriptors[0].release();
        descriptors[1].release();
    }

    cout << "\nTotal "<< n_img <<" -- -- -- -- -- -- -- -- -- --"  << endl;
    cout << "-- Total Possible matches  ["<< tot_matches <<"]"  << endl;
    cout << "-- Total Good Matches      ["<<green<<tot_good<<reset<<"]"  << endl;
    t = 1000 * ((double) getTickCount() - t) / getTickFrequency();        
    cout << "   Execution time: " << t << " ms" <<endl;

    return 0;
}