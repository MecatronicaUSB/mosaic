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

/// Dimensions to resize images
#define TARGET_WIDTH	640   
#define TARGET_HEIGHT	480 

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
    catch (args::Help){
        std::cout << parser;
        return 1;
    }
    catch (args::ParseError e){
        std::cerr << e.what() << std::endl;
        std::cerr << "Use -h, --help command to see usage" << std::endl;
        //std::cout << parser; // to relaunch parser options in console
        return 1;
    }
    catch (args::ValidationError e){
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

    m2d::SubMosaic sub_mosaic;
    sub_mosaic.stitcher = new m2d::Stitcher(
        op_grid,                                            // use grid
        op_pre,                                             // apply histsretch algorithm
        op_surf ? m2d::USE_SURF : op_kaze ? m2d::USE_KAZE : op_orb ? m2d::USE_ORB : m2d::USE_SIFT, // select feature extractor
        op_flann ? m2d::USE_FLANN : m2d::USE_BRUTE_FORCE   // select feature matcher
    );

    // Two images as input
    if (op_img){
        for(const string img_name: args::get(op_img)){
            file_names.push_back(img_name);
        }
    }
    // Path as input
    string dir_ent;
    if(op_dir){
        dir_ent = args::get(op_dir);
        file_names = read_filenames(dir_ent);
    }

    t = (double) getTickCount();

    for(i=0; i < file_names.size(); i++){

        img = imread(file_names[i], IMREAD_COLOR);

        if(!img.data){
            cout<< " --(!) Error reading image "<< i << endl; 
            return -1;
        }

        if(!sub_mosaic.add2Mosaic(img)){
            cout<< "Couldn't Stitch images. Exiting..." << endl;
            return -1;
        }

        if(i > 0){
            n_matches = sub_mosaic.stitcher->matches.size();
            n_good = sub_mosaic.stitcher->good_matches.size();

            cout << "Pair  "<< n_img++ <<" -- -- -- -- -- -- -- -- -- --"  << endl;
            cout << "-- Possible matches  ["<< n_matches <<"]"  << endl;
            cout << "-- Good Matches      ["<<green<<n_good<<reset<<"]"  << endl;
        }
        if(op_out && i>0){
            t = 1000 * ((double) getTickCount() - t) / getTickFrequency();        
            cout << "   Execution time: " << t << " ms" <<endl;
            imshow("STITCH", sub_mosaic.final_scene);
            waitKey(0);
            t = (double) getTickCount();
        }
    }
    if (op_save)
    {
        imwrite("/home/victor/dataset/Results/"+args::get(op_save)+".png", sub_mosaic.final_scene);
    }

    t = 1000 * ((double) getTickCount() - t) / getTickFrequency();        
    cout << "   Execution time: " << t << " ms" <<endl;

    return 0;
}