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
#include "include/utils.h"
#include "include/stitch.hpp"
#include "include/blend.hpp"
#include "include/mosaic.hpp"

const std::string green("\033[1;32m");
const std::string reset("\033[0m");

/// User namespaces
using namespace std;

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

    parser.helpParams.proglineOptions = "[detector] [matcher] {OPTIONAL}";

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

        }
    }

    return 0;
}