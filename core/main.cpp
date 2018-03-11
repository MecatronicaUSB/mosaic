/**
 * @file main.cpp
 * @brief Contais the main code for automated 2d mosaic construction
 * @version 0.2
 * @date 10/02/2018
 * @author Victor Garcia
 * @title Main code
 */

#include "include/options.h"
#include "include/mosaic.hpp"

/// User namespaces
using namespace std;

const string green("\033[1;32m");
const string yellow("\033[1;33m");
const string cyan("\033[1;36m");
const string red("\033[1;31m");
const string reset("\033[0m");

/*
 * @function main
 * @brief Main function
 */
int main( int argc, char** argv ) {

    double t;
    cv::Mat img;
    string directory;
    vector<string> file_names;

    parser.helpParams.proglineOptions = "[DIRECTORY] [--matcher] [--detector] {OPTIONAL}";
    
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
        return 1;
    }
    catch (args::ValidationError e) {
        std::cerr << "Bad input commands" << std::endl;
        std::cerr << "Use -h, --help command to see usage" << std::endl;
        return 1;
    }

    directory = args::get(input_dir);
    file_names = read_filenames(directory);

    // Veobose section -----
    cout << endl << "2D mosaic generation"<<endl;
    cout << "Author: Victor Garcia"<<endl<<endl;
    cout << "Built with OpenCV\t" <<yellow<< CV_VERSION << reset << endl;
    cout << "  Directory:\t\t"<<cyan<< directory<<reset << endl;
    // Feature Extractor
    cout << "  Feature extractor:\t" << cyan;
    detector_surf ? cout << "SURF" :
    detector_sift ? cout << "SIFT" :
    detector_akaze ? cout << "A-KAZE" : cout << "KAZE\t(Default)";
    cout << reset << endl;
    // Feature Matcher
    cout << "  Feature Matcher:\t" << cyan;
    matcher_brutef ? cout << "BRUTE FORCE" : cout << "FLANN\t(Default)";
    cout << reset << endl;
    // # bands for multiband blender
    cout << "  Nº bands (blender):\t" << cyan;
    blender_bands ? cout << args::get(blender_bands) : cout << 5;
    cout << reset << endl;
    // Optional commands
    cout << boolalpha;
    cout << "  Apply preprocessing:\t"<<cyan<< apply_pre <<reset << endl;
    cout << "  Use grid detection:\t"<<cyan<< use_grid <<reset << endl<<endl;

    m2d::Mosaic mosaic(apply_pre);
    mosaic.stitcher = new m2d::Stitcher(
        use_grid,                                                   
        detector_surf  ? m2d::USE_SURF :
        detector_sift  ? m2d::USE_SIFT :
        detector_akaze ? m2d::USE_AKAZE:
        m2d::USE_KAZE,
        matcher_flann  ? m2d::USE_FLANN : m2d::USE_BRUTE_FORCE        // select feature matcher
    );
    mosaic.blender = new m2d::Blender(blender_bands ? args::get(blender_bands) : 5);

    t = (double) getTickCount();

    for (int i=0; i<file_names.size(); i++) {
        img = imread(file_names[i], IMREAD_COLOR);
        if (!img.data) {
            cout<< red <<" --(!) Error reading image "<< reset << endl;
            return -1;
        }

        if(!mosaic.addFrame(img))
            break;
        
        if (output) {
            mosaic.show();
        }
    }
    //mosaic.compute();

    t = 1000000 * ((double) getTickCount() - t) / getTickFrequency();        
    cout << endl << "\tExecution time:\t" << green << t << reset <<" s" <<endl;

    return 0;
}