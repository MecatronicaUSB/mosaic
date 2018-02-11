// STITCH MODULE
/**
 * @file main.h
 * @brief Contais the main code for image stitching module
 * @version 1.0
 * @date 20/01/2018
 * @author Victor Garcia
 * @title Main code
 */

#include "../include/options.h"
#include "../include/stitch.hpp"

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

    double t;
    long tot_matches=0, tot_good =0;
    int n_matches=0, n_good=0;
    int i=0, n_img=0;
    int n_iter = 0, step_iter = 0;
    Mat img[2];
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
        return 1;
    }
    catch (args::ValidationError e){
        std::cerr << "Bad imput commands" << std::endl;
        std::cerr << "Use -h, --help command to see usage" << std::endl;
        return 1;
    }

    // Veobose section -----
    cout << "Built with OpenCV " << CV_VERSION << endl;
    cout << "\tTwo images as imput\t" << endl;
    cout << "\tFeature extractor:\t" << "KAZE" << endl;
    cout << "\tFeature Matcher:\t" << "FLANN" << endl;
    cout << boolalpha;
    cout << "\tApply preprodessing:\t"<< op_pre << endl;
    cout << "\tUse grid detection:\t"<< op_grid << endl;


    // Create Stitcher class based on input options
    m2d::Stitcher mosaic(
        op_grid,                                            // use grid
        op_pre,                                             // apply histsretch algorithm
        TARGET_WIDTH,                                       // frame width
        TARGET_HEIGHT,                                      // frame heigt
        op_akaze ? m2d::USE_AKAZE : m2d::USE_KAZE,          // select feature extractor
        op_flann ? m2d::USE_FLANN : m2d::USE_BRUTE_FORCE    // select feature matcher
    );
    // m2d::Stitcher mosaic(
    //     1,                                            // use grid
    //     1,                                             // apply histsretch algorithm
    //     TARGET_WIDTH,                                       // frame width
    //     TARGET_HEIGHT,                                      // frame heigt
    //     0,          // select feature extractor
    //     1    // select feature matcher
    // );
    // Two images as imput
    if (op_img){
        n_iter = 1;
        t = (double) getTickCount();
        // Check for two image flags and patchs (-i imageName)
        for(const string img_name: args::get(op_img)){
            img[i++] = imread(img_name, IMREAD_COLOR);
            if( !img[i-1].data){
                cout<< " --(!) Error reading image "<< i << endl; 
                cerr << parser;
                return -1;
            }
            //cout<< " -- Loaded image "<< img_name << endl;
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
        n_iter = file_names.size();
        img[m2d::ReferenceImg::SCENE] = imread(dir_ent+"/"+file_names[0],IMREAD_COLOR);
    } 

    t = (double) getTickCount();
    mosaic.setScene(img[m2d::ReferenceImg::SCENE]);

    for(i=0; i<n_iter; i++){
        if(op_dir && i>0){
            img[m2d::ReferenceImg::OBJECT] = imread(dir_ent+"/"+file_names[i],IMREAD_COLOR);
        }

        mosaic.stitch(img[m2d::ReferenceImg::OBJECT]);
        
        n_matches = mosaic.matches.size();
        n_good = mosaic.good_matches.size();
        tot_matches+=n_matches;
        tot_good+=n_good;

        cout << "Pair  "<< n_img++ <<" -- -- -- -- -- -- -- -- -- --"  << endl;
        cout << "-- Possible matches  ["<< n_matches <<"]"  << endl;
        cout << "-- Good Matches      ["<<green<<n_good<<reset<<"]"  << endl;

        if(op_out && !op_img){
            imshow("STITCH",mosaic.scene_ori);
            t = 1000 * ((double) getTickCount() - t) / getTickFrequency();        
            cout << "   Execution time: " << t << " ms" <<endl;
            waitKey(0);
            t = (double) getTickCount();
        }

    }
    imshow("STITCH",mosaic.scene_ori);
    t = 1000 * ((double) getTickCount() - t) / getTickFrequency();        
    cout << "   Execution time: " << t << " ms" <<endl;
    waitKey(0);
    return 0;
}