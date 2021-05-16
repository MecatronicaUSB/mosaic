/**
 * @file main.cpp
 * @brief Contais the main code for automated 2d mosaic construction
 * @version 0.2
 * @date 10/02/2018
 * @author Victor Garcia
 * @title Main code
 */

#include "../include/mosaic.hpp"
#include "../include/options.h"

/// User namespaces
// TODO: consistent use of std:: as qualifier of constants.
using namespace std;

/*
 * @function main
 * @brief Main function
 */
int main(int argc, char **argv) {

  double t;
  cv::Mat img;
  string input_directory, output_directory, calibration_file;
  vector<string> file_names;

  cout << cyan << "mosaic" << reset << endl;
  cout << "\tOpenCV version:\t" << yellow << CV_VERSION << reset << endl;
  cout << "\tGit commit:\t" << yellow << GIT_COMMIT << reset << endl;

  try {
    parser.ParseCLI(argc, argv);
  } catch (args::Help) {
    std::cout << parser;
    return 1;
  } catch (args::ParseError e) {
    std::cerr << e.what() << std::endl;
    std::cerr << "Use -h, --help command to see usage" << std::endl;
    return 1;
  } catch (args::ValidationError e) {
    //        std::cout << "mosaic:  Built with OpenCV\t" <<cyan<< CV_VERSION <<
    //        reset << endl;
    std::cerr << "Bad input commands" << std::endl;
    std::cerr << "Use -h, --help command to see usage" << std::endl;
    return 1;
  }

  input_directory = args::get(input_dir);
  output_directory = args::get(output_dir);
  calibration_file = args::get(calibration_dir);

  //-- VERBOSE SECTION --//
  cout << endl
       << "2D mosaic generation. Part of the underwater image processing "
          "toolbox."
       << endl;
  cout << "Author: Victor Garcia" << endl << endl;
  cout << "Maintainers: Jose Cappelletto" << endl << endl;
  cout << "  Built with OpenCV\t" << cyan << CV_VERSION << reset << endl;
  cout << "  Input directory:\t" << cyan << input_directory << reset << endl;
  cout << "  Output directory:\t" << cyan << output_directory << reset << endl;
  calibration_dir ? cout << "  Calibration file:\t" << cyan << calibration_file
                         << reset << endl
                  : cout << "  No calibration file given\t" << yellow
                         << "(Assuming undistorted images)" << reset << endl;
  //-- Feature Extractor
  cout << "  Feature extractor:\t";
  detector_surf
      ? cout << cyan << "SURF" << reset << endl
      : detector_kaze
            ? cout << cyan << "KAZE" << reset << endl
            : detector_akaze
                  ? cout << cyan << "A-KAZE" << reset << endl
                  : cout << cyan << "SIFT\t(Default)" << reset << endl;
  //-- Feature Matcher
  cout << "  Feature Matcher:\t";
  matcher_brutef ? cout << cyan << "BRUTE FORCE" << reset << endl
                 : cout << cyan << "FLANN\t(Default)" << reset << endl;
  //-- # bands for multiband blender
  cout << "  Nº bands (blender):\t";
  blender_bands ? cout << cyan << args::get(blender_bands) << reset << endl
                : cout << cyan << "5 (default)" << reset << endl;
  //-- Mosaic mode
  cout << "  Mosaic Mode:\t\t" << cyan;
  euclidean_mode ? cout << cyan << "Euclidean" : cout << cyan << "Perspective";
  cout << reset << endl;
  //-- Seam finder
  cout << "  Seam finder:\t\t" << cyan;
  graph_cut ? cout << cyan << "Graph cut" : cout << cyan << "Simple";
  cout << reset << endl;
  //-- Optional commands
  cout << boolalpha;
  cout << "  Use grid detection:\t" << cyan << use_grid << reset << endl;
  // cout << "  Eclidean correction:\t"<<cyan<< euclidean_correction <<reset <<
  // endl;
  cout << "  Apply SCB:\t\t" << cyan << final_scb << reset << endl << endl;

  // create mosaic object
  m2d::Mosaic mosaic(true);
  mosaic.stitcher = new m2d::Stitcher(
      use_grid, // grid detection
      detector_surf
          ? m2d::USE_SURF
          : detector_kaze ? m2d::USE_KAZE
                          : detector_akaze ? m2d::USE_AKAZE : m2d::USE_SIFT,
      matcher_flann ? m2d::USE_FLANN : m2d::USE_BRUTE_FORCE);
  mosaic.blender =
      new m2d::Blender(blender_bands ? args::get(blender_bands) : 5, color_c,
                       graph_cut, final_scb);

  FileStorage fs;
  cv::Mat camera_matrix;
  cv::Mat distortion_coeff;
  if (!calibration_dir) {
    cout << "Using default path for <calibration.xml>" << endl;
    calibration_file = "../calibration.xml";
  }

  fs.open(calibration_file, FileStorage::READ);
  if (!fs.isOpened()) {
    cout << "No calibration file. Assuming undistorted images" << endl;
  } else {
    cout << "Importing parameters from calibration file" << endl;
    fs["camera_matrix"] >> camera_matrix;
    fs["distortion_coefficients"] >> distortion_coeff;
  }
  // cout << "[main]" << " Setting camera matrix" << endl;
  mosaic.SetCameraMatrix(camera_matrix, distortion_coeff);

  t = (double)getTickCount();
  // There is an error when non-image files are contained in the source
  // directory

  // cout << "[main]" << " Reading input images" << endl;
  file_names = read_filenames(input_directory);
  if (file_names.size() == 0) {
    cout << red << "[main] No image file detected for:" << input_directory
         << reset << endl;
  }
  for (int i = 0; i < file_names.size(); i++) {
    // TODO: if a no-image file is contained (such as a directory), it will fail
    img = imread(file_names[i], IMREAD_COLOR);
    if (!img.data) {
      cout << red << " --(!) Error reading image " << reset << file_names[i]
           << endl;
      continue;
    }
    mosaic.feed(img);
  }
  // cout << green << "[main]" << reset << " mosaic.compute" << endl;
  mosaic.compute(euclidean_mode);
  // cout << green << "[main]" << reset << "\tmosaic.merge" << endl;
  mosaic.merge(true);
  // cout << green << "[main]" << reset << "\tmosaic.save" << endl;
  mosaic.save(output_directory);

  // retrieve elapsed time
  t = ((double)getTickCount() - t) / getTickFrequency();
  cout << endl
       << endl
       << "  Execution time:\t" << green << t << reset << " s" << endl;

  if (output) {
    mosaic.show();
  }

  return 0;
}
