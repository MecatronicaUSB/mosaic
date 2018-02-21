/**
 * @file options.h
 * @brief Argument parser options
 * @version 1.0
 * @date 20/01/2018
 * @author Victor Garcia
 */

#pragma once
#include "args.hxx"


args::ArgumentParser parser("2D mosaic generation pipeline", "Author: Victor Garcia");
args::HelpFlag help(parser, "help", "Display this help menu", {'h','?', "help"});

args::Group matcher(parser, "Select the Feature Matcher:", args::Group::Validators::AtMostOne);
args::Flag matcher_brutef(matcher, "Brute Force", "Brute force matcher", {'b',"bf"});
args::Flag matcher_flann(matcher, "Flann", "Flann force matcher (default)", {'f',"flann"});

args::Group feature(parser, "Select the Feature Extractor and Descriptor:", args::Group::Validators::AtMostOne);
args::Flag detector_sift(feature, "SIFT", "Use SURF --Scale Invariant Feature Detector--", {"sift"});
args::Flag detector_surf(feature, "SURF", "Use SURF --Speed-Up Robust Feature--", {"surf"});
args::Flag detector_akaze(feature, "AKAZE", "Use SURF --A-Kaze Features--", {"akaze"});
args::Flag detector_kaze(feature, "KAZE", "Use KAZE --Kaze Features-- (default)", {"kaze"});

args::Group group_data(parser, "Select input data:", args::Group::Validators::AtLeastOne);
args::Positional<std::string> input_dir(group_data, "directory", "Directory to load the frames.");

args::Group group_optional(parser, "(Optional)", args::Group::Validators::DontCare);
args::Flag output(group_optional, "image-name", "Show final blended images", {'o'});
args::Flag apply_pre(group_optional, "Pre-processing", "Apply pre-processing algorithm to test improvement in keypoints search", {"pre"});
args::Flag use_grid(group_optional, "grid", "Filter keypoints based on grid distribution. the grid is fixed at 10x10 cels", {"grid"});
args::ValueFlag<std::string> save(group_optional, "image-name", "Save the blended sub-mosaic", {'s'});
