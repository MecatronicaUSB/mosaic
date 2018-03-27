/**
 * @file options.h
 * @brief Argument parser options
 * @version 1.0
 * @date 20/01/2018
 * @author Victor Garcia
 */

#pragma once
#include "args.hxx"

// Use description and information
args::ArgumentParser parser("2D mosaic generation pipeline.", "Author: Victor Garcia.");
args::HelpFlag help(parser, "help", "Display this help menu.", {'h', '?', "help"});

// Select the feature matcher
args::Group matcher(parser, "Select the Feature Matcher:", args::Group::Validators::AtMostOne);
args::Flag matcher_brutef(matcher, "Brute Force", "Brute force matcher.", {'b', "bf"});
args::Flag matcher_flann(matcher, "Flann", "Flann force matcher (default).", {'f', "flann"});

// select the feature detector and descriptor
args::Group feature(parser, "Select the Feature Extractor and Descriptor:", args::Group::Validators::AtMostOne);
args::Flag detector_sift(feature, "SIFT", "Use SIFT -- Scale Invariant Feature Detector.", {"sift"});
args::Flag detector_surf(feature, "SURF", "Use SURF -- Speed-Up Robust Feature.", {"surf"});
args::Flag detector_akaze(feature, "AKAZE", "Use A-KAZE -- A-Kaze Features.", {"akaze"});
args::Flag detector_kaze(feature, "KAZE", "Use KAZE -- Kaze Features (default).", {"kaze"});

// input path of images and output filename
args::Group group_data(parser, "Select input data:", args::Group::Validators::AtLeastOne);
args::Positional<std::string> input_dir(group_data, "Input directory", "Directory to load the frames.");
args::Positional<std::string> output_dir(group_data, "Output directory", "Directory to save the final mosaics.");

// Optional commands
args::Group group_optional(parser, "(Optional)", args::Group::Validators::DontCare);
args::ValueFlag<int> blender_bands(group_optional, "", "number of bands for multi-band blender (avilable for graph-cut option only).", {'n'});
args::Flag use_grid(group_optional, "grid", "Filter key points based on grid distribution. the grid is fixed at 10x10 cels.", {"grid"});
args::Flag final_scb(group_optional, "SCB", "Apply Simple-Color-Balance algorithm to enhance final image.", {"scb"});
args::Flag euclidean_mode(group_optional, "", "Euclidean mode, aproximate projective transformation to best euclidean.", {'e'});
args::Flag graph_cut(group_optional, "", "Graph-cut seam findel --pixel level-- (slower).", {'g'});
args::Flag output(group_optional, "print", "Show final blended images.", {'o'});
