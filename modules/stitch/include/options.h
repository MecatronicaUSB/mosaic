/**
 * @file options.h
 * @brief Argument parser options
 * @version 1.0
 * @date 20/01/2018
 * @author Victor Garcia
 */

#pragma once

#include "args.hxx"

args::ArgumentParser parser("Stitch module", "Author: Victor Garcia");
args::HelpFlag help(parser, "help", "Display this help menu", {'h','?', "help"});

args::Group op_matcher(parser, "Select the Feature Matcher:", args::Group::Validators::AtMostOne);
args::Flag op_brutef(op_matcher, "Brute Force", "Brute force matcher (default)", {'b',"bf"});
args::Flag op_flann(op_matcher, "Flann", "Flann force matcher", {'f',"flann"});

args::Group op_feature(parser, "Select the Feature Extractor and Descriptor:", args::Group::Validators::AtMostOne);
args::Flag op_kaze(op_feature, "KAZE", "Use KAZE Extractor and Descriptor (default)", {"kaze"});
args::Flag op_surf(op_feature, "SURF", "Use SURF Extractor and Descriptor", {"surf"});
args::Flag op_sift(op_feature, "SIFT", "Use SIFT Extractor and Descriptor", {"sift"});
args::Flag op_orb(op_feature, "ORB", "Use ORB Extractor and Descriptor", {"orb"});

args::Group op_data(parser, "Select input data:", args::Group::Validators::AtLeastOne);
args::ValueFlagList<std::string> op_img(op_data, "image-name", "Image input name. Mus specify two file names (one per flag)",{'i'});
args::ValueFlag<std::string> op_dir(op_data,"path","Directory to load the frames.",{'d'});

args::Group optional(parser, "(Optional)", args::Group::Validators::DontCare);
args::Flag op_out(optional, "Output", "Show final blended images", {'o'});
args::Flag op_pre(optional, "Pre-processing", "Apply pre-processing algorithm to test improvement in keypoints search", {"pre"});
args::Flag op_grid(optional, "grid", "Filter keypoints based on grid distribution. the grid is fixed at 10x10 cels", {"grid"});
args::ValueFlag<std::string> op_save(op_data,"save","Save image",{'s'});