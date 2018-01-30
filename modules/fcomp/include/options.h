/**
 * @file options.h
 * @brief Argument parser options
 * @version 1.0
 * @date 20/01/2018
 * @author Victor Garcia
 */

#ifndef OPTIONS
#define OPTIONS

#include "args.hxx"

args::ArgumentParser parser("Feature Detectors comparison", "Author: Victor Garcia");
args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});

args::Group sub_p(parser, "", args::Group::Validators::AllOrNone);

args::Group op_feature(sub_p, "Select the Feature Extractor and Descriptor:", args::Group::Validators::AtLeastOne);
args::Flag op_sift(op_feature, "SIFT", "Use SIFT Extractor and Descriptor", {"sift"});
args::Flag op_surf(op_feature, "SURF", "Use SURF Extractor and Descriptor", {"surf"});
args::Flag op_orb(op_feature, "ORB", "Use ORB Extractor and Descriptor", {"orb"});
args::Flag op_kaze(op_feature, "KAZE", "Use KAZE Extractor and Descriptor", {"kaze"});

args::Group op_matcher(sub_p, "Select the Feature Matcher:", args::Group::Validators::AtLeastOne);
args::Flag op_brutef(op_matcher, "Brute Force", "Brute force matcher (default)", {'b'});
args::Flag op_flann(op_matcher, "Flann", "Flann force matcher (default)", {'f'});

args::Group op_data(sub_p, "Select imput data:", args::Group::Validators::AtMostOne);
args::ValueFlagList<std::string> op_img(op_data, "ImageName", "Image imput name. Mus specify two file names (one per flag)",{'i'});
args::ValueFlag<std::string> op_dir(op_data,"","Directory to load the frames. (Not yet implemented)",{'d'});

args::Group optional(parser, "(Optional)", args::Group::Validators::DontCare);
args::Flag op_out(optional, "Output", "Only for images imput. Show output for good matches.", {'o'});
args::Flag op_pre(optional, "Pre-processing", "Apply pre-processing algorithm to test improvement in keypoints search", {"pre"});
args::Flag op_grid(optional, "grid", "Filter keypoints based on grid distribution. the grid is 10x10 cels", {"grid"});

#endif