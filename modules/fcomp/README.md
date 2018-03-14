# Feature extractors and matchers Comparison
Monocular video based Mosaic Generation System for mobile robots implemented in OpenCV 3.2. In this module a performance comparison module of four feature extractors (Sift, Surf, Orb and Kaze), and feature matchers (BruteForce, Flann) is implemented.

## Requirements
- *OpenCV 3.0+*
- *OpenCV extra modules 3.0+*

## How to compile?
Provided with this module, there is a CMakeLists.txt file, which you can use to directly compile the code as follows:
```bash
cd <mosaic-opencv_directory>
mkdir build
cd build
cmake ..
make
```
## How to run?
After compilation, run the program with the following syntax:
```bash
./fcomp [INPUT DATA] --DETECTOR -MATCHER
```
To see detailed usage information, enter the following command:
```bash
./fcomp -h
```
### Usage examples:
```bash
./fcomp -i image1.jpg -i image2.jpg --sift -f -o
```
Detect and compute the features between two input images with sift detector and FLANN matcher, aditionally shows the matching points.

```bash
./fcomp -v video.mp4 --surf -b
```
Detect and compute the features between all the pairs of images extracted from a video file. The selection method is fixed to one frame per second. SURF detector and BruteForce matcher are used.
