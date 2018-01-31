# Stitch Module
Module to calculate the homography matrix and stitch two or more images.

## Requirements
- *OpenCV 3.0+*
- *OpenCV extra modules 3.0+*

## How to compile?
Provided with this repo is a CMakeLists.txt file, which you can use to directly compile the code as follows:
```bash
cd <mosaic-directory>/modules/stitch
mkdir build
cd build
cmake ..
make
```
## How to run?
After compilation, run the program with the following sintax:
```bash
./stitch [IMPUT DATA] {OPTIONS}
```
Type the following command to see correct usage:
```bash
./stitch -h
```
### Usage examples:
```bash
./stitch -i image1.jpg -i image2.jpg -o --pre --grid
```
Match and stitch two imput images in one. Simple Color Balance (SCB-RGB) preprocessing algorithm applied, grid selection for best keypoints, and shows the output best matches.
