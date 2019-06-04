# Stitch Module
Module that stitch two or more images, using the homography matrix.

## Requirements
- *OpenCV 3.0+*
- *OpenCV extra modules 3.0+*

## How to compile?
Within this repository there is a CMakeLists.txt file, which can be used to directly compile the code as follows:
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
Match and stitch two imput images in one. Simple Color Balance (SCB-RGB) preprocessing algorithm is applied, then a grid-based selection for a better keypoints extraction, and finally shows the output of best matches.
