# Automated mosaic generation pipeline
Monocular video based Mosaic Generation System for mobile robots implemented in OpenCV 3.2.

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
### Input data considerations
The input data must be a folder path containing at least two temporal consecutive frames, 
with at least 30% of overlap area between them.

### Run
After compilation, run the program with the following syntax:
```bash
./mosaic [INPUT DIRECTORY] [OUTPUT DIRECTORY] {OPTIONS}
```
To see detailed usage information, enter the following command:
```bash
./mosaic -h
```
### Usage examples:
```bash
./mosaic /home/data/frames2MOSAIC/ /home/data/outputPATH -o -g -n 3
```
The above code will build a mosaic using perspective transformations, 
to find the best seam line a graph cut algorithm wil be used, then a multiband blender with 3 bands will be applied.
Finally the resulting mosaic will be shown.

```bash
./mosaic /home/data/frames2MOSAIC/ /home/data/outputPATH -e --sift --bf --scb
```
The above code will build a mosaic using euclidean transformations, the SCB algorithm will be applied on each frame to 
enhance final mosaic view, SIFT detector and BruteForce matcher are used.

In both cases the resulting mosaic and track map will be saved in the provided outpt path.

# Results
### Test using:
- Graph cut seam finder.
- multiband blender with five bands.
- Simple color balance (in BGR).
- 28 images input.
- Perspective transformations.

<p align="center">
  <img src="https://github.com/MecatronicaUSB/mosaic/blob/master/core/results/MVI_0234-28imgs-graphcut-5bands-scb.png" width="350"/>
  <img src="https://github.com/MecatronicaUSB/mosaic/blob/master/core/results/MVI_0234-28imgs-graphcut-5bands-scb-MAP.jpg" width="350"/>
</p>
- Lef. Final Mosaic.<br />
- Right. Final Map.<br />

### Test using:
- Graph cut seam finder.
- 11 Images input.
- Perspective transformations.
- Correct global distortion by euclidean model.

<p align="center">
  <img src="https://github.com/MecatronicaUSB/mosaic/blob/master/core/results/ScottReef-SM0-before_correct.png" width="350"/>
  <img src="https://github.com/MecatronicaUSB/mosaic/blob/master/core/results/ScottReef-SM0-after-correct.png" width="350"/>
</p>
- Left. Mosaic before average homography correction.<br />
- Right. Mosaic after average homography correction.<br />
