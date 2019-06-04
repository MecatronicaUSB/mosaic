cmake_minimum_required(VERSION 2.8)
project( core-mosaic )
find_package( OpenCV REQUIRED xfeatures2d stitching)

#include_directories( ${OpenCV_INCLUDE_DIRS} )
file(GLOB mosaic-files
    "include/*.h"
    "include/*.hxx"
    "include/*.hpp"
    "src/*.cpp"
    "main.cpp"
)
add_executable( mosaic ${mosaic-files} )
target_compile_options( mosaic PUBLIC -std=c++11)
target_link_libraries( mosaic ${OpenCV_LIBS})

