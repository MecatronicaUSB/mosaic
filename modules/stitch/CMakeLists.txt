cmake_minimum_required(VERSION 2.8)
project( stitch-opencv )
find_package( OpenCV 3.0 REQUIRED xfeatures2d)

include_directories( ${OpenCV_INCLUDE_DIRS} )
file(GLOB stitch-cv
    "include/*.h"
    "include/*.hxx"
    "src/*.cpp"
)
add_executable( stitch ${stitch-cv} )
target_compile_options(stitch PUBLIC -std=c++11)
target_link_libraries( stitch ${OpenCV_LIBS} )

