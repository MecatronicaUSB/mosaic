cmake_minimum_required(VERSION 3.0)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

project( core-mosaic )
find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

file(GLOB mosaic-files
    "include/*.h"
    "include/*.hxx"
    "include/*.hpp"
    "src/*.cpp"
    "main.cpp"
)

message(STATUS "OpenCV library status:")
message(STATUS " config: ${OPENCV_DIR}")
message(STATUS " version: ${OpenCV_VERSION}")
message(STATUS " libraries: ${OpenCV_LIBS}")
message(STATUS " include_path: ${OpenCV_INCLUDE_DIRS}")


add_executable( mosaic ${mosaic-files} )
target_compile_options( mosaic PUBLIC -std=c++11)
target_link_libraries( mosaic ${OpenCV_LIBS})

