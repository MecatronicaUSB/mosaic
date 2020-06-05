cmake_minimum_required(VERSION 2.8)
project( core-mosaic )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(OpenCV 4.2 REQUIRED )
message("OpenCV_LIBS: " ${OpenCV_LIBS} )

# if(DEFINED ENV{OPENCV_INSTALL_DIR})
#     message("OPENCV_INSTALL_DIR set by ENV variable, searching...")
#     find_package(OpenCV 4.2 REQUIRED PATHS $ENV{OPENCV_INSTALL_DIR} )
# else()
#     message("OPENCV_INSTALL_DIR not set, searching in default location(s)")
#     find_package(OpenCV 4.2 REQUIRED )
# endif(DEFINED ENV{OPENCV_INSTALL_DIR})

file(GLOB mosaic-files
    "include/*.h"
    "include/*.hxx"
    "include/*.hpp"
    "src/*.cpp"
    "main.cpp"
)

add_executable( mosaic ${mosaic-files} )	# bind all together
target_compile_options( mosaic PUBLIC -std=c++11)	# c++11 requried
target_link_libraries( mosaic ${OpenCV_LIBS})		# link libraries
