cmake_minimum_required(VERSION 2.8)
project( core-mosaic )


if(DEFINED ENV{OPENCV_INSTALL_DIR})
    message("OPENCV_INSTALL_DIR set by ENV variable, searching...")
    find_package(OpenCV 4.2 REQUIRED PATHS $ENV{OPENCV_INSTALL_DIR})
else()
    message("OPENCV_INSTALL_DIR not set, searching in default location(s)")
    find_package(OpenCV 3.2 REQUIRED )
endif(DEFINED ENV{OPENCV_INSTALL_DIR})

file(GLOB mosaic-files
    "include/*.h"
    "include/*.hxx"
    "include/*.hpp"
    "src/*.cpp"
    "main.cpp"
)

exec_program(
    "git"
    ${CMAKE_CURRENT_SOURCE_DIR}
    ARGS "describe --abbrev=4 --dirty --always --tags"
    OUTPUT_VARIABLE GIT_INFO )

add_definitions( -DGIT_COMMIT="${GIT_INFO}" )

add_executable( mosaic ${mosaic-files} )
target_compile_options( mosaic PUBLIC -std=c++11)
target_link_libraries( mosaic ${OpenCV_LIBS})

