cmake_minimum_required(VERSION 2.8)
project( core-mosaic )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(OpenCV 4.2 REQUIRED )
message("OpenCV_LIBS: " ${OpenCV_LIBS} )


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

add_executable( mosaic ${mosaic-files} )	# bind all together
target_compile_options( mosaic PUBLIC -std=c++11)	# c++11 requried
target_link_libraries( mosaic ${OpenCV_LIBS})		# link libraries
