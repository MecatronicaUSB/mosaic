cmake_minimum_required(VERSION 3.18)  # future releases may require 3.20. Improved FindModules for GDAL, CGAL, TBB & CUDA toolkit
if(NOT DEFINED CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebug RelWithDebInfo MinSizeRel."
      FORCE)
endif()
#################################
set(CMAKE_COLOR_MAKEFILE ON)  # For fancy colouring scheme
if (${FORCE_COLORED_OUTPUT})
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        target_compile_options (project_options INTERFACE -fdiagnostics-color=always)
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        target_compile_options (project_options INTERFACE -fcolor-diagnostics)
    endif ()
endif ()

if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(ColourBold  "${Esc}[1m")
  set(Red         "${Esc}[31m")
  set(Green       "${Esc}[32m")
  set(Yellow      "${Esc}[33m")
  set(Blue        "${Esc}[34m")
  set(Magenta     "${Esc}[35m")
  set(Cyan        "${Esc}[36m")
  set(White       "${Esc}[37m")
  set(BoldRed     "${Esc}[1;31m")
  set(BoldGreen   "${Esc}[1;32m")
  set(BoldYellow  "${Esc}[1;33m")
  set(BoldBlue    "${Esc}[1;34m")
  set(BoldMagenta "${Esc}[1;35m")
  set(BoldCyan    "${Esc}[1;36m")
  set(BoldWhite   "${Esc}[1;37m")
endif()
#################################
project( core-mosaic )
set(CMAKE_CXX_STANDARD 11)  # let's push this to C++17 for STL multithreading
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
#################################
if (USE_CUDA)
  message ("${BoldBlue} Asked for CUDA, searching... ${ColourReset}")
  find_package(CUDA) # FindCUDA is deprecated, future releases will use find_package(CUDAToolkit)
  if(CUDA_FOUND)
    add_definitions( -DUSE_CUDA=1 )
    include(FindCUDA)
    message("${BoldGreen}" "CUDA library status: ${ColourReset}")
    message(STATUS "    toolkit root dir: ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "    version: ${CUDA_VERSION}")
    message(STATUS "    libraries: ${CUDA_LIBRARIES}")
    message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
  endif(CUDA_FOUND)
endif()

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
find_package(OpenCV 4   REQUIRED 
                        PATHS /usr/local /local/software/opencv/4.5.3
                        NO_DEFAULT_PATH)

# If the package has been found, several variables will be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
message("${BoldGreen}" "OpenCV library status:" "${ColourReset}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

file(GLOB mosaic-headers    # greedy search pattern for headers
    "include/*.h"
    "include/*.hxx"
    "include/*.hpp"
)

# Add context information about current git commit, useful for debugging purposes
exec_program(
    "git"
    ${CMAKE_CURRENT_SOURCE_DIR}
    ARGS "describe --abbrev=4 --dirty --always --tags"
    OUTPUT_VARIABLE GIT_INFO )
add_definitions( -DGIT_COMMIT="${GIT_INFO}" )

add_executable( mosaic ${mosaic-headers}        # being strict, we only add the specific modules required for this build target
                        src/mosaic.cpp
                        src/submosaic.cpp 
                        src/stitch.cpp 
                        src/utils.cpp
                        src/frame.cpp
                        src/blend.cpp
                        src/main.cpp)	# bind all together

target_compile_options( mosaic PUBLIC -std=c++11)	# c++11 requried
target_link_libraries( mosaic ${OpenCV_LIBS})		# link libraries
