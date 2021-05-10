cmake_minimum_required(VERSION 3.12)

project( otsu_openmp )

find_package( OpenCV 4.1.2 REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package( OpenMP REQUIRED )

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lpthread")

add_executable( otsu_openmp otsu_openmp.cpp )

target_link_libraries( otsu_openmp ${OpenCV_LIBS} )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")
