cmake_minimum_required(VERSION 3.12)

project( otsu_ompi )

find_package( OpenCV 4.1.2 REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package( OpenMP REQUIRED )

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lpthread")

add_executable( otsu_ompi otsu_ompi.cpp )

target_link_libraries( otsu_ompi ${OpenCV_LIBS} )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

find_package( MPI REQUIRED )

include_directories(SYSTEM ${MPI_INCLUDE_PATH})

target_link_libraries( otsu_ompi ${MPI_C_LIBRARIES})


