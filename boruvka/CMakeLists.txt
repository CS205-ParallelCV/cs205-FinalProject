cmake_minimum_required(VERSION 3.12)

project( boruvka )

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package( OpenCV 4.1.2 REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package( OpenMP REQUIRED )

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lpthread")

add_executable( boruvka boruvka_main.cpp Graph.hpp DisjointSet.hpp Boruvka.hpp Boruvka_parallel.hpp timing.h timing.c)

# add_executable( boruvka boruvka_main_parallel.cpp Graph.hpp DisjointSet.hpp Boruvka.hpp Boruvka_parallel.hpp timing.h timing.c)

target_link_libraries( boruvka ${OpenCV_LIBS} )

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")