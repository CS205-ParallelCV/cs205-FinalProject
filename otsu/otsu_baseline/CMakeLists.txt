cmake_minimum_required(VERSION 3.12)

project( otsu )

find_package( OpenCV 4.1.2 REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lpthread")

add_executable( otsu otsu.cpp )

target_link_libraries( otsu ${OpenCV_LIBS} )