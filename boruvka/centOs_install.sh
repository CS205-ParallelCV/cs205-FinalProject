#! /bin/bash

sudo yum install gcc
sudo yum install g++
wget https://github.com/Kitware/CMake/releases/download/v3.20.2/cmake-3.20.2.tar.gz
tar xzf cmake-3.20.2.tar.gz
cd cmake-3.20.2
sudo yum groupinstall 'Development Tools'
sudo yum install openssl-devel
./bootstrap
make
sudo make install
cd ..
sudo yum update
sudo yum install -y cmake g++ wget unzip
wget -O opencv.zip https://github.com/opencv/opencv/archive/master.zip
unzip opencv.zip
mkdir -p build && cd build
cmake  ../opencv-master
cmake --build .
cd .
export OpenCV_DIR=$PWD/build