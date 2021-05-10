#!/usr/bin/env bash

trap 'exit 1' INT
cmake .
make

for i in $(seq 0 64); do
    IMAGE="test_imgs/test_$i.png"
    echo -ne $IMAGE
    ./boruvka $IMAGE 
done