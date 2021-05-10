#!/usr/bin/env bash

trap 'exit 1' INT
cmake .
make

for i in $(seq 1 670); do
    IMAGE=cell_imgs/"$i".png
    echo -ne $IMAGE
    ./boruvka $IMAGE 
done