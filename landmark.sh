#!/bin/bash

# docker run -v (pwd)/data:/home/openface-build/data -it algebr/openface
# /home/openface-build# ./build/bin/FaceLandmarkImg -fdir data/1BHOflzxPjI -out_dir data/1BHOflzxPjI/processed -3Dfp -pose
# /home/openface-build# ./build/bin/FaceLandmarkImg -f data/1BHOflzxPjI/frame_0468.png -out_dir data/1BHOflzxPjI/processed

for dir in ./noisy/pretrain/$1/*
do
    ./build/bin/FaceLandmarkImg -fdir $dir -out_dir $dir/processed -3Dfp -pose
done

for dir in ./noisy/trainval/$1/*
do
    ./build/bin/FaceLandmarkImg -fdir $dir -out_dir $dir/processed -3Dfp -pose
done
