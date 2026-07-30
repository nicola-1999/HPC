#!/usr/bin/env bash

if [[ $# -ne 3 ]];
then 
    echo "Usage: $0 <startiter> <enditer> <stepiter>"
    exit 0
fi

echo "[mkvideo.sh]: running mkframe.gpl $1 $2 $3 ..."

cd video-mpi
gnuplot -c ../mkframe.gpl $1 $2 $3

convert -delay 50 -loop 0 frame-*.png video.gif

exit 0