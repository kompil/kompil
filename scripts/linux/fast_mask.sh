#!/bin/bash


MASK_PATH='res/masks'

mkdir $MASK_PATH

python3 kompil.py maskmaker build facetrack $1 -o $MASK_PATH/face.pth --resolution=320p --params display False

python3 kompil.py maskmaker build sobelyst $1 -o $MASK_PATH/sobel.pth --resolution=320p --params blur True display False

python3 kompil.py maskmaker build saliency $1 -o $MASK_PATH/saliency.pth --resolution=320p --params blur True display False

python3 kompil.py maskmaker merge $MASK_PATH/sobel.pth $MASK_PATH/face.pth $MASK_PATH/saliency.pth -m max -o $MASK_PATH/mask.pth
