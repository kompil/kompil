#!/bin/bash

ffmpeg -i $1 -vcodec mjpeg -qscale 1 -an $2