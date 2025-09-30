#!/bin/bash

nvcc -arch=sm_86 -maxrregcount=64 ./main.cu -o arch && \
  ./arch && \
  python plotter.py
