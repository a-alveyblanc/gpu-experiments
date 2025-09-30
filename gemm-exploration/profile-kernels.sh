#!/bin/bash

NCU=$(which ncu)

sudo $NCU \
  -o profiled-kernels \
  -f \
  --set full \
  ./main --threadblock_2d
