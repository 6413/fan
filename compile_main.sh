#!/bin/bash
mkdir build

set -e

cd build
cmake .. -G Ninja
ninja
mv a.exe ..