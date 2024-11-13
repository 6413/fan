#!/bin/bash

#used after cmake is setup

mkdir build

set -e

cd build
ninja
mv a.exe ..