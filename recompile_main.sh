#!/bin/bash

#used after cmake is setup

mkdir -p build

set -e

cd build
ninja
mv a.exe ..