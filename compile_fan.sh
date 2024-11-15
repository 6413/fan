#!/bin/bash

#does a fresh build, you can speedup by removing -B

set -e

make -B -f make_imgui "$@"
make -B -f make_nfd "$@"
make -B -f make_fmt "$@"
make -B -f make_fan "$@"