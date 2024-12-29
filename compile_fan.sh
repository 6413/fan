#!/bin/bash

#does a fresh build, you can speedup by removing -B

set -e

make -B -f make_fan "$@"
