#!/bin/bash

#does a fresh build, you can speedup by removing -B

set -e

make -f make_fan "$@"