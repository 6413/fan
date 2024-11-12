#!/bin/bash

set -e

make -B -f make_imgui "$@"
make -B -f make_nfd "$@"
make -B -f make_fmt "$@"