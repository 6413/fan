#!/bin/bash

THIRDPARTY_DIR="$(pwd)/thirdparty/fan"

if [ -d "$THIRDPARTY_DIR" ]; then
	echo "removing $THIRDPARTY_DIR"
	rm -r "$THIRDPARTY_DIR/"
else
	echo "$THIRDPARTY_DIR does not exist"
fi
