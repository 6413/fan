#!/bin/bash

if [ -z "$1" ]; then
  echo "usage: $0 input_file output_file"
  exit 1
fi

ffmpeg -i "$1" __temp.flac
opusenc --padding 0 --discard-comments --discard-pictures __temp.flac __temp.ogg
./ogg2sac.exe __temp.ogg
mv __temp.ogg "$2"
rm __temp.flac
