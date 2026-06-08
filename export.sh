#!/bin/bash
script_dir="$(dirname "$0")"
input_exe="$1"
[ -z "$input_exe" ] && input_exe="fan.exe"
base="$(basename "$input_exe")"
base_noext="${base%.*}"
read -p "Output folder name [$base_noext]: " outname
[ -z "$outname" ] && outname="$base_noext"
outdir="$script_dir/$outname"
python3 "$script_dir/export.py" "$input_exe" "$outdir" --force
mv "$outdir/$base" "$outdir/$outname.exe"