#!/bin/bash

script_dir="$(dirname "$0")"

input_exe="$1"
[ -z "$input_exe" ] && input_exe="fan.exe"

base="$(basename "$input_exe" .exe)"

read -p "New exe name [$base]: " newname
[ -z "$newname" ] && newname="$base"

outdir="$script_dir/$newname"
export_args=("${@:2}")

python3 "$script_dir/export.py" "$input_exe" "$outdir" --force "${export_args[@]}"

mv "$outdir/$(basename "$input_exe")" "$outdir/$newname.exe"
