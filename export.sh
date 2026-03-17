#!/bin/bash

script_dir="$(dirname "$0")"

input_exe="$1"
[ -z "$input_exe" ] && input_exe="fan.exe"

default_name="$(basename "$input_exe" .exe)"

read -p "New exe name [$default_name]: " newname
[ -z "$newname" ] && newname="$default_name"

outdir="$script_dir/export_minimal_linux"

python3 "$script_dir/export.py" "$input_exe" "$outdir" --force

mv "$outdir/$(basename "$input_exe")" "$outdir/$newname.exe"