#!/bin/bash

fan_include_path=/usr/local/include/fan
fan_lib_path=/usr/local/lib/fan

cerr () {
  if [ $? -ne 0 ]; then
    echo "command failed. exiting."
    exit 1
  fi
}

delete_link () {
  if [ $# -ne 1 ]; then
    echo "delete_link needs path"
    exit 1
  fi

  if sudo [ -e $1 ] || sudo [ -L $1 ]; then
    if sudo [ ! -L $1 ]; then
      echo "delete_link: $1 exists but its not a symbolic link"
      exit 1
    fi
    sudo rm $1
    cerr
  fi
}

delete_link $fan_include_path
delete_link $fan_lib_path

sudo ln -s $(realpath fan) $fan_include_path
cerr
sudo ln -s $(realpath lib/fan) $fan_lib_path
cerr
