#!/bin/bash

remove_dir_if_exists() {
	DIR_PATH=$1
	if [ -d "$DIR_PATH" ]; then
		rm -rf "$DIR_PATH"
	fi
}

remove_dir_if_exists "/usr/local/include/WITCH"
remove_dir_if_exists "/usr/local/include/BCOL"
remove_dir_if_exists "/usr/local/include/BLL"
remove_dir_if_exists "/usr/local/include/BVEC"
remove_dir_if_exists "/usr/local/include/BDBT"
remove_dir_if_exists "/usr/local/include/bcontainer"