#pragma once

#include _FAN_PATH(types/vector.h)

#include _FAN_PATH(graphics/webp.h)

#include <fstream>
#include <string>
#include <filesystem>

namespace fan {
  namespace io {
		static bool directory_exists(const fan::string& directory) {
			return std::filesystem::exists(directory.c_str());
		}

    static void create_directory(const std::string& folders) {
      std::filesystem::create_directories(folders);
    }

		struct iterate_sort_t {
			fan::string path;
			uint64_t area;

			static bool comp_cb(const iterate_sort_t& a,const iterate_sort_t& b) { return a.area > b.area; }
		};

    static void handle_string_out(fan::string& str) {
      return std::replace(str.begin(), str.end(), '\\', '/');;
    }
    static void handle_string_in(fan::string& str) {
      std::replace(str.begin(), str.end(), '/', '\\');;
    }

		static void iterate_directory_by_image_size_(
			const std::string& path,
			std::vector<iterate_sort_t>* sorted,
			const std::function<void(const fan::string& path)>& function
		) {

			for (const auto& entry : std::filesystem::directory_iterator(path)) {
				if (entry.is_directory()) {
					iterate_directory_by_image_size_(entry.path().string(), sorted, function);
					continue;
				}
				fan::string str = entry.path().string().data();
        handle_string_out(str);
				fan::vec2ui image_size;
				if (fan::webp::get_image_size(str, &image_size)) {
					fan::throw_error("failed to get image size:" + str);
				}
				iterate_sort_t sort;
				sort.path = str;
				sort.area = image_size.multiply();
				sorted->push_back(sort);
			}
		}

    static bool is_readable_path(const fan::string& path){
      try {
        std::filesystem::directory_iterator(path.c_str());
        std::filesystem::directory_entry(path.c_str());
        return true;
      }
      catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "error accessing directory: " << e.what() << std::endl;
      }
      catch (const std::exception& e) {
        std::cerr << "unexpected error: " << e.what() << std::endl;
      }
      return false;
    };

		static void iterate_directory_by_image_size(
			const std::string& path,
			const std::function<void(const fan::string& path)>& function
		) {
			std::vector<iterate_sort_t> sorted;
			iterate_directory_by_image_size_(path, &sorted, function);
			std::sort(sorted.begin(), sorted.end(), iterate_sort_t::comp_cb);
			for (const auto& i : sorted) {
				function(i.path);
			}
		}

    static fan::string exclude_path(const fan::string& full_path) {
      std::size_t found = full_path.find_last_of('/');
      if (found == fan::string::npos) {
        return full_path;
      }
      return full_path.substr(found + 1);
    }

    static void iterate_directory(
      const std::string& path,
      const std::function<void(const fan::string& path, bool is_directory)>& function
    ) {

      if (!directory_exists(path.c_str())) {
        fan::throw_error("directory does not exist");
      }

      try {
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
          fan::string str = entry.path().string();
          std::replace(str.begin(), str.end(), '\\', '/');
          function(str, entry.is_directory());
        }
      }
      catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "error accessing directory: " << e.what() << std::endl;
      }
      catch (const std::exception& e) {
        std::cerr << "unexpected error: " << e.what() << std::endl;
      }
    }

		static void iterate_directory_files(
			const std::string& path,
			const std::function<void(const fan::string& path)>& function
		) {

			if (!fan::io::directory_exists(path.c_str())) {
				fan::throw_error("directory does not exist");
			}

			for (const auto& entry : std::filesystem::directory_iterator(path)) {
				if (entry.is_directory()) {
          iterate_directory_files(entry.path().string(), function);
					continue;
				}
				fan::string str = entry.path().string().data();
				std::replace(str.begin(), str.end(), '\\', '/');
				function(str);
			}
		}
  }
}