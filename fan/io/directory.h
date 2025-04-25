#pragma once

#include _FAN_PATH(types/vector.h)

#include _FAN_PATH(graphics/webp.h)

#include <fstream>
#include <string>
#include <filesystem>

namespace fan {
  namespace io {
		static bool directory_exists(const std::string& directory) {
			return std::filesystem::exists(directory.c_str());
		}

    static void create_directory(const std::string& folders) {
      std::filesystem::create_directories(folders);
    }

		struct iterate_sort_t {
			std::string path;
			uint64_t area;

			static bool comp_cb(const iterate_sort_t& a,const iterate_sort_t& b) { return a.area > b.area; }
		};

    static void handle_string_out(std::string& str) {
      return std::replace(str.begin(), str.end(), '\\', '/');
    }
    static void handle_string_in(std::string& str) {
      std::replace(str.begin(), str.end(), '/', '\\');
    }

		static void iterate_directory_by_image_size_(
			const std::string& path,
			std::vector<iterate_sort_t>* sorted,
			const std::function<void(const std::string& path)>& function
		) {

			for (const auto& entry : std::filesystem::directory_iterator(path)) {
				if (entry.is_directory()) {
					iterate_directory_by_image_size_(entry.path().string(), sorted, function);
					continue;
				}
				std::string str = entry.path().string().data();
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

    static bool is_readable_path(const std::string& path){
      try {
        std::filesystem::directory_iterator(path.c_str());
        std::filesystem::directory_entry(path.c_str());
        return true;
      }
      catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "error accessing directory: " << e.what() << std::endl;
      }
      //catch (const std::exception& e) {
//        std::cerr << "unexpected error: " << e.what() << std::endl;
      //}
      return false;
    };

		static void iterate_directory_by_image_size(
			const std::string& path,
			const std::function<void(const std::string& path)>& function
		) {
			std::vector<iterate_sort_t> sorted;
			iterate_directory_by_image_size_(path, &sorted, function);
			std::sort(sorted.begin(), sorted.end(), iterate_sort_t::comp_cb);
			for (const auto& i : sorted) {
				function(i.path);
			}
		}

    static std::string exclude_path(const std::string& full_path) {
      std::size_t found = full_path.find_last_of('/');
      if (found == std::string::npos) {
        return full_path;
      }
      return full_path.substr(found + 1);
    }

    static void iterate_directory(
      const std::string& path,
      const std::function<void(const std::string& path, bool is_directory)>& function
    ) {

      if (!directory_exists(path.c_str())) {
        fan::throw_error("directory does not exist");
      }

      try {
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
          std::string str = entry.path().string();
          std::replace(str.begin(), str.end(), '\\', '/');
          function(str, entry.is_directory());
        }
      }
      catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "error accessing directory: " << e.what() << std::endl;
      }
      //catch (const std::exception& e) {
     //   std::cerr << "unexpected error: " << e.what() << std::endl;
    //  }
    }

    static void iterate_directory(
      const std::filesystem::path& path,
      const std::function<void(const std::filesystem::directory_entry& path)>& function
    ) {

      if (!directory_exists(path.string())) {
        fan::throw_error("directory does not exist");
      }

      try {
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
          function(entry);
        }
      }
      catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "error accessing directory: " << e.what() << std::endl;
      }
    //  catch (const std::exception& e) {
    //    std::cerr << "unexpected error: " << e.what() << std::endl;
   //   }
    }

   static void iterate_directory_sorted_by_name(
    const std::filesystem::path& path,
    const std::function<void(const std::filesystem::directory_entry&)>& function
    ) {
        if (!std::filesystem::exists(path)) {
            fan::throw_error("directory does not exist");
        }

        std::vector<std::filesystem::directory_entry> entries;

        try {
            // Collect all entries
            for (const auto& entry : std::filesystem::directory_iterator(path)) {
                entries.push_back(entry);
            }

            // Sort entries by type (directories first) and then by name
            std::sort(entries.begin(), entries.end(), 
                [](const std::filesystem::directory_entry& a, const std::filesystem::directory_entry& b) -> bool {
    // Check if both entries are of the same type (either both directories or both files)
    if (a.is_directory() == b.is_directory()) {
        // Convert stems to lowercase for case-insensitive comparison
        std::string a_stem = a.path().stem().string();
        std::string b_stem = b.path().stem().string();
        std::transform(a_stem.begin(), a_stem.end(), a_stem.begin(), ::tolower);
        std::transform(b_stem.begin(), b_stem.end(), b_stem.begin(), ::tolower);

        // If both are of the same type, sort alphabetically by filename without extension, case-insensitively
        return a_stem < b_stem;
    }
    // If they are not of the same type, directories come first
    return a.is_directory();
}
            );

            // Apply function to sorted entries
            for (const auto& entry : entries) {
                function(entry);
            }
        }
        catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "error accessing directory: " << e.what() << std::endl;
        }
     //   catch (const std::exception& e) {
    //        std::cerr << "unexpected error: " << e.what() << std::endl;
     //   }
    }

		static void iterate_directory_files(
			const std::string& path,
			const std::function<void(const std::string& path)>& function
		) {

			if (!fan::io::directory_exists(path.c_str())) {
				fan::throw_error("directory does not exist");
			}

			for (const auto& entry : std::filesystem::directory_iterator(path)) {
				if (entry.is_directory()) {
          iterate_directory_files(entry.path().string(), function);
					continue;
				}
				std::string str = entry.path().string().data();
				std::replace(str.begin(), str.end(), '\\', '/');
				function(str);
			}
		}
  }
}