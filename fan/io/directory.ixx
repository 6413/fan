module;

#include <fan/math/math.h>

import fan.graphics.webp;
import fan.types.vector;
import fan.types.print;

import std;

export module fan.io.directory;

export namespace fan {
  namespace io {
    fan_api bool directory_exists(const std::string& directory) {
      return std::filesystem::exists(directory.c_str());
    }

    fan_api void create_directory(const std::string& folders) {
      std::filesystem::create_directories(folders);
    }

    struct iterate_sort_t {
      std::string path;
      uint64_t area;

      static fan_api bool comp_cb(const iterate_sort_t& a, const iterate_sort_t& b) { return a.area > b.area; }
    };

    fan_api void handle_string_out(std::string& str) {
      return std::replace(str.begin(), str.end(), '\\', '/');
    }
    fan_api void handle_string_in(std::string& str) {
      std::replace(str.begin(), str.end(), '/', '\\');
    }

    fan_api void iterate_directory_by_image_size_(
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

    fan_api bool is_readable_path(const std::string& path) {
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

    fan_api void iterate_directory_by_image_size(
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

    fan_api std::string exclude_path(const std::string& full_path) {
      std::size_t found = full_path.find_last_of('/');
      if (found == std::string::npos) {
        return full_path;
      }
      return full_path.substr(found + 1);
    }

    fan_api void iterate_directory(
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

    fan_api void iterate_directory(
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

    fan_api void iterate_directory_sorted_by_name(
      const std::filesystem::path& path,
      const std::function<void(const std::filesystem::directory_entry&)>& function
    ) {
      if (!std::filesystem::exists(path)) {
        fan::throw_error("directory does not exist");
      }

      std::vector<std::filesystem::directory_entry> entries;

      try {
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
          entries.push_back(entry);
        }

        std::sort(entries.begin(), entries.end(),
          [](const std::filesystem::directory_entry& a, const std::filesystem::directory_entry& b) -> bool {
            if (a.is_directory() == b.is_directory()) {
              std::string a_stem = a.path().stem().string();
              std::string b_stem = b.path().stem().string();
              std::transform(a_stem.begin(), a_stem.end(), a_stem.begin(),
               [](unsigned char c) { return std::tolower(c); });
              std::transform(b_stem.begin(), b_stem.end(), b_stem.begin(),
                             [](unsigned char c) { return std::tolower(c); });

              return a_stem < b_stem;
            }
            return a.is_directory();
          }
        );

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

    fan_api void iterate_directory_files(
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