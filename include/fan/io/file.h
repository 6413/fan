#pragma once

#include _FAN_PATH(types/vector.h)

#include <fstream>
#include <string>
#undef index // xos.h
#include <filesystem>

namespace fan {
	namespace io {
    namespace file {

      inline bool exists(const fan::string& name) {
        std::ifstream file(name.c_str());
        return file.good();
      }

      inline bool rename(const fan::string& from, const fan::string& to) {
        return std::rename(from.c_str(), to.c_str());
      }

      struct fstream {
        fstream() = default;
        fstream(const fan::string& path) {
          file_name = path;
          open(path);
        }
        fstream(const fan::string& path, fan::string* str) {
          file_name = path;
          open(path);
          read(str);
        }

        bool open(const fan::string& path) {
          file_name = path;
          auto flags = std::ios::in | std::ios::out | std::ios::binary;

          if (!exists(path)) {
            flags |= std::ios::trunc;
          }

          file = std::fstream(path, flags);
          return !file.good();
        }

        bool read(const fan::string& path, fan::string* str) {
          file_name = path;
          open(path);
          return read(str);
        }

        bool read(fan::string* str) {
          if (file.is_open()) {
            file.seekg(0, std::ios::end);
            str->resize(file.tellg());
            file.seekg(0, std::ios::beg);
            file.read(&(*str)[0], str->size());
          }
          else {
            fan::print_warning("file is not opened:");
            return 1;
          }
          return 0;
        }

        bool write(fan::string* str) {
          auto flags = std::ios::in | std::ios::out | std::ios::binary | std::ios::trunc;

          file = std::fstream(file_name, flags);
          if (file.is_open()) {
            file.write(&(*str)[0], str->size());
            file.flush();
          }
          else {
            fan::print_warning("file is not opened:");
            return 1;
          }
          flags &= ~std::ios::trunc;
          file = std::fstream(file_name, flags);
          return 0;
        }

        std::fstream file;
        fan::string file_name;
      };

      using file_t = FILE;

      struct properties_t {
        const char* mode;
      };

      static bool close(file_t* f) {
        int ret = fclose(f);
        #if fan_debug >= fan_debug_low
        if (ret != 0) {
          fan::print_warning("failed to close file stream");
          return 1;
        }
        #endif
        return 0;
      }
      static bool open(file_t** f, const char* path, const properties_t& p) {
        *f = fopen(path, p.mode);
        if (f == nullptr) {
          fan::print_warning(fan::string("failed to open file:") + path);
          close(*f);
          return 1;
        }
        return 0;
      }

      static bool write(file_t* f, void* data, uint64_t size, uint64_t elements) {
        uint64_t ret = fwrite(data, size, elements, f);
        #if fan_debug >= fan_debug_low
        if (ret != elements && size != 0) {
          fan::print_warning("failed to write from file stream");
          return 1;
        }
        #endif
        return 0;
      }
      static bool read(file_t* f, void* data, uint64_t size, uint64_t elements) {
        uint64_t ret = fread(data, size, elements, f);
        #if fan_debug >= fan_debug_low
        if (ret != elements && size != 0) {
          fan::print_warning("failed to read from file stream");
          return 1;
        }
        #endif
        return 0;
      }

      inline uint64_t file_size(const fan::string& filename)
      {
        std::ifstream f(filename.c_str(), std::ifstream::ate | std::ifstream::binary);
        return f.tellg();
      }

      inline bool write(
        fan::string path,
        const fan::string& data,
        decltype(std::ios_base::binary | std::ios_base::app) mode = std::ios_base::binary | std::ios_base::app
      ) {
        std::ofstream ofile(path.c_str(), mode);
        if (ofile.fail()) {
          fan::print_warning("failed to write to:" + path);
          return 0;
        }
        ofile.write(data.c_str(), data.size());
        return 1;
      }

      template <typename T>
      static inline void write(
        fan::string path,
        const std::vector<T>& vector,
        decltype(std::ios_base::binary | std::ios_base::app) mode = std::ios_base::binary | std::ios_base::app
      ) {
        std::ofstream ofile(path.c_str(), mode);
        if (ofile.fail()) {
          fan::throw_error("failed to write to:" + path);
        }
        ofile.write(reinterpret_cast<const char*>(&vector[0]), vector.size() * sizeof(T));
      }

      static std::vector<fan::string> read_line(const fan::string& path) {

        std::ifstream file(path.c_str(), std::ifstream::binary);
        if (file.fail()) {
          fan::throw_error("path does not exist " + path);
        }
        std::vector<fan::string> data;
        for (std::string line; std::getline(file, line); ) {
          data.push_back(line.c_str());
        }
        return data;
      }

      static bool read(const fan::string& path, fan::string* str) {

        std::ifstream file(path.c_str(), std::ifstream::ate | std::ifstream::binary);
        if (file.fail()) {
          fan::print_warning_no_space("path does not exist:" + path);
          return 1;
        }
        str->resize(file.tellg());
        file.seekg(0, std::ios::beg);
        file.read(&(*str)[0], str->size());
        file.close();
        return 0;
      }

      template <typename T>
      static std::vector<T> read(const fan::string& path) {
        std::ifstream file(path.c_str(), std::ifstream::ate | std::ifstream::binary);
        if (!file.good()) {
          fan::print("path does not exist " + path);
          exit(1);
        }
        std::vector<T> vector;
        uint64_t size = file.tellg();
        vector.resize(size / sizeof(T));
        file.seekg(0, std::ios::beg);
        file.read(reinterpret_cast<char*>(&vector[0]), size);
        return vector;
      }

      struct str_int_t {
        std::size_t begin, end;
        int64_t value;
      };

      struct str_vec2i_t {
        std::size_t begin, end;
        fan::vec2i value;
      };

      static const char* digits = "0123456789";

      static int get_string_valuei(const fan::string& str, const fan::string& find, std::size_t offset = 0) {

        std::size_t found = str.find(find, offset);

        int64_t begin = str.find_first_of(digits, found);

        bool negative = 0;

        if (begin - 1 >= 0) {
          if (str[begin - 1] == '-') {
            negative = 1;
          }
        }

        std::size_t end = str.find_first_not_of(digits, begin);

        if (end == fan::string::npos) {
          end = str.size();
        }

        fan::string ret(str.begin() + begin - negative, str.begin() + end);
        return std::stoi(ret.data());
      }

      static str_int_t get_string_valuei_n(const fan::string& str, std::size_t offset = 0) {
        int64_t begin = str.find_first_of(digits, offset);

        bool negative = 0;

        if (begin - 1 >= 0) {
          if (str[begin - 1] == '-') {
            negative = 1;
          }
        }

        std::size_t end = str.find_first_not_of(digits, begin);
        if (end == fan::string::npos) {
          end = str.size();
        }

        fan::string ret(str.begin() + begin - negative, str.begin() + end);
        return { (std::size_t)begin, end, std::stoi(ret.c_str()) };
      }

      static str_vec2i_t get_string_valuevec2i_n(const fan::string& str, std::size_t offset = 0) {

        fan::vec2i v;

        std::size_t begin, end;

        auto r = get_string_valuei_n(str, offset);

        begin = r.begin;

        v.x = r.value;

        r = get_string_valuei_n(str, r.end);

        v.y = r.value;

        end = r.end;

        return { begin, end, v };
      }
    }
	}
}
