#pragma once

#include <fan/types/vector.h>
#include <fan/types/print.h>

#include <fstream>
#include <string>
#undef index // xos.h

namespace fan {
	namespace io {
    namespace file {

      std::string extension(const std::string& file_path);

      bool exists(const std::string& name);

      bool rename(const std::string& from, const std::string& to);

      struct fstream {
        fstream() = default;
        fstream(const std::string& path);
        fstream(const std::string& path, std::string* str);

        bool open(const std::string& path);

        bool read(const std::string& path, std::string* str);

        bool read(std::string* str);

        bool write(std::string* str);

        std::fstream file;
        std::string file_name;
      };

      using file_t = FILE;

      struct properties_t {
        const char* mode;
      };

      bool close(file_t* f);
      bool open(file_t** f, const std::string& path, const properties_t& p);

      bool write(file_t* f, void* data, uint64_t size, uint64_t elements);
      bool read(file_t* f, void* data, uint64_t size, uint64_t elements);

      uint64_t file_size(const std::string& filename);

      bool write(
        std::string path,
        const std::string& data,
        decltype(std::ios_base::binary | std::ios_base::app) mode = std::ios_base::binary | std::ios_base::app
      );

      template <typename T>
      void write(
        std::string path,
        const std::vector<T>& vector,
        decltype(std::ios_base::binary | std::ios_base::app) mode = std::ios_base::binary | std::ios_base::app
      ) {
        std::ofstream ofile(path.c_str(), mode);
        if (ofile.fail()) {
          fan::throw_error("failed to write to:" + path);
        }
        ofile.write(reinterpret_cast<const char*>(&vector[0]), vector.size() * sizeof(T));
      }

      std::vector<std::string> read_line(const std::string& path);

      bool read(const std::string& path, std::string* str);

      template <typename T>
      std::vector<T> read(const std::string& path) {
        std::ifstream file(path.c_str(), std::ifstream::ate | std::ifstream::binary);
        if (!file.good()) {
          fan::print("path does not exist:" + path);
          exit(1);
        }
        std::vector<T> vector;
        uint64_t size = file.tellg();
        vector.resize(size / sizeof(T));
        file.seekg(0, std::ios::beg);
        file.read(reinterpret_cast<char*>(&vector[0]), size);
        return vector;
      }

      std::string extract_variable_type(const std::string& string_data, const std::string& var_name);

      struct str_int_t {
        std::size_t begin, end;
        int64_t value;
      };

      struct str_vec2i_t {
        std::size_t begin, end;
        fan::vec2i value;
      };

      static constexpr const char* digits = "0123456789";

      int get_string_valuei(const std::string& str, const std::string& find, std::size_t offset = 0);

      str_int_t get_string_valuei_n(const std::string& str, std::size_t offset = 0);

      str_vec2i_t get_string_valuevec2i_n(const std::string& str, std::size_t offset = 0);
    }
	}
}

namespace fan {

  namespace tmpl {
    template<typename>
    struct is_std_vector : std::false_type {};

    template<typename T, typename A>
    struct is_std_vector<std::vector<T, A>> : std::true_type {};
  }


  template <typename T>
  void write_to_file(fan::io::file::file_t* f, const T& o) {
    if constexpr (std::is_same<fan::string, T>::value ||
      std::is_same<std::string, T>::value) {
      uint64_t len = o.size();
      fan::io::file::write(f, (uint8_t*)&len, sizeof(len), 1);
      fan::io::file::write(f, (uint8_t*)o.data(), len, 1);
    }
    else if constexpr (tmpl::is_std_vector<T>::value) {
      uint64_t len = o.size();
      fan::io::file::write(f, (uint8_t*)&len, sizeof(len), 1);
      fan::io::file::write(f, (uint8_t*)o.data(), len * sizeof(typename T::value_type), 1);
    }
    else {
      fan::io::file::write(f, (uint8_t*)&o, sizeof(o), 1);
    }
  }
}