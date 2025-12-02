module;

#include <fan/utility.h>

#if defined(fan_platform_windows)
  #include <Windows.h>
#elif defined(fan_platform_unix)
  #include <unistd.h>
  #include <limits.h>
#endif

#include <fstream>
#include <string>
#include <iostream>
#include <filesystem>
#include <source_location>
#include <vector>


export module fan.io.file;

import fan.print;

export namespace fan {
  namespace io {
    namespace file {

      template<typename T>
      concept path_t = std::same_as<T, std::filesystem::path>;

      std::string extension(const std::string& file_path);
      bool exists(const std::string& name);
      bool rename(const std::string& from, const std::string& to);

      std::filesystem::path relative_path(const std::filesystem::path& path, const std::filesystem::path& base);

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

      bool open(file_t** f, const std::string& path, const properties_t& p);
      bool close(file_t* f);
      bool read(file_t* f, void* data, std::uint64_t size, std::uint64_t elements);
      bool write(file_t* f, void* data, std::uint64_t size, std::uint64_t elements);
      std::uint64_t size(const std::string& filename);

      using fs_mode = decltype(std::ios_base::binary | std::ios_base::app);

      bool write(std::string path, const std::string& data, fs_mode mode);
      
      template <typename T>
      void write(
        std::string path,
        const std::vector<T>& vector,
        fs_mode mode = std::ios_base::binary | std::ios_base::app
      ) {
        std::ofstream ofile(path.c_str(), mode);
        if (ofile.fail()) {
          fan::throw_error("failed to write to:" + path);
        }
        ofile.write(reinterpret_cast<const char*>(&vector[0]), vector.size() * sizeof(T));
      }

      std::vector<std::string> read_line(const std::string& path);
      bool try_write(std::string path, const std::string& data, fs_mode mode);
      std::string get_exe_path();
      std::filesystem::path find_relative_path(const std::string& filepath, const std::source_location& location = std::source_location::current());
      bool read(const std::string& path, std::string* str);
      bool read(const std::string& path, std::string* str, std::size_t length);
      std::string read(const std::string& path, bool* success = nullptr);

      bool read(const path_t auto& path, std::string* str) {
        std::ifstream file(path, std::ifstream::ate | std::ifstream::binary);
        if (file.fail()) {
          return 1;
        }
        str->resize(file.tellg());
        file.seekg(0, std::ios::beg);
        file.read(&(*str)[0], str->size());
        file.close();
        return 0;
      }

      bool read(const path_t auto& path, std::string* str, std::size_t length) {
        std::ifstream file(path, std::ifstream::binary);
        if (file.fail()) {
          return 1;
        }
        str->resize(length);
        file.seekg(0, std::ios::beg);
        file.read(&(*str)[0], length);
        file.close();
        return 0;
      }

      template <typename T>
      std::vector<T> read(const std::string& path) {
        std::ifstream file(path.c_str(), std::ifstream::ate | std::ifstream::binary);
        if (!file.good()) {
          return {};
        }
        std::vector<T> vector;
        std::uint64_t size = file.tellg();
        vector.resize(size / sizeof(T));
        file.seekg(0, std::ios::beg);
        file.read(reinterpret_cast<char*>(&vector[0]), size);
        return vector;
      }

    }
  }
}

export namespace fan {
  namespace tmpl {
    template<typename>
    struct is_std_vector : std::false_type {};
    template<typename T, typename A>
    struct is_std_vector<std::vector<T, A>> : std::true_type {};
  }
}

export namespace fan {
  template <typename T>
  void write_to_file(fan::io::file::file_t* f, const T& o) {
    if constexpr (std::is_same<std::string, T>::value ||
                  std::is_same<std::string, T>::value) {
      std::uint64_t len = o.size();
      fan::io::file::write(f, (uint8_t*)&len, sizeof(len), 1);
      fan::io::file::write(f, (uint8_t*)o.data(), len, 1);
    }
    else if constexpr (tmpl::is_std_vector<T>::value) {
      std::uint64_t len = o.size();
      fan::io::file::write(f, (uint8_t*)&len, sizeof(len), 1);
      fan::io::file::write(f, (uint8_t*)o.data(), len * sizeof(typename T::value_type), 1);
    }
    else {
      fan::io::file::write(f, (uint8_t*)&o, sizeof(o), 1);
    }
  }
}