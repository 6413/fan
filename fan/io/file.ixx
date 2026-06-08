module;

#if defined(fan_platform_windows)
  #include <Windows.h>
#elif defined(fan_platform_unix)
  #include <unistd.h>
#endif

export module fan.io.file;

import std;

import fan.print.error;
export import fan.io.types;

export namespace fan {
  namespace io {
    namespace file {

      template <path_t P>
      std::string to_str(P&& p) {
        if constexpr (std::is_same_v<std::remove_cvref_t<P>, std::filesystem::path>)
          return p.string();
        else
          return std::string(std::string_view(p));
      }

      template <path_t P>
      inline std::string strip_extension(P&& file_path) {
        std::string path = to_str(std::forward<P>(file_path));
        std::size_t dot_pos = path.find_last_of('.');
        std::size_t sep_pos = path.find_last_of("/\\");
        if (dot_pos != std::string::npos && (sep_pos == std::string::npos || dot_pos > sep_pos))
          return path.substr(0, dot_pos);
        return path;
      }
      std::string extension(const std::string& file_path);
      inline void ensure_extension(std::string& path, std::string_view ext) {
        if (extension(path) != ext) {
          path += ext;
        }
      }
      bool exists(std::string_view path);
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
        std::string file_name;
        void* file_ptr = nullptr;
      };

      bool open(file_t** f, const std::string& path, const properties_t& p);
      bool close(file_t* f);
      bool read(file_t* f, void* data, std::uint64_t size, std::uint64_t elements);
      bool write(file_t* f, void* data, std::uint64_t size, std::uint64_t elements);
      std::uint64_t size(const std::string& filename);

      bool write(std::string_view path, const std::string& data, fs_mode mode);

      template <typename T>
      void write(std::string path, const std::vector<T>& vector, fs_mode mode);

      std::vector<std::string> read_line(const std::string& path);
      bool try_write(std::string path, const std::string& data, fs_mode mode);
      std::string get_exe_path();

      std::filesystem::path find_relative_path(std::string_view file_path,
        const std::source_location& location = std::source_location::current());

      std::uint64_t file_size(std::string_view path);
      bool read_bytes(std::string_view path, void* dst, std::size_t size);

      bool read(std::string_view path, std::string* str, std::size_t length,
        std::source_location loc = std::source_location::current());
      bool read(std::string_view path, std::string* str,
        std::source_location loc = std::source_location::current());
      std::string read(std::string_view path, bool* success = nullptr,
        std::source_location loc = std::source_location::current());

      template <typename T = std::uint8_t>
      std::vector<T> read_binary(std::string_view path) {
        auto sz = file_size(path);
        if (!sz || sz % sizeof(T)) return {};
        std::vector<T> v(sz / sizeof(T));
        if (!read_bytes(path, v.data(), sz)) return {};
        return v;
      }

      template <path_t P>
      bool write(P&& path, const std::string& data, fs_mode mode) {
        return write(std::string_view(to_str(std::forward<P>(path))), data, mode);
      }

      template <path_t P>
      bool exists(P&& p) { return exists(std::string_view(to_str(std::forward<P>(p)))); }

      template <path_t P>
      std::uint64_t file_size(P&& p) { return file_size(std::string_view(to_str(std::forward<P>(p)))); }

      template <path_t P>
      bool read_bytes(P&& p, void* dst, std::size_t sz) {
        return read_bytes(std::string_view(to_str(std::forward<P>(p))), dst, sz);
      }

      template <path_t P>
      bool read(P&& p, std::string* str,
        std::source_location loc = std::source_location::current()) {
        return read(std::string_view(to_str(std::forward<P>(p))), str, loc);
      }

      template <path_t P>
      std::string read(P&& p, bool* success = nullptr,
        std::source_location loc = std::source_location::current()) {
        return read(std::string_view(to_str(std::forward<P>(p))), success, loc);
      }

      template <path_t P>
      std::filesystem::path find_relative_path(P&& p,
        const std::source_location& loc = std::source_location::current()) {
        return find_relative_path(std::string_view(to_str(std::forward<P>(p))), loc);
      }

      template <path_t P>
      std::uint64_t file_size_p(P&& p) { return file_size(std::forward<P>(p)); }
    }
  }

  template <typename T>
  void write_to_file(fan::io::file::file_t* f, const T& o) {
    if constexpr (std::is_same<std::string, T>::value) {
      std::uint64_t len = o.size();
      fan::io::file::write(f, (std::uint8_t*)&len, sizeof(len), 1);
      fan::io::file::write(f, (std::uint8_t*)o.data(), len, 1);
    }
    else if constexpr (tmpl::is_std_vector<T>::value) {
      std::uint64_t len = o.size();
      fan::io::file::write(f, (std::uint8_t*)&len, sizeof(len), 1);
      fan::io::file::write(f, (std::uint8_t*)o.data(), len * sizeof(typename T::value_type), 1);
    }
    else {
      fan::io::file::write(f, (std::uint8_t*)&o, sizeof(o), 1);
    }
  }
}

export namespace fan::path {
  inline std::string join(const std::string& dir, const std::string& file) {
    return (std::filesystem::path(dir.empty() ? "." : dir) / file).string();
  }
}