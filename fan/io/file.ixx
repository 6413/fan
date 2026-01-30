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
#include <istream>
#include <string_view>

export module fan.io.file;

import fan.print;

export namespace fan {
  namespace io {
    namespace file {

      template <typename T>
      concept path_t =
        std::is_convertible_v<T, std::string_view> ||
        std::is_convertible_v<T, const char*> ||
        std::is_same_v<std::remove_cvref_t<T>, std::filesystem::path>;


      std::string extension(const std::string& file_path);
      bool exists(const path_t auto& name) {
        std::string path;

        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(name)>, std::filesystem::path>) {
          path = name.string();
        }
        else {
          path = std::string {name};
        }

        std::ifstream file(path, std::ios::binary);
        return file.good();
      }

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
      std::filesystem::path find_relative_path(const path_t auto& file_path,
        const std::source_location& location = std::source_location::current()) {
        // dont look here
        namespace fs = std::filesystem;
        std::string_view filename = file_path;

        if (filename.empty()) {
          return {};
        }
        if (fan::io::file::exists(filename)) {
          return filename;
        }

        std::error_code ec;
        fs::path current_dir = fs::current_path(ec);
        if (ec) {
          return {};
        }

        auto try_candidate = [&](const fs::path& p) {
          if (fs::is_regular_file(p, ec) && !ec) {
            fs::path r = fs::relative(p, current_dir, ec);
            return ec ? p : r;
          }
          return fs::path {};
        };

        if (auto r = try_candidate(filename); !r.empty()) {
          return r;
        }

        fs::path src_dir = fs::path(location.file_name()).parent_path();
        fs::path exe_dir = get_exe_path();

        auto try_all = [&](const fs::path& base, const fs::path& name) {
          if (base.empty()) {
            return fs::path {};
          }
          return try_candidate(base / name);
        };

        if (auto r = try_all(src_dir, filename); !r.empty()) {
          return r;
        }
        if (auto r = try_all(current_dir, filename); !r.empty()) {
          return r;
        }

        if (!exe_dir.empty() && exe_dir != current_dir) {
          if (auto r = try_all(exe_dir, filename); !r.empty()) {
            return r;
          }
        }

        if (!exe_dir.empty()) {
          fs::path p = current_dir;
          while (p.has_parent_path()) {
            fs::path next = p.parent_path();
            if (next == p) {
              break;
            }
            p = next;
            if (auto r = try_all(p, filename); !r.empty()) {
              return r;
            }
            if (fs::equivalent(p, exe_dir, ec) && !ec) {
              break;
            }
          }
        }

        {
          fs::path name_only = fs::path(filename).filename();
          if (auto r = try_all(src_dir, name_only); !r.empty()) {
            return r;
          }
        }

        {
          std::error_code ec2;
          fs::path fp = fs::path(filename);

          if (fs::is_regular_file(fp, ec2)) {
            return fs::relative(fp, current_dir, ec2);
          }

          fs::path p = src_dir;
          while (p.has_parent_path()) {
            fs::path next = p.parent_path();
            if (next == p) {
              break;
            }
            if (auto r = try_candidate(p.filename() / fp); !r.empty()) {
              return r;
            }
            p = next;
          }
        }

        {
          fs::path name = fs::path(filename);
          fs::path p = src_dir;
          while (p.has_parent_path()) {
            if (auto r = try_all(p, name); !r.empty()) {
              return r;
            }
            fs::path next = p.parent_path();
            if (next == p) {
              break;
            }
            p = next;
          }
        }

        {
          fs::path name = fs::path(fs::absolute(filename).filename());
          fs::path p = exe_dir;
          while (p.has_parent_path()) {
            fs::path remaining = src_dir;
            while (!remaining.empty()) {
              if (auto r = try_all(p / remaining, name); !r.empty()) {
                return r;
              }
              auto it = remaining.begin();
              if (it != remaining.end()) {
                ++it;
                fs::path new_remaining;
                for (; it != remaining.end(); ++it) {
                  new_remaining /= *it;
                }
                remaining = new_remaining;
              }
              else {
                break;
              }
            }
            fs::path next = p.parent_path();
            if (next == p) {
              break;
            }
            p = next;
          }
        }
        fan::print("failed to find path for:", std::string{file_path}, ". called from", src_dir.generic_string());
        return {};
      }
      inline std::uint64_t file_size(const path_t auto& path) {
        std::ifstream f(path, std::ifstream::ate | std::ifstream::binary);
        return f.good() ? static_cast<std::uint64_t>(f.tellg()) : 0;
      }
      inline bool read_bytes(const path_t auto& path, void* dst, std::size_t size) {
        std::ifstream f(path, std::ifstream::binary);
        if (!f.good()) return false;
        f.read(static_cast<char*>(dst), size);
        return f.good();
      }
      bool read(const path_t auto& path, std::string* str, std::size_t length) {
        str->resize(length);
        return !read_bytes(path, str->data(), length);
      }
      std::string read(const path_t auto& path, bool* success = nullptr) {
        std::string data;
        bool ret = fan::io::file::read(path, &data);
        if (success) *success = ret;
        return data;
      }
      bool read(const path_t auto& path, std::string* str) {
        auto size = file_size(path);
        if (!size) return true;
        str->resize(size);
        return !read_bytes(path, str->data(), size);
      }
      template <typename T>
      std::vector<T> read(const path_t auto& path) {
        auto size = file_size(path);
        if (!size || size % sizeof(T)) return {};
        std::vector<T> v(size / sizeof(T));
        if (!read_bytes(path, v.data(), size)) return {};
        return v;
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