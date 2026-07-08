module;

#include <fan/utility.h>

#include <climits>

#if defined(fan_platform_windows)
#include <Windows.h>
#elif defined(fan_platform_unix)
#include <unistd.h>
#endif

module fan.io.file;

import fan.print;
import fan.utility;

namespace fan::io::file {

  std::string extension(const std::string& file_path) {
    std::size_t dot_pos = file_path.find_last_of('.');
    std::size_t sep_pos = file_path.find_last_of("/\\");
    if (dot_pos != std::string::npos && (sep_pos == std::string::npos || dot_pos > sep_pos))
      return file_path.substr(dot_pos);
    return "";
  }

  bool exists(std::string_view name) {
    std::error_code ec;
    return !name.empty() && std::filesystem::exists(name, ec);
  }

  bool rename(const std::string& from, const std::string& to) {
    return std::rename(from.c_str(), to.c_str());
  }

  std::filesystem::path relative_path(const std::filesystem::path& path, const std::filesystem::path& base) {
    auto base_dir = std::filesystem::absolute(base);
    if (!std::filesystem::is_directory(base_dir))
      base_dir = base_dir.parent_path();
    return std::filesystem::relative(path, base_dir);
  }

  fstream::fstream(const std::string& path) {
    file_name = path; open(path);
  }

  fstream::fstream(const std::string& path, std::string* str) {
    file_name = path; open(path); read(str);
  }

  bool fstream::open(const std::string& path) {
    file_name = path;
    auto flags = std::ios::in | std::ios::out | std::ios::binary;
    if (!exists(path)) {
      auto resolved = find_relative_path(path);
      if (!resolved.empty()) file_name = resolved.generic_string();
      else flags |= std::ios::trunc;
    }
    auto* f = new std::fstream(file_name, flags);
    delete static_cast<std::fstream*>(file_ptr);
    file_ptr = f;
    return !f->good();
  }

  bool fstream::read(const std::string& path, std::string* str) {
    file_name = path; open(path); return read(str);
  }

  bool fstream::read(std::string* str) {
    auto* f = static_cast<std::fstream*>(file_ptr);
    if (f && f->is_open()) {
      f->seekg(0, std::ios::end);
      str->resize(f->tellg());
      f->seekg(0, std::ios::beg);
      f->read(&(*str)[0], str->size());
      return 0;
    }
    auto resolved = find_relative_path(file_name);
    if (!resolved.empty()) {
      std::fstream f2(resolved, std::ios::in | std::ios::binary);
      if (f2.is_open()) {
        f2.seekg(0, std::ios::end);
        str->resize(f2.tellg());
        f2.seekg(0, std::ios::beg);
        f2.read(&(*str)[0], str->size());
        return 0;
      }
    }
    fan::print_log(fan::log_level_e::warning, "fan::io::file", "file is not opened:");
    return 1;
  }

  bool fstream::write(std::string* str) {
    auto flags = std::ios::in | std::ios::out | std::ios::trunc | std::ios::binary;
    auto* f = new std::fstream(file_name, flags);
    delete static_cast<std::fstream*>(file_ptr);
    file_ptr = f;
    if (f->is_open()) {
      f->write(&(*str)[0], str->size());
      f->flush();
      flags &= ~std::ios::trunc;
      auto* f2 = new std::fstream(file_name, flags);
      delete f;
      file_ptr = f2;
      return 0;
    }
    fan::print_log(fan::log_level_e::warning, "fan::io::file", "file is not opened:");
    return 1;
  }

  bool open(file_t** f, const std::string& path, const properties_t& p) {
    *f = std::fopen(path.c_str(), p.mode);
    return *f == nullptr;
  }

  bool close(file_t* f) {
    int ret = std::fclose(f);
  #if FAN_DEBUG >= fan_debug_low
    if (ret != 0) { fan::print_log(fan::log_level_e::warning, "fan::io::file", "failed to close file stream"); return 1; }
  #endif
    return 0;
  }

  bool read(file_t* f, void* data, std::uint64_t size, std::uint64_t elements) {
    std::uint64_t ret = std::fread(data, size, elements, f);
  #if FAN_DEBUG >= fan_debug_low
    if (ret != elements && size != 0) { fan::print_log(fan::log_level_e::warning, "fan::io::file", "failed to read from file stream"); return 1; }
  #endif
    return 0;
  }

  bool write(file_t* f, void* data, std::uint64_t size, std::uint64_t elements) {
    std::uint64_t ret = std::fwrite(data, size, elements, f);
  #if FAN_DEBUG >= fan_debug_low
    if (ret != elements && size != 0) { fan::print_log(fan::log_level_e::warning, "fan::io::file", "failed to write from file stream"); return 1; }
  #endif
    return 0;
  }

  std::uint64_t size(const std::string& filename) {
    std::ifstream f(filename.c_str(), std::ifstream::ate | std::ifstream::binary);
    return f.tellg();
  }

  std::vector<std::string> read_line(const std::string& path) {
    std::ifstream file(path.c_str(), std::ifstream::binary);
    if (file.fail()) fan::throw_error("path does not exist:" + path);
    std::vector<std::string> data;
    for (std::string line; std::getline(file, line);)
      data.push_back(line);
    return data;
  }

  bool try_write(std::string path, const std::string& data, fs_mode mode) {
    if (exists(path)) return false;
    return write(std::string_view(path), data, mode);
  }

  std::string get_exe_path() {
  #if defined(fan_platform_windows)
    char buffer[MAX_PATH];
    DWORD len = GetModuleFileNameA(NULL, buffer, MAX_PATH);
    if (len == 0 || len == MAX_PATH) return {};
    return std::filesystem::path(buffer).parent_path().generic_string();
  #elif defined(fan_platform_unix)
    char buffer[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (count <= 0) return {};
    buffer[count] = '\0';
    return std::filesystem::path(buffer).parent_path().generic_string();
  #else
    fan::throw_error("get_exe_path not implemented");
    __unreachable();
  #endif
  }

  std::uint64_t file_size(std::string_view path) {
    std::ifstream f(std::string(path), std::ifstream::ate | std::ifstream::binary);
    return f.good() ? static_cast<std::uint64_t>(f.tellg()) : 0;
  }

  bool read_bytes(std::string_view path, void* dst, std::size_t size) {
    std::ifstream f(std::string(path), std::ifstream::binary);
    if (!f.good()) return false;
    f.read(static_cast<char*>(dst), size);
    return f.good();
  }

  static std::string resolve(std::string_view path, const std::source_location& loc) {
    auto r = find_relative_path(path, loc);
    return r.empty() ? std::string {} : r.generic_string();
  }

  bool read(std::string_view path, std::string* str, std::size_t length,
    std::source_location loc) {
    if (read_bytes(path, str->data(), length)) return false;
    auto r = resolve(path, loc);
    if (r.empty()) return true;
    str->resize(length);
    return !read_bytes(r, str->data(), length);
  }

  bool read(std::string_view path, std::string* str,
    std::source_location loc) {
    auto sz = file_size(path);
    if (!sz) {
      auto r = resolve(path, loc);
      if (r.empty()) return true;
      sz = file_size(r);
      if (!sz) return true;
      str->resize(sz);
      return !read_bytes(r, str->data(), sz);
    }
    str->resize(sz);
    return !read_bytes(path, str->data(), sz);
  }

  std::string read(std::string_view path, bool* success,
    std::source_location loc) {
    std::string data;
    bool ret = read(path, &data, loc);
    if (success) *success = ret;
    return data;
  }

  std::filesystem::path find_relative_path(std::string_view file_path,
    const std::source_location& location) {
    namespace fs = std::filesystem;
    if (file_path.empty()) return {};
    if (exists(file_path)) return std::string(file_path);

    std::error_code ec;
    fs::path current_dir = fs::current_path(ec);
    if (ec) return {};

    auto try_candidate = [&](const fs::path& p) {
      if (fs::exists(p, ec) && !ec) {
        fs::path r = fs::relative(p, current_dir, ec);
        return ec ? p : r;
      }
      return fs::path {};
    };

    if (auto r = try_candidate(std::string(file_path)); !r.empty()) return r;

    fs::path src_dir = fs::path(location.file_name()).parent_path();
    fs::path exe_dir = get_exe_path();

    auto try_all = [&](const fs::path& base, const fs::path& name) -> fs::path {
      if (base.empty()) return {};
      return try_candidate(base / name);
    };

    if (auto r = try_all(src_dir, std::string(file_path)); !r.empty()) return r;
    if (auto r = try_all(current_dir, std::string(file_path)); !r.empty()) return r;

    if (!exe_dir.empty() && exe_dir != current_dir) {
      if (auto r = try_all(exe_dir, std::string(file_path)); !r.empty()) return r;
    }

    if (!exe_dir.empty()) {
      fs::path p = current_dir;
      while (p.has_parent_path()) {
        fs::path next = p.parent_path();
        if (next == p) break;
        p = next;
        if (auto r = try_all(p, std::string(file_path)); !r.empty()) return r;
        if (fs::equivalent(p, exe_dir, ec) && !ec) break;
      }
    }

    {
      fs::path name_only = fs::path(file_path).filename();
      if (auto r = try_all(src_dir, name_only); !r.empty()) return r;
    }

    {
      std::error_code ec2;
      fs::path fp = fs::path(file_path);
      if (fs::exists(fp, ec2))
        return fs::relative(fp, current_dir, ec2);
      fs::path p = src_dir;
      while (p.has_parent_path()) {
        fs::path next = p.parent_path();
        if (next == p) break;
        if (auto r = try_candidate(p.filename() / fp); !r.empty()) return r;
        p = next;
      }
    }

    {
      fs::path name = fs::path(file_path);
      fs::path p = src_dir;
      while (p.has_parent_path()) {
        if (auto r = try_all(p, name); !r.empty()) return r;
        fs::path next = p.parent_path();
        if (next == p) break;
        p = next;
      }
    }

    {
      fs::path name = fs::path(fs::absolute(std::string(file_path)).filename());
      fs::path p = exe_dir;
      while (p.has_parent_path()) {
        fs::path remaining = src_dir;
        while (!remaining.empty()) {
          if (auto r = try_all(p / remaining, name); !r.empty()) return r;
          auto it = remaining.begin();
          if (it != remaining.end()) {
            ++it;
            fs::path new_remaining;
            for (; it != remaining.end(); ++it) new_remaining /= *it;
            remaining = new_remaining;
          }
          else break;
        }
        fs::path next = p.parent_path();
        if (next == p) break;
        p = next;
      }
    }

    fan::print_impl("failed to find path for:", std::string {file_path}, ". called from", src_dir.generic_string());
    return {};
  }

  bool is_pe(const fan::bytes_t& d) {
    if (d.size() < 0x40 || d[0] != 'M' || d[1] != 'Z') {
      return false;
    }
    std::uint32_t pe;
    std::memcpy(&pe, d.data() + 0x3c, sizeof(pe));
    return pe <= d.size() - 4 && std::memcmp(d.data() + pe, "PE\0\0", 4) == 0;
  }

  void ensure_extension(std::string& path, std::string_view ext) {
    if (extension(path) != ext) {
      path += ext;
    }
  }

  bool is_temp_file(std::string_view path) {
    return path.empty() || path.back() == '~' || path.find(".tmp") != std::string_view::npos;
  }
  bool is_up_to_date(std::string_view source_path, std::string_view cache_path) {
    std::error_code ec_src, ec_cache;
    if (!std::filesystem::exists(cache_path, ec_cache) || ec_cache) return false;
    auto src_time = std::filesystem::last_write_time(source_path, ec_src);
    auto cache_time = std::filesystem::last_write_time(cache_path, ec_cache);
    if (ec_src || ec_cache || src_time > cache_time) return false;
    return std::filesystem::file_size(cache_path, ec_cache) > 0 && !ec_cache;
  }

  void file_writer_t::write_repeat(std::uint8_t b, std::size_t n) {
    std::array<std::uint8_t, 4096> buf; buf.fill(b);
    while (n) { std::size_t w = std::min<std::size_t>(n, buf.size()); write_bytes(std::span<const std::uint8_t>(buf.data(), w)); n -= w; }
  }
  void file_writer_t::write_bytes(std::span<const std::uint8_t> bytes) {
    if (!bytes.empty() && fan::io::file::write(fp, (void*)bytes.data(), 1, bytes.size())) { throw std::runtime_error("write failed"); }
  }
}

std::string fan::path::join(const std::string& dir, const std::string& file) {
  return (std::filesystem::path(dir.empty() ? "." : dir) / file).string();
}
