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

static constexpr fan::io::file::fs_mode FS_BINARY = std::ios_base::binary;
static constexpr fan::io::file::fs_mode FS_APP = std::ios_base::app;

std::string fan::io::file::extension(const std::string& file_path) {
  std::size_t dot_pos = file_path.find_last_of('.');
  std::size_t sep_pos = file_path.find_last_of("/\\");
  if (dot_pos != std::string::npos && (sep_pos == std::string::npos || dot_pos > sep_pos))
    return file_path.substr(dot_pos);
  return "";
}

bool fan::io::file::exists(std::string_view name) {
  if (name.empty()) return false;
  std::ifstream file(std::string(name), std::ios::binary);
  return file.good();
}

bool fan::io::file::rename(const std::string& from, const std::string& to) {
  return std::rename(from.c_str(), to.c_str());
}

std::filesystem::path fan::io::file::relative_path(const std::filesystem::path& path, const std::filesystem::path& base) {
  auto base_dir = std::filesystem::absolute(base);
  if (!std::filesystem::is_directory(base_dir))
    base_dir = base_dir.parent_path();
  return std::filesystem::relative(path, base_dir);
}

fan::io::file::fstream::fstream(const std::string& path) {
  file_name = path; open(path);
}

fan::io::file::fstream::fstream(const std::string& path, std::string* str) {
  file_name = path; open(path); read(str);
}

bool fan::io::file::fstream::open(const std::string& path) {
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

bool fan::io::file::fstream::read(const std::string& path, std::string* str) {
  file_name = path; open(path); return read(str);
}

bool fan::io::file::fstream::read(std::string* str) {
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
  fan::print_warning("file is not opened:");
  return 1;
}

bool fan::io::file::fstream::write(std::string* str) {
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
  fan::print_warning("file is not opened:");
  return 1;
}

bool fan::io::file::open(file_t** f, const std::string& path, const properties_t& p) {
  *f = std::fopen(path.c_str(), p.mode);
  return *f == nullptr;
}

bool fan::io::file::close(file_t* f) {
  int ret = std::fclose(f);
#if FAN_DEBUG >= fan_debug_low
  if (ret != 0) { fan::print_warning("failed to close file stream"); return 1; }
#endif
  return 0;
}

bool fan::io::file::read(file_t* f, void* data, std::uint64_t size, std::uint64_t elements) {
  std::uint64_t ret = std::fread(data, size, elements, f);
#if FAN_DEBUG >= fan_debug_low
  if (ret != elements && size != 0) { fan::print_warning("failed to read from file stream"); return 1; }
#endif
  return 0;
}

bool fan::io::file::write(file_t* f, void* data, std::uint64_t size, std::uint64_t elements) {
  std::uint64_t ret = std::fwrite(data, size, elements, f);
#if FAN_DEBUG >= fan_debug_low
  if (ret != elements && size != 0) { fan::print_warning("failed to write from file stream"); return 1; }
#endif
  return 0;
}

std::uint64_t fan::io::file::size(const std::string& filename) {
  std::ifstream f(filename.c_str(), std::ifstream::ate | std::ifstream::binary);
  return f.tellg();
}

bool fan::io::file::write(std::string_view path, const std::string& data, fs_mode mode) {
  std::ofstream ofile(std::string(path).c_str(), mode);
  if (ofile.fail()) { fan::print_warning("failed to write to:" + std::string(path)); return false; }
  ofile.write(data.data(), data.size());
  return !ofile.fail();
}

std::vector<std::string> fan::io::file::read_line(const std::string& path) {
  std::ifstream file(path.c_str(), std::ifstream::binary);
  if (file.fail()) fan::throw_error("path does not exist:" + path);
  std::vector<std::string> data;
  for (std::string line; std::getline(file, line);)
    data.push_back(line);
  return data;
}

bool fan::io::file::try_write(std::string path, const std::string& data, fs_mode mode) {
  if (fan::io::file::exists(path)) return false;
  return write(std::string_view(path), data, mode);
}

std::string fan::io::file::get_exe_path() {
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

std::uint64_t fan::io::file::file_size(std::string_view path) {
  std::ifstream f(std::string(path), std::ifstream::ate | std::ifstream::binary);
  return f.good() ? static_cast<std::uint64_t>(f.tellg()) : 0;
}

bool fan::io::file::read_bytes(std::string_view path, void* dst, std::size_t size) {
  std::ifstream f(std::string(path), std::ifstream::binary);
  if (!f.good()) return false;
  f.read(static_cast<char*>(dst), size);
  return f.good();
}

static std::string resolve(std::string_view path, const std::source_location& loc) {
  auto r = fan::io::file::find_relative_path(path, loc);
  return r.empty() ? std::string{} : r.generic_string();
}

bool fan::io::file::read(std::string_view path, std::string* str, std::size_t length,
  std::source_location loc)
{
  if (read_bytes(path, str->data(), length)) return false;
  auto r = resolve(path, loc);
  if (r.empty()) return true;
  str->resize(length);
  return !read_bytes(r, str->data(), length);
}

bool fan::io::file::read(std::string_view path, std::string* str,
  std::source_location loc)
{
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

std::string fan::io::file::read(std::string_view path, bool* success,
  std::source_location loc)
{
  std::string data;
  bool ret = read(path, &data, loc);
  if (success) *success = ret;
  return data;
}

std::filesystem::path fan::io::file::find_relative_path(std::string_view file_path,
  const std::source_location& location) {
  namespace fs = std::filesystem;
  if (file_path.empty()) return {};
  if (fan::io::file::exists(file_path)) return std::string(file_path);

  std::error_code ec;
  fs::path current_dir = fs::current_path(ec);
  if (ec) return {};

  auto try_candidate = [&](const fs::path& p) {
    if (fs::is_regular_file(p, ec) && !ec) {
      fs::path r = fs::relative(p, current_dir, ec);
      return ec ? p : r;
    }
    return fs::path{};
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
    if (fs::is_regular_file(fp, ec2))
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

  fan::print_impl("failed to find path for:", std::string{file_path}, ". called from", src_dir.generic_string());
  return {};
}