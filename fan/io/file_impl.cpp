module;

#include <fan/utility.h>
#include <cstdio>

#if defined(fan_platform_windows)
  #include <Windows.h>
#elif defined(fan_platform_unix)
  #include <unistd.h>
  #include <limits.h>
#endif

#include <string>
#include <fstream>
#include <source_location>
#include <filesystem>
#include <vector>

module fan.io.file;

import fan;

std::string fan::io::file::extension(const std::string& file_path) {
  size_t dotPosition = file_path.find_last_of('.');
  size_t sepPosition = file_path.find_last_of("/\\");
  if (dotPosition != std::string::npos && (sepPosition == std::string::npos || dotPosition > sepPosition)) {
    return file_path.substr(dotPosition);
  }
  else {
    return "";
  }
}

bool fan::io::file::rename(const std::string& from, const std::string& to) {
  return std::rename(from.c_str(), to.c_str());
}

std::filesystem::path fan::io::file::relative_path(const std::filesystem::path& path, const std::filesystem::path& base) {
  auto base_dir = std::filesystem::absolute(base);
  if (!std::filesystem::is_directory(base_dir)) {
    base_dir = base_dir.parent_path();
  }
  return std::filesystem::relative(path, base_dir);
}

fan::io::file::fstream::fstream(const std::string& path) {
  file_name = path;
  open(path);
}

fan::io::file::fstream::fstream(const std::string& path, std::string* str) {
  file_name = path;
  open(path);
  read(str);
}

bool fan::io::file::fstream::open(const std::string& path) {
  file_name = path;
  auto flags = std::ios::in | std::ios::out | std::ios::binary;
  if (!exists(path)) {
    flags |= std::ios::trunc;
  }
  file = std::fstream(path, flags);
  return !file.good();
}

bool fan::io::file::fstream::read(const std::string& path, std::string* str) {
  file_name = path;
  open(path);
  return read(str);
}

bool fan::io::file::fstream::read(std::string* str) {
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

bool fan::io::file::fstream::write(std::string* str) {
  auto flags = std::ios::in | std::ios::out | std::ios::trunc | std::ios::binary;
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

bool fan::io::file::open(file_t** f, const std::string& path, const properties_t& p) {
  *f = fopen(path.c_str(), p.mode);
  if (*f == nullptr) {
    return 1;
  }
  return 0;
}

bool fan::io::file::close(file_t* f) {
  int ret = fclose(f);
#if FAN_DEBUG >= fan_debug_low
  if (ret != 0) {
    fan::print_warning("failed to close file stream");
    return 1;
  }
#endif
  return 0;
}

bool fan::io::file::read(file_t* f, void* data, std::uint64_t size, std::uint64_t elements) {
  std::uint64_t ret = fread(data, size, elements, f);
#if FAN_DEBUG >= fan_debug_low
  if (ret != elements && size != 0) {
    fan::print_warning("failed to read from file stream");
    return 1;
  }
#endif
  return 0;
}

bool fan::io::file::write(file_t* f, void* data, std::uint64_t size, std::uint64_t elements) {
  std::uint64_t ret = fwrite(data, size, elements, f);
#if FAN_DEBUG >= fan_debug_low
  if (ret != elements && size != 0) {
    fan::print_warning("failed to write from file stream");
    return 1;
  }
#endif
  return 0;
}

std::uint64_t fan::io::file::size(const std::string& filename) {
  std::ifstream f(filename.c_str(), std::ifstream::ate | std::ifstream::binary);
  return f.tellg();
}

bool fan::io::file::write(std::string path, const std::string& data, fs_mode mode) {
  std::ofstream ofile(path.c_str(), mode);
  if (ofile.fail()) {
    fan::print_warning("failed to write to:" + path);
    return 0;
  }
  ofile.write(data.c_str(), data.size());
  return 1;
}

std::vector<std::string> fan::io::file::read_line(const std::string& path) {
  std::ifstream file(path.c_str(), std::ifstream::binary);
  if (file.fail()) {
    fan::throw_error("path does not exist:" + path);
  }
  std::vector<std::string> data;
  for (std::string line; std::getline(file, line); ) {
    data.push_back(line.c_str());
  }
  return data;
}

bool fan::io::file::try_write(std::string path, const std::string& data, fs_mode mode) {
  if (fan::io::file::exists(path)) {
    return false;
  }
  return write(path, data, mode);
}

std::string fan::io::file::get_exe_path() {
#if defined(fan_platform_windows)
  char buffer[MAX_PATH];
  DWORD len = GetModuleFileNameA(NULL, buffer, MAX_PATH);
  if (len == 0 || len == MAX_PATH) {
    return {};  // failure
  }
  return std::filesystem::path(buffer).parent_path().generic_string();

#elif defined(fan_platform_unix)
  char buffer[PATH_MAX];
  ssize_t count = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
  if (count <= 0) {
    return {};  // failure
  }
  buffer[count] = '\0';
  return std::filesystem::path(buffer).parent_path().generic_string();

#else
  fan::throw_error("get_exe_path not implemented on this platform");
  __unreachable();
#endif
}