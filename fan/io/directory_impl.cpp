module;

module fan.io.directory;

import std;

import fan.print;
import fan.print.error;

std::string fan::io::file_to_directory(const std::string& file) {
  return file.substr(0, file.rfind((char)std::filesystem::path::preferred_separator)) +
    (char)std::filesystem::path::preferred_separator;
}
bool fan::io::directory_exists(const std::string& directory) {
  std::error_code ec;
  return std::filesystem::exists(directory, ec);
}
void fan::io::create_directory(const std::string& folders) {
  std::filesystem::create_directories(folders);
}
bool fan::io::iterate_sort_t::comp_cb(const iterate_sort_t& a, const iterate_sort_t& b) { return a.area > b.area; }
void fan::io::handle_string_out(std::string& str) {
  return std::replace(str.begin(), str.end(), '\\', '/');
}
void fan::io::handle_string_in(std::string& str) {
  std::replace(str.begin(), str.end(), '/', '\\');
}
bool fan::io::is_readable_path(const std::string& path) {
  try {
    std::filesystem::directory_iterator(path.c_str());
    std::filesystem::directory_entry(path.c_str());
    return true;
  }
  catch (const std::filesystem::filesystem_error& e) {
    fan::throw_error("error accessing directory: ", e.what());
  }
  return false;
}
std::string fan::io::exclude_path(const std::string& full_path) {
  std::size_t found = full_path.find_last_of('/');
  if (found == std::string::npos) {
    return full_path;
  }
  return full_path.substr(found + 1);
}
void fan::io::iterate_directory(
  const std::string& path,
  const std::function<void(const std::string& path, bool is_directory)>& function
) {
  if (!directory_exists(path)) {
    fan::throw_error("directory does not exist");
  }
  try {
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
      std::string str = entry.path().generic_string();
      function(str, entry.is_directory());
    }
  }
  catch (const std::filesystem::filesystem_error& e) {
    fan::throw_error("error accessing directory: ", e.what());
  }
}
void fan::io::iterate_directory(
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
    fan::throw_error("error accessing directory: ", e.what());
  }
}
void fan::io::iterate_directory_sorted_by_name(
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
    fan::throw_error("error accessing directory: ");
  }
}
void fan::io::iterate_directory_files(
  const std::string& path,
  const std::function<void(const std::string& path)>& function
) {
  if (!directory_exists(path)) {
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

void fan::io::iterate_files_recursive(
  const std::filesystem::path& path,
  const std::function<void(const std::filesystem::path& full, const std::filesystem::path& rel)>& function
) {
  std::error_code ec;
  for (const auto& e : std::filesystem::recursive_directory_iterator(path, ec)) {
    if (e.is_regular_file(ec)) {
      function(e.path(), std::filesystem::relative(e.path(), path));
    }
  }
}

bool fan::io::is_safe_path(const std::filesystem::path& path) {
  if (path.empty() || path.is_absolute()) { return false; }
  for (const auto& part : path) { if (part == "..") { return false; } }
  return true;
}
