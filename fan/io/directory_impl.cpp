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

std::uint8_t fan::io::vfs_provider_t::read(std::uint64_t offset) const {
  fan::bytes_t b; read_range(offset, 1, b); return b.empty() ? 0 : b[0];
}

std::uint64_t fan::io::vfs_provider_t::read_range(std::uint64_t offset, std::uint64_t length, fan::bytes_t& out_buffer) const {
  if (offset >= total_size) { out_buffer.clear(); return 0; }
  std::uint64_t actual = std::min(length, total_size - offset), end = offset + actual;
  out_buffer.assign(std::size_t(actual), 0);
  for (const auto& s : segments) {
    std::uint64_t s_end = s.start + s.size;
    if (s_end <= offset) { continue; }
    if (s.start >= end) { break; }
    std::uint64_t p = std::max(offset, s.start), n = std::min(end, s_end) - p, local = p - s.start;
    auto* dst = out_buffer.data() + std::size_t(p - offset);
    if (!s.bytes.empty()) {
      std::copy_n(s.bytes.data() + std::size_t(local), std::size_t(n), dst);
    }
    else {
      fan::io::file::file_t* f = nullptr;
      if (!fan::io::file::open(&f, s.file_path.string(), {"rb"})) {
        if (local > 0) { std::vector<std::uint8_t> dump(local); fan::io::file::read(f, dump.data(), 1, dump.size()); }
        if (fan::io::file::read(f, dst, 1, n)) { fan::io::file::close(f); throw std::runtime_error("short read"); }
        fan::io::file::close(f);
      }
      else { throw std::runtime_error("read failed: " + s.file_path.string()); }
    }
  }
  return actual;
}

void fan::io::vfs_provider_t::append_bytes(std::span<const std::uint8_t> bytes) {
  segment_t s; s.start = total_size; s.size = bytes.size(); s.bytes.assign(bytes.begin(), bytes.end());
  segments.push_back(std::move(s)); total_size += bytes.size();
}

void fan::io::vfs_provider_t::append_file(const std::filesystem::path& path, std::uint64_t size) {
  segment_t s; s.start = total_size; s.size = size; s.file_path = path;
  if (size <= 16 * 1024 * 1024) { s.bytes = fan::io::file::read_binary(path.string()); }
  segments.push_back(std::move(s)); total_size += size;
}

void fan::io::file::archive_extractor_t::put(std::uint8_t b) {
  if (state == state_e::data) { write_data(b); return; }
  tmp.push_back(b);
  if (tmp.size() != need) { return; }

  if (state == state_e::file_count) {
    file_count = fan::memory::read_le32(tmp.data()); file_index = 0; total_header += 4;
    set_state(file_count ? state_e::path_len : state_e::done, file_count ? 2 : 0);
  } else if (state == state_e::path_len) {
    path_len = fan::memory::read_le16(tmp.data()); total_header += 2; set_state(state_e::path, path_len);
  } else if (state == state_e::path) {
    archive_path.assign(reinterpret_cast<const char*>(tmp.data()), tmp.size()); total_header += path_len;
    if (default_out && file_count == 1 && out_dir.filename().string() == archive_path) { out_dir = "."; }
    set_state(state_e::size, 8);
  } else if (state == state_e::size) {
    remaining = fan::memory::read_le64(tmp.data()); total_header += 8; open_file();
    if (file_index + 1 == file_count) {
      std::size_t pad = (4 - (total_header & 3)) & 3;
      if (pad) { set_state(state_e::padding, pad); } else { enter_data(); }
    } else { enter_data(); }
  } else if (state == state_e::padding) { enter_data(); }
}

void fan::io::file::archive_extractor_t::enter_data() {
  if (remaining == 0) finish_file(); else set_state(state_e::data, 0);
}

void fan::io::file::archive_extractor_t::finish() {
  flush(); close_file(); if (state != state_e::done) { throw std::runtime_error("truncated archive"); }
}

void fan::io::file::archive_extractor_t::open_file() {
  std::filesystem::path p = out_dir / archive_path;
  fan::io::create_directory(p.parent_path().string());
  if (fan::io::file::open(&fp, p.string(), {"wb"})) { throw std::runtime_error("write failed"); }
}

void fan::io::file::archive_extractor_t::write_data(std::uint8_t b) {
  write_buffer.push_back(b);
  if (write_buffer.size() == write_buffer.capacity()) { flush(); }
  if (--remaining == 0) { finish_file(); }
}

void fan::io::file::archive_extractor_t::flush() {
  if (!fp || write_buffer.empty()) { return; }
  if (fan::io::file::write(fp, write_buffer.data(), 1, write_buffer.size())) { throw std::runtime_error("write failed"); }
  write_buffer.clear();
}

void fan::io::file::archive_extractor_t::close_file() {
  if (fp) { fan::io::file::close(fp); fp = nullptr; }
}
