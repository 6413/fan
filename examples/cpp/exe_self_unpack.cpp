#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

import std;
import fan;

static int g_exit_code = 1;
static bool g_done = false;

static std::filesystem::path get_executable_path() {
#if defined(_WIN32)
  char buf[MAX_PATH];
  GetModuleFileNameA(NULL, buf, MAX_PATH);
  return std::filesystem::path(buf);
#elif defined(__linux__)
  char buf[4096];
  ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  return n > 0 ? std::filesystem::path(std::string(buf, n)) : std::filesystem::path("");
#else
  return "";
#endif
}

static std::filesystem::path temp_exe_path(std::string name) {
  if (name.empty()) {
    name = "fan_unpacked";
  }

#if defined(_WIN32)
  if (!name.ends_with(".exe")) {
    name += ".exe";
  }
#endif

  auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  return std::filesystem::temp_directory_path() / (std::to_string(stamp) + "_" + name);
}

void pack_exe(const std::filesystem::path& self_path, const std::filesystem::path& target_exe) {
  fan::bytes_t target_data = fan::io::file::read_binary(target_exe.string());
  if (target_data.empty()) {
    fan::print_error("failed to read target");
    return;
  }

  std::vector<fan::io::file_buffer_t> files;
  fan::io::file_buffer_t fb;
  fb.path = target_exe.filename().string();
  fb.data = target_data;
  files.push_back(fb);

  fan::bytes_t compressed = fan::fcs::compress(files);
  fan::bytes_t self_data = fan::io::file::read_binary(self_path.string());

  if (self_data.empty()) {
    fan::print_error("failed to read self");
    return;
  }

  std::filesystem::path out_path = target_exe.parent_path() / ("packed_" + target_exe.filename().string());
  std::uint64_t payload_size = compressed.size();
  std::string_view magic = "FANPACK1";

  fan::bytes_t out_data;
  out_data.reserve(self_data.size() + compressed.size() + sizeof(payload_size) + magic.size());
  out_data.insert(out_data.end(), self_data.begin(), self_data.end());
  out_data.insert(out_data.end(), compressed.begin(), compressed.end());
  
  std::uint8_t* size_ptr = reinterpret_cast<std::uint8_t*>(&payload_size);
  out_data.insert(out_data.end(), size_ptr, size_ptr + sizeof(payload_size));
  out_data.insert(out_data.end(), magic.begin(), magic.end());

  if (!fan::io::file::write(out_path.string(), out_data)) {
    fan::print_error("failed to write packed exe");
    return;
  }

#if defined(__linux__)
  std::filesystem::permissions(
    out_path, 
    std::filesystem::perms::owner_exec | std::filesystem::perms::group_exec | std::filesystem::perms::others_exec, 
    std::filesystem::perm_options::add
  );
#endif

  fan::print("successfully packed into:", out_path.string());
}

fan::event::task_t run_embedded(std::filesystem::path self_path) {
  fan::bytes_t self_data = fan::io::file::read_binary(self_path.string());
  if (self_data.size() < 16) {
    g_done = true;
    co_return;
  }

  std::string_view magic(reinterpret_cast<const char*>(self_data.data() + self_data.size() - 8), 8);
  if (magic != "FANPACK1") {
    fan::print("drag and drop an executable onto this file to pack it");
    g_exit_code = 0;
    g_done = true;
    co_return;
  }

  std::uint64_t payload_size;
  std::memcpy(&payload_size, self_data.data() + self_data.size() - 16, sizeof(payload_size));
  if (self_data.size() < 16 + payload_size) {
    fan::print_error("corrupted payload");
    g_done = true;
    co_return;
  }

  fan::bytes_t archive(self_data.end() - 16 - payload_size, self_data.end() - 16);
  std::vector<fan::io::file_buffer_t> files;

  try {
    files = fan::fcs::decompress(archive);
  }
  catch (...) {
    fan::print_error("decompress failed");
    g_done = true;
    co_return;
  }

  if (files.empty()) {
    fan::print_error("archive is empty");
    g_done = true;
    co_return;
  }

  auto exe_path = temp_exe_path(files[0].path);

  if (!fan::io::file::write(exe_path.string(), files[0].data)) {
    fan::print_error("failed to write:", exe_path.string());
    g_done = true;
    co_return;
  }

#if defined(__linux__)
  std::filesystem::permissions(
    exe_path, 
    std::filesystem::perms::owner_exec | std::filesystem::perms::group_exec | std::filesystem::perms::others_exec, 
    std::filesystem::perm_options::add
  );
#endif

  auto r = co_await fan::process::run_async({exe_path.string()}, nullptr);

  std::error_code ec;
  std::filesystem::remove(exe_path, ec);

  g_exit_code = r.exit_code;
  g_done = true;
}

int main(int argc, char** argv) {
  std::filesystem::path self_path = get_executable_path();

  if (argc > 1) {
    pack_exe(self_path, argv[1]);
    return 0;
  }

  fan::event::add_awaitable(run_embedded(self_path));

  while (!g_done) {
    fan::event::loop(fan::event::get_loop(), true);
    fan::event::process();
  }

  return g_exit_code;
}