#include <coroutine>
#include <string>
#include <fstream>
#include <uv.h>
#undef min
#undef max

import fan;

fan::event::task_t example_async_file_read_string(const std::string& path) {
  int fd = co_await fan::io::file::async_open(path);
  int offset = 0;
  std::string buffer;
  while (true) {
    // defaults to buffer size 4096
    std::size_t result = co_await fan::io::file::async_read(fd, &buffer, offset);
    if (result == 0) {
      break;
    }

    fan::printr(buffer);
    offset += result;
    co_await fan::co_sleep(100);
  }

  co_await fan::io::file::async_close(fd);
}

fan::event::task_t example_async_file_read_char(const std::string& path) {
  int fd = co_await fan::io::file::async_open(path);
  int offset = 0;
  while (true) {
    char buffer[64];
    std::size_t result = co_await fan::io::file::async_read(fd, buffer, sizeof(buffer), offset);
    if (result == 0) {
      break;
    }
    buffer[result] = '\0';
    fan::printr(buffer);
    offset += result;
    co_await fan::co_sleep(100);
  }

  co_await fan::io::file::async_close(fd);
}

fan::event::task_t example_async_file_write_string(const std::string& path, const std::string& data) {
  int fd = co_await fan::io::file::async_open(path, fan::fs_out);
 
  size_t offset = 0;
  size_t buffer_size = 4096;
  size_t total_written = 0;

  while (total_written < data.size()) {
    size_t remaining = data.size() - total_written;
    size_t to_write = std::min(remaining, buffer_size);

    std::string buffer(data.data() + total_written, to_write);

    std::size_t written = co_await fan::io::file::async_write(fd, buffer.data(), buffer.size(), offset + total_written);

    if (written == 0) {
      fan::throw_error("write failed");
    }

    total_written += written;
  }

  co_await fan::io::file::async_close(fd);
}



int main() {
  auto example_tasks = []() -> fan::event::task_t {
    std::string data;
    auto task_read_file = example_async_file_read_string("CMakeLists.txt");
    auto task_read_file_char = example_async_file_read_char("CMakeLists.txt");

    auto task = fan::io::file::async_read_cb("CMakeLists.txt", [&data](const std::string& chunk) {
      data += chunk;
      fan::printr(chunk);
    });

    auto task2 = fan::io::file::async_read_cb("CMakeLists.txt", [](const std::string& chunk) -> fan::event::task_t {
      fan::printr(chunk);
      co_await fan::co_sleep(100);
    });

    co_await task_read_file;
    co_await task_read_file_char;
    co_await task;
    co_await task2;
    auto task3 = example_async_file_write_string("2.txt", data);
    co_await task3;
  }();


  fan::event::loop();

  return 0;
}
