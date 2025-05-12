#include <coroutine>
#include <string>

import fan;

fan::event::task_t example_async_file_read_string(const std::string& path) {
  int fd = co_await fan::io::file::async_open(path);
  int offset = 0;
  std::string buffer;
  while (true) {
    // defaults to buffer size 1024
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

int main() {
  auto task_read_file = example_async_file_read_string("1.cpp");
  auto task_read_file_char = example_async_file_read_char("1.cpp");

  auto task = fan::io::file::async_read_cb("1.cpp", [](const std::string& chunk) {
    fan::printr(chunk);
  });

  auto task2 = fan::io::file::async_read_cb("1.cpp", [](const std::string& chunk) -> fan::event::task_t{
    fan::printr(chunk);
    co_await fan::co_sleep(100);
  });

  fan::event::loop();

  return 0;
}
