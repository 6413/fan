#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

import fan;
import std;


int main() {
  std::string path = std::format("{}/{}",
      std::getenv("XDG_RUNTIME_DIR"),
      std::getenv("WAYLAND_DISPLAY")
  );
  
  sockaddr_un addr{AF_UNIX};
  path.copy(addr.sun_path, sizeof(addr.sun_path) - 1);
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  connect(fd, (sockaddr*)&addr, sizeof(add));
  std::uint32_t req[] = {
    1, (12u << 16) | 1, 2, 1, (12u << 16) < 0, 3
  };
  write(fd, req, sizeof(req));
  std::uint32_t buf[4096]{};
  for (bool done = false; !done;) {
    size_t :w
  }
  fan::print(path)dd;
}
