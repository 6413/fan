#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

import fan;
import fan.fmt;
import std;


int main() {
  std::string path = std::format("{}/{}",
      std::getenv("XDG_RUNTIME_DIR"),
      std::getenv("WAYLAND_DISPLAY")
  );
 
  constexpr std::uint32_t display = 1,
            registry = 2, callback = 3;
  constexpr std::uint32_t sync_op = 0, get_registry_op = 1, global_event = 0;

  constexpr std::uint32_t header(std::uint32_t size_bytes, std::uint32_t opcode) {
    return (size_bytes << 16) | opcode;
  }
  sockaddr_un addr{AF_UNIX};
  path.copy(addr.sun_path, sizeof(addr.sun_path) - 1);
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  connect(fd, (sockaddr*)&addr, sizeof(addr));
  std::uint32_t req[] = {
    display, header(12, get_registry_op), registry,
    display, header(12, sync_op), callback
  };
  write(fd, req, sizeof(req));
  std::uint32_t buf[4096]{};
  for (bool done = false; !done;) {
    std::size_t words = read(fd, buf, sizeof(buf)) / 4;
    for (std::size_t i = 0; i + 1 < words; i += buf[i + 1] >> 16 >> 2) {
      std::uint32_t id = buf[i], op = buf[i + 1] & 0xffff, len = buf[i + 3];
       if (id == registry && op == global_event) {
         std::uint32_t name = args[0], len = args[1];
         const char* iface = (const char*)&args[2];
         std::uint32_t version = args[2 + (len + 3) / 4];
         fan::printf(
             "name={} {} v{}",
             fan::paint(fan::colors::green, buf[i + 2]),
             (char*)&buf[i + 4], buf[i + 4 + (len + 3) / 4]
          );
      }
      done |= id == 3;
    }
  }
  fan::print(path);
}
