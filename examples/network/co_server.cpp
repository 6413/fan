#include <fan/types/print.h>
#include <fan/network/sock.h>

int main() {
  fan::network::server_t server;
  auto server_task = server.run("127.0.0.1", 12346);

  while (!server_task.handle.done()) {
    get_event_loop().run_one();
  }

  return 0;
}