// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
  #define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 3
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define loco_vulkan

#define loco_window
#define loco_context

//#define loco_rectangle
#define loco_framebuffer
#define loco_sprite
#include _FAN_PATH(graphics/loco.h)

struct pile_t {

  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  pile_t() {
    fan::vec2 window_size = loco.get_window()->get_size();
    loco.open_matrices(
      &matrices,
      ortho_x,
      ortho_y
    );
    loco.get_window()->add_resize_callback([&](const fan::window_t::resize_cb_data_t& d) {
      fan::vec2 window_size = d.size;
      fan::vec2 ratio = window_size / window_size.max();
      std::swap(ratio.x, ratio.y);
      //matrices.set_ortho(
      //  ortho_x * ratio.x, 
      //  ortho_y * ratio.y
      //);
      viewport.set(loco.get_context(), 0, d.size, d.size);
    });
    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, window_size, window_size);
  }

  loco_t loco;
  loco_t::matrices_t matrices;
  fan::graphics::viewport_t viewport;
  fan::graphics::cid_t cid[2];
};
#include <winsock2.h>

int main()
{
  SOCKET s;
  struct sockaddr_in server, si_other;
  int slen, recv_len;
  WSADATA wsa;

  slen = sizeof(si_other);

  //Initialise winsock
  printf("\nInitialising Winsock...");
  if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0)
  {
    printf("Failed. Error Code : %d", WSAGetLastError());
    exit(EXIT_FAILURE);
  }

  // Create a UDP socket
  SOCKET sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if (sock == INVALID_SOCKET)
  {
    return -1;
  }

  // Get the maximum receive buffer size for the socket
  int max_buf_size = 0;
  int size = sizeof(max_buf_size);
  int ret = getsockopt(sock, SOL_SOCKET, SO_RCVBUF, (char*)&max_buf_size, &size);
  if (ret == SOCKET_ERROR)
  {
    return -1;
  }
  fan::print(max_buf_size);
  // The maximum receive buffer size is stored in the max_buf_size variable

  // Close the socket
  closesocket(sock);

  return 0;
}