#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

import fan;
import fan.fmt;
import std;

struct wl_t {
  static void str(std::vector<std::uint32_t>& a, std::string_view s) {
    std::uint32_t len = s.size() + 1, words = (len + 3) / 4;
    a.push_back(len);
    a.resize(a.size() + words);
    std::memcpy(&a[a.size() - words], s.data(), s.size());
  }

  void init() {
    std::string path = std::format("{}/{}",
        std::getenv("XDG_RUNTIME_DIR"),
        std::getenv("WAYLAND_DISPLAY")
    );
    sockaddr_un addr{AF_UNIX};
    path.copy(addr.sun_path, sizeof(addr.sun_path) - 1);
    fd = socket(AF_UNIX, SOCK_STREAM, 0);
    connect(fd, (sockaddr*)&addr, sizeof(addr));
  }

  std::uint32_t new_id() {
    return next++;
  }

  void send(
      std::uint32_t id,
      std::uint32_t op,
      std::vector<std::uint32_t> a = {},
      int send_fd = -1)
  {
    std::uint32_t size = (a.size() + 2) * 4;
    a.insert(a.begin(), {id, size << 16 | op});
    iovec iov{a.data(), a.size() * 4};
    msghdr mh{};
    mh.msg_iov = &iov;
    mh.msg_iovlen = 1;
    alignas(cmsghdr) char ctl[CMSG_SPACE(sizeof(int))]{};
    if(send_fd >= 0) {
      mh.msg_control = ctl;
      mh.msg_controllen = sizeof(ctl);
      cmsghdr* c = CMSG_FIRSTHDR(&mh);
      c->cmsg_level = SOL_SOCKET;
      c->cmsg_type = SCM_RIGHTS;
      c->cmsg_len = CMSG_LEN(sizeof(int));
      std::memcpy(CMSG_DATA(c), &send_fd, sizeof(int));
    }
    sendmsg(fd, &mh, 0);
  }
  
  std::uint32_t bind(std::string_view iface, std::uint32_t version) {
    std::uint32_t id = new_id();
    std::vector<std::uint32_t> a{globals[std::string(iface)]};
    str(a, iface);
    a.push_back(version);
    a.push_back(id);
    send(registry, 0, a);
    return id;
  }

  template <typename F>
  void dispatch(F&& f) {
    std::uint8_t tmp[16384];
    ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
    if (n <= 0) {
      closed = true;
      return;
    }

    rx.insert(rx.end(), tmp, tmp + n);
    std::size_t off = 0;
    while (rx.size() - off >= 8) {
      std::uint32_t h[2];
      std::memcpy(h, &rx[off], 8);
      std::size_t size = h[1] >> 16;
      if (rx.size() - off < size) {
        break;
      }
      std::vector<std::uint32_t> a((size - 8) / 4);
      std::memcpy(a.data(), &rx[off + 8], size - 8);
      f(h[0], h[1] & 0xffff, a);
      off += size;
    }
    rx.erase(rx.begin(), rx.begin() + off);
  }

  std::map<std::string, std::uint32_t> globals;
  std::vector<std::uint8_t> rx;
  std::uint32_t next = 4;
  int fd = -1;
  bool closed = false;
  inline static constexpr std::uint32_t display = 1,
            registry = 2, callback = 3;
};

struct window_t : wl_t {

  static constexpr std::uint32_t w = 640, h = 480, size = w * h * 4;
  window_t() {
    init();
    send(wl_t::display, 1, {wl_t::registry});
    send(wl_t::display, 0, {wl_t::callback});
  }
  void on_event(std::uint32_t id, std::uint32_t op, const std::vector<std::uint32_t>& a)
  {
    std::uint32_t comp = 0, shm = 0, wm = 0, surface = 0, xsurf = 0, top = 0;
    bool configured = false;

    if (id == wl_t::display && op == 0 ) {
      std::println("error obj={} code={} {}", a[0], a[1], (const char*)&a[3]);
    }
    else if (id == wl_t::registry && op == 0) {
      globals[std::string((const char*)&a[2], a[1] - 1)] = a[0];
    }
    else if (id == wl_t::callback) {
      comp = bind("wl_compositor", 4);
      shm = bind("wl_shm", 1);
      wm = bind("xdg_wm_base", 1);
      surface = new_id();
      send(comp, 0, {surface});
      xsurf = new_id();
      send(wm, 2, {xsurf, surface});
      top = new_id();
      send(xsurf, 1, {top});
      send(surface, 6);
    }
    else if (id == wm && op == 0) {
      send(wm, 3, {a[0]});
    }
    else if (id == top && op == 1) {
      quit = true;
    }
    else if (id == xsurf && op == 0) {
      send(xsurf, 4, {a[0]});
      if (configured) {
        return;
      }
      configured = true;
      int mfd = memfd_create("shm", 0);
      ftruncate(mfd, size);
      auto* px = (std::uint32_t*)mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, mfd, 0);
      for (std::uint32_t y = 0; y < h; ++y) {
        for (std::uint32_t x = 0; x < w; ++x) {
          px[y * w + x] = 0xff000080u | (x * 255 / w) << 16 | (y * 255 / h) << 8;
        }
      }
      std::uint32_t pool = new_id(), buf = wl.new_id();
      send(shm, 0, {pool, size}, mfd);
      send(pool, 0, {buf, 0, w, h, w * 4, 1});
      close(mfd);
      send(surface, 1, {buf, 0, 0});
      send(surface, 2, {0, 0, w, h});
      send(surface, 6);
    } 
  }
};

int main() {
  window_t window;
  bool quit = false;
  while(!quit && !window.closed) {
    window.dispatch(on_event);
  }
}
