module;
#include <coroutine>
#include <uv.h>
#undef min
#undef max
#undef NO_ERROR
#include <fan/utility.h>

export module fan.network.socket;

import std;
import fan.types;
import fan.utility;
import fan.event.types;
import fan.event;
import fan.network;

export namespace fan::network {
  struct socket_t;
  struct peer_slot_t;

  struct peer_nr_t {
    std::uint32_t index {std::uint32_t(-1)};
    bool valid() const { return index != std::uint32_t(-1); }
    void invalidate() { index = std::uint32_t(-1); }
    bool operator==(const peer_nr_t&) const = default;
  };

  struct read_data_t {
    const std::uint8_t* data;
    std::size_t         size;
    operator std::string_view() const {
      return {reinterpret_cast<const char*>(data), size};
    }
    inline friend std::ostream& operator<<(std::ostream& os, const read_data_t& rd) {
      return os.write(reinterpret_cast<const char*>(rd.data), rd.size);
    }
  };

  using read_cb_t = std::function<bool(void* ext_data, peer_slot_t&, read_data_t)>;
  using write_cb_t = std::function<bool(void* ext_data, peer_slot_t&, fan::bytes_t&)>;

  template<typename cb_t>
  struct cb_node_t {
    void* ext_data {nullptr};
    cb_t          cb;
    std::uint32_t next {std::uint32_t(-1)};
    std::uint32_t prev {std::uint32_t(-1)};
  };

  template<typename node_t>
  struct cb_list_t {
    static constexpr std::uint32_t null = std::uint32_t(-1);

    struct nr_t {
      std::uint32_t idx {null};
      bool valid() const { return idx != null; }
      nr_t next(const cb_list_t& list) const { return valid() ? nr_t{list.nodes[idx].next} : nr_t{}; }
    };

    std::vector<node_t>        nodes;
    std::vector<std::uint32_t> free_stack;
    std::uint32_t              head {null};
    std::uint32_t              tail {null};

    std::uint32_t _alloc_idx(node_t n) {
      std::uint32_t idx = free_stack.empty() ? std::uint32_t(nodes.size()) : free_stack.back();
      if (free_stack.empty()) {
        nodes.push_back(std::move(n));
      } else {
        free_stack.pop_back();
        nodes[idx] = std::move(n);
      }
      return idx;
    }

    nr_t push_back(node_t n) {
      std::uint32_t idx = _alloc_idx(std::move(n));
      nodes[idx].next = null;
      nodes[idx].next = null;
      nodes[idx].prev = tail;
      if (tail != null) { nodes[tail].next = idx; } else { head = idx; }
      tail = idx;
      return {idx};
    }

    void remove(nr_t nr) {
      if (!nr.valid()) { return; }
      auto& n = nodes[nr.idx];
      if (n.prev != null) { nodes[n.prev].next = n.next; } else { head = n.next; }
      if (n.next != null) { nodes[n.next].prev = n.prev; } else { tail = n.prev; }
      free_stack.push_back(nr.idx);
    }

    bool empty() const { return head == null; }

    template<typename F>
    void dispatch_from(std::uint32_t from_idx, F&& f) {
      for (std::uint32_t i = from_idx; i != null; ) {
        std::uint32_t nxt = nodes[i].next;
        if (!f(nr_t{i}, nodes[i])) { break; }
        i = nxt;
      }
    }

    nr_t insert_before(nr_t before, node_t n) {
      if (!before.valid()) { return push_back(std::move(n)); }
      std::uint32_t idx = _alloc_idx(std::move(n));
      std::uint32_t p = nodes[before.idx].prev;
      nodes[idx].next = before.idx;
      nodes[idx].prev = p;
      if (p != null) { nodes[p].next = idx; } else { head = idx; }
      nodes[before.idx].prev = idx;
      return {idx};
    }
  };

  using read_cb_list_t = cb_list_t<cb_node_t<read_cb_t>>;
  using write_cb_list_t = cb_list_t<cb_node_t<write_cb_t>>;

  struct peer_slot_t {
    static constexpr std::uint32_t null = std::uint32_t(-1);

    struct uv_tcp_deleter_t {
      void operator()(uv_tcp_t* p) const {
        uv_close(reinterpret_cast<uv_handle_t*>(p), [](uv_handle_t* h) { delete reinterpret_cast<uv_tcp_t*>(h); });
      }
    };

    peer_slot_t() = default;
    peer_slot_t(const peer_slot_t&) = delete;
    peer_slot_t& operator=(const peer_slot_t&) = delete;
    peer_slot_t(peer_slot_t&&) = default;
    peer_slot_t& operator=(peer_slot_t&&) = default;

    void block() { blocked = true; }
    void resume_connect();

    read_cb_list_t::nr_t  add_read_cb(void* ext_data, read_cb_t cb) { return read_cbs.push_back({.ext_data = ext_data, .cb = std::move(cb)}); }
    write_cb_list_t::nr_t add_write_cb(void* ext_data, write_cb_t cb) { return write_cbs.push_back({.ext_data = ext_data, .cb = std::move(cb)}); }
    void remove_read_cb(read_cb_list_t::nr_t nr) { read_cbs.remove(nr); }
    void remove_write_cb(write_cb_list_t::nr_t nr) { write_cbs.remove(nr); }

    void dispatch_read(read_data_t rd) { dispatch_read({read_cbs.head}, rd); }
    void dispatch_read(read_cb_list_t::nr_t from, read_data_t rd) {
      if (from.valid()) { read_cbs.dispatch_from(from.idx, [&](auto, auto& node) { return node.cb(node.ext_data, *this, rd); }); }
    }

    void write(fan::bytes_t buf) { dispatch_write({write_cbs.head}, buf); }
    void dispatch_write(write_cb_list_t::nr_t from, fan::bytes_t buf) {
      if (from.valid()) { write_cbs.dispatch_from(from.idx, [&](auto, auto& node) { return node.cb(node.ext_data, *this, buf); }); }
    }

    peer_nr_t     nr;
    socket_t* owner {nullptr};
    std::uint32_t connect_idx {0};
    bool          blocked {false};
    bool          connected {false};

    read_cb_list_t  read_cbs;
    write_cb_list_t write_cbs;

    std::unique_ptr<uv_tcp_t, uv_tcp_deleter_t> tcp;
    fan::bytes_t                                read_buf;
  };

  struct extension_base_t {
    virtual void alloc_for(peer_nr_t nr) = 0;
    virtual void free_for(peer_nr_t nr) = 0;
    virtual void* peer_data(peer_nr_t nr) = 0;
    virtual ~extension_base_t() = default;

    std::function<void(void* ext_data, peer_slot_t&)> on_connect;
    std::function<void(void* ext_data, peer_slot_t&)> on_disconnect;
  };

  template<typename data_t>
  struct extension_impl_t : extension_base_t {
    void alloc_for(peer_nr_t nr) override {
      if (peer_data_buf.size() <= nr.index) { peer_data_buf.resize(std::size_t(nr.index) + 1); }
      peer_data_buf[nr.index] = std::make_unique<data_t>();
    }
    void free_for(peer_nr_t nr) override {
      if (nr.index < peer_data_buf.size()) { peer_data_buf[nr.index].reset(); }
    }
    void* peer_data(peer_nr_t nr) override { return peer_data_buf[nr.index].get(); }
    data_t& data_of(peer_slot_t& peer) { return *static_cast<data_t*>(peer_data(peer.nr)); }

    std::vector<std::unique_ptr<data_t>> peer_data_buf;
  };

  struct sentinel_write_req_t {
    uv_write_t   req;
    fan::bytes_t buf;
  };

  struct sentinel_ext_t : extension_impl_t<std::monostate> {
    static bool write_cb(void*, peer_slot_t& peer, fan::bytes_t& buf) {
      if (!peer.tcp) { return true; }
      auto* stream = reinterpret_cast<uv_stream_t*>(peer.tcp.get());
      if (!uv_is_writable(stream)) { return true; }

      auto* d = new sentinel_write_req_t{ {}, buf };
      d->req.data = d;
      uv_buf_t uvbuf = uv_buf_init(reinterpret_cast<char*>(d->buf.data()), d->buf.size());
      uv_write(&d->req, stream, &uvbuf, 1, [](uv_write_t* req, int) { delete static_cast<sentinel_write_req_t*>(req->data); });
      return true;
    }
  };

  struct peer_bll_t {
    std::vector<std::unique_ptr<peer_slot_t>> slots;
    std::vector<std::uint32_t>                free_stack;

    peer_nr_t alloc() {
      std::uint32_t idx = free_stack.empty() ? std::uint32_t(slots.size()) : free_stack.back();
      if (free_stack.empty()) {
        slots.push_back(std::make_unique<peer_slot_t>());
      } else {
        free_stack.pop_back();
        slots[idx] = std::make_unique<peer_slot_t>();
      }
      slots[idx]->nr = {idx};
      return {idx};
    }

    void free(peer_nr_t nr) {
      slots[nr.index].reset();
      free_stack.push_back(nr.index);
    }

    peer_slot_t& operator[](peer_nr_t nr) { return *slots[nr.index]; }
    bool exists(peer_nr_t nr) const { return nr.index < slots.size() && slots[nr.index] != nullptr; }
    bool valid(peer_nr_t nr) const { return exists(nr) && slots[nr.index]->connected; }
  };

  struct socket_t {
    template<typename ext_t, typename... args_t>
    ext_t* add_extension(args_t&&... args) {
      extensions.push_back(std::make_unique<ext_t>(std::forward<args_t>(args)...));
      return static_cast<ext_t*>(extensions.back().get());
    }

    peer_nr_t open_peer(const std::string& ip, std::uint16_t port) {
      peer_nr_t nr = peers.alloc();
      peer_slot_t& peer = peers[nr];
      peer.owner = this;
      peer.nr = nr;
      peer.connected = true;

      peer.tcp.reset(new uv_tcp_t);
      uv_tcp_init(reinterpret_cast<uv_loop_t*>(fan::event::get_loop()), peer.tcp.get());
      peer.tcp->data = &peer;

      for (auto& ext : extensions) { ext->alloc_for(nr); }
      peer.add_write_cb(nullptr, &sentinel_ext_t::write_cb);

      auto* req = new uv_connect_t;
      req->data = this;
      struct sockaddr_in addr;
      uv_ip4_addr(ip.c_str(), port, &addr);

      uv_tcp_connect(req, peer.tcp.get(), reinterpret_cast<const sockaddr*>(&addr), [](uv_connect_t* req, int status) {
        auto* sock = static_cast<socket_t*>(req->data);
        auto* peer = static_cast<peer_slot_t*>(reinterpret_cast<uv_tcp_t*>(req->handle)->data);
        delete req;
        if (status != 0) {
          std::cout << "connect failed: " << uv_strerror(status) << "\n";
          sock->close_peer(peer->nr);
          return;
        }
        sock->_start_read(*peer);
        sock->_begin_connect_chain(*peer);
      });

      return nr;
    }

    void close_peer(peer_nr_t nr) {
      if (!peers.valid(nr)) { return; }
      peer_slot_t& peer = peers[nr];
      _run_disconnect_chain(peer);
      for (auto& ext : extensions) { ext->free_for(nr); }
      peers.free(nr);
    }

    void _advance_connect_chain(peer_slot_t& peer) {
      peer.blocked = false;
      _begin_connect_chain(peer);
    }

    peer_slot_t& operator[](peer_nr_t nr) { return peers[nr]; }
    bool         valid(peer_nr_t nr) { return peers.valid(nr); }
    bool         exists(peer_nr_t nr) const { return peers.exists(nr); }

    std::vector<std::unique_ptr<extension_base_t>> extensions;
    peer_bll_t                                     peers;

  private:
    void _begin_connect_chain(peer_slot_t& peer) {
      while (peer.connect_idx < std::uint32_t(extensions.size())) {
        auto& ext = *extensions[peer.connect_idx++];
        if (ext.on_connect) {
          ext.on_connect(ext.peer_data(peer.nr), peer);
          if (peer.blocked) { return; }
        }
      }
    }

    void _run_disconnect_chain(peer_slot_t& peer) {
      for (auto& ext : extensions) {
        if (ext->on_disconnect) { ext->on_disconnect(ext->peer_data(peer.nr), peer); }
      }
    }

    void _start_read(peer_slot_t& peer) {
      uv_read_start(
        reinterpret_cast<uv_stream_t*>(peer.tcp.get()),
        [](uv_handle_t* h, std::size_t suggested, uv_buf_t* buf) {
          auto* p = static_cast<peer_slot_t*>(h->data);
          p->read_buf.resize(suggested);
          *buf = uv_buf_init(reinterpret_cast<char*>(p->read_buf.data()), suggested);
        },
        [](uv_stream_t* stream, ssize_t nread, const uv_buf_t*) {
          if (nread > 0) {
            auto* peer = static_cast<peer_slot_t*>(stream->data);
            peer->dispatch_read({peer->read_buf.data(), std::size_t(nread)});
          }
        }
      );
    }
  };

  inline void peer_slot_t::resume_connect() {
    if (owner) { owner->_advance_connect_chain(*this); }
  }
} // namespace fan::network