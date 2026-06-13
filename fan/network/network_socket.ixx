module;
#include <coroutine>
#include <uv.h>
#undef min
#undef max
#undef NO_ERROR
#include <fan/utility.h>

export module fan.network.socket;

import std;
import fan.utility;
import fan.event.types;
import fan.event;
import fan.network;

export namespace fan::network {
  struct socket_t;
  struct peer_slot_t;

  struct peer_nr_t {
    std::uint32_t index {std::uint32_t(-1)};
    bool valid()     const { return index != std::uint32_t(-1); }
    void invalidate() { index = std::uint32_t(-1); }
    bool operator==(const peer_nr_t&) const = default;
  };

  struct read_data_t {
    const std::byte* data;
    std::size_t      size;
  };

  using read_cb_t = std::function<bool(void* ext_data, peer_slot_t&, read_data_t)>;
  using write_cb_t = std::function<bool(void* ext_data, peer_slot_t&, std::vector<std::byte>&)>;

  struct read_cb_node_t {
    void* ext_data {nullptr};
    read_cb_t     cb;
    std::uint32_t next {std::uint32_t(-1)};
    std::uint32_t prev {std::uint32_t(-1)};
  };

  struct write_cb_node_t {
    void* ext_data {nullptr};
    write_cb_t    cb;
    std::uint32_t next {std::uint32_t(-1)};
    std::uint32_t prev {std::uint32_t(-1)};
  };

  template<typename node_t>
  struct cb_list_t {
    static constexpr std::uint32_t null = std::uint32_t(-1);

    struct nr_t {
      std::uint32_t idx {null};
      bool valid() const { return idx != null; }

      nr_t next(const cb_list_t& list) const {
        if (!valid()) { return {}; }
        return {list.nodes[idx].next};
      }
    };

    std::vector<node_t>        nodes;
    std::vector<std::uint32_t> free_stack;
    std::uint32_t              head {null};
    std::uint32_t              tail {null};

    nr_t push_back(node_t n) {
      std::uint32_t idx;
      if (!free_stack.empty()) {
        idx = free_stack.back();
        free_stack.pop_back();
        nodes[idx] = std::move(n);
      }
      else {
        idx = std::uint32_t(nodes.size());
        nodes.push_back(std::move(n));
      }
      nodes[idx].next = null;
      nodes[idx].prev = tail;
      if (tail != null) { nodes[tail].next = idx; }
      else { head = idx; }
      tail = idx;
      return {idx};
    }

    void remove(nr_t nr) {
      if (!nr.valid()) { return; }
      auto& n = nodes[nr.idx];
      if (n.prev != null) { nodes[n.prev].next = n.next; }
      else { head = n.next; }
      if (n.next != null) { nodes[n.next].prev = n.prev; }
      else { tail = n.prev; }
      free_stack.push_back(nr.idx);
    }

    bool empty() const { return head == null; }

    template<typename F>
    void dispatch_from(std::uint32_t from_idx, F&& f) {
      std::uint32_t i = from_idx;
      while (i != null) {
        std::uint32_t nxt = nodes[i].next;
        if (!f(nr_t {i}, nodes[i])) { break; }
        i = nxt;
      }
    }

    void dispatch_from_head(auto&& f) { dispatch_from(head, f); }

    nr_t insert_before(nr_t before, node_t n) {
      if (!before.valid()) { return push_back(std::move(n)); }
      std::uint32_t idx;
      if (!free_stack.empty()) {
        idx = free_stack.back();
        free_stack.pop_back();
        nodes[idx] = std::move(n);
      }
      else {
        idx = std::uint32_t(nodes.size());
        nodes.push_back(std::move(n));
      }
      std::uint32_t prev_idx = nodes[before.idx].prev;
      nodes[idx].next = before.idx;
      nodes[idx].prev = prev_idx;
      if (prev_idx != null) { nodes[prev_idx].next = idx; }
      else { head = idx; }
      nodes[before.idx].prev = idx;
      return {idx};
    }
  };

  using read_cb_list_t = cb_list_t<read_cb_node_t>;
  using write_cb_list_t = cb_list_t<write_cb_node_t>;

  struct peer_slot_t {
    static constexpr std::uint32_t null = std::uint32_t(-1);

    struct uv_tcp_deleter_t {
      void operator()(uv_tcp_t* p) const {
        uv_close(reinterpret_cast<uv_handle_t*>(p), [](uv_handle_t* h) {
          delete reinterpret_cast<uv_tcp_t*>(h);
        });
      }
    };

    peer_slot_t() = default;
    peer_slot_t(const peer_slot_t&) = delete;
    peer_slot_t& operator=(const peer_slot_t&) = delete;
    peer_slot_t(peer_slot_t&&) = default;
    peer_slot_t& operator=(peer_slot_t&&) = default;

    void block() { blocked = true; }
    void resume_connect();

    read_cb_list_t::nr_t  add_read_cb(void* ext_data, read_cb_t cb) {
      return read_cbs.push_back({.ext_data = ext_data, .cb = std::move(cb)});
    }
    write_cb_list_t::nr_t add_write_cb(void* ext_data, write_cb_t cb) {
      return write_cbs.push_back({.ext_data = ext_data, .cb = std::move(cb)});
    }
    void remove_read_cb(read_cb_list_t::nr_t nr) { read_cbs.remove(nr); }
    void remove_write_cb(write_cb_list_t::nr_t nr) { write_cbs.remove(nr); }

    void dispatch_read(read_data_t rd) {
      read_cbs.dispatch_from_head([&](read_cb_list_t::nr_t, read_cb_node_t& node) {
        return node.cb(node.ext_data, *this, rd);
      });
    }

    void dispatch_read(read_cb_list_t::nr_t from, read_data_t rd) {
      if (!from.valid()) { return; }
      read_cbs.dispatch_from(from.idx, [&](read_cb_list_t::nr_t, read_cb_node_t& node) {
        return node.cb(node.ext_data, *this, rd);
      });
    }

    void write(std::vector<std::byte> buf) {
      write_cbs.dispatch_from_head([&](write_cb_list_t::nr_t, write_cb_node_t& node) {
        return node.cb(node.ext_data, *this, buf);
      });
    }

    void dispatch_write(write_cb_list_t::nr_t from, std::vector<std::byte> buf) {
      if (!from.valid()) { return; }
      write_cbs.dispatch_from(from.idx, [&](write_cb_list_t::nr_t, write_cb_node_t& node) {
        return node.cb(node.ext_data, *this, buf);
      });
    }

    peer_nr_t      nr;
    socket_t* owner {nullptr};
    std::uint32_t  connect_idx {0};
    bool           blocked {false};
    bool           connected {false};

    read_cb_list_t  read_cbs;
    write_cb_list_t write_cbs;

    std::unique_ptr<uv_tcp_t, uv_tcp_deleter_t> tcp;
    std::vector<std::byte>                       read_buf;
  };

  struct extension_base_t {
    virtual void        construct_peer_data(void* p) const = 0;
    virtual void        destroy_peer_data(void* p)   const = 0;
    virtual std::size_t peer_data_size()              const = 0;
    virtual ~extension_base_t() = default;

    std::function<void(void* ext_data, peer_slot_t&)> on_connect;
    std::function<void(void* ext_data, peer_slot_t&)> on_disconnect;

    std::vector<std::byte> peer_data_buf;

    void* peer_data(peer_nr_t nr) {
      return peer_data_buf.data() + std::size_t(nr.index) * peer_data_size();
    }

    void alloc_for(peer_nr_t nr) {
      std::size_t sz = peer_data_size();
      std::size_t needed = (std::size_t(nr.index) + 1) * sz;
      if (peer_data_buf.size() < needed) {
        std::size_t old_n = peer_data_buf.size() / sz;
        peer_data_buf.resize(needed);
        for (std::size_t i = old_n; i <= nr.index; ++i) {
          construct_peer_data(peer_data_buf.data() + i * sz);
        }
      }
      else {
        construct_peer_data(peer_data(nr));
      }
    }

    void free_for(peer_nr_t nr) {
      std::size_t off = std::size_t(nr.index) * peer_data_size();
      if (off < peer_data_buf.size()) {
        destroy_peer_data(peer_data_buf.data() + off);
      }
    }
  };

  template<typename data_t>
  struct extension_impl_t : extension_base_t {
    void construct_peer_data(void* p) const override { new (p) data_t {}; }
    void destroy_peer_data(void* p)   const override { reinterpret_cast<data_t*>(p)->~data_t(); }
    std::size_t peer_data_size()      const override { return sizeof(data_t); }

    data_t& data_of(peer_slot_t& peer) {
      return *reinterpret_cast<data_t*>(peer_data(peer.nr));
    }
  };

  struct sentinel_write_req_t {
    uv_write_t             req;
    std::vector<std::byte> buf;
  };

  struct sentinel_ext_t : extension_impl_t<std::monostate> {
    sentinel_ext_t() {
      on_connect = nullptr;
      on_disconnect = nullptr;
    }

    static bool write_cb(void*, peer_slot_t& peer, std::vector<std::byte>& buf) {
      if (!peer.tcp) { return true; }
      auto* stream = reinterpret_cast<uv_stream_t*>(peer.tcp.get());
      if (!uv_is_writable(stream)) { return true; }
      auto* d = new sentinel_write_req_t;
      d->buf = buf;
      d->req.data = d;
      uv_buf_t uvbuf = uv_buf_init(
        reinterpret_cast<char*>(d->buf.data()),
        d->buf.size()
      );
      uv_write(&d->req, stream, &uvbuf, 1, [](uv_write_t* req, int) {
        delete static_cast<sentinel_write_req_t*>(req->data);
      });
      return true;
    }
  };

  struct peer_bll_t {
    std::vector<peer_slot_t>   slots;
    std::vector<std::uint32_t> free_stack;

    peer_nr_t alloc() {
      std::uint32_t idx;
      if (!free_stack.empty()) {
        idx = free_stack.back();
        free_stack.pop_back();
        slots[idx] = peer_slot_t {};
      }
      else {
        idx = std::uint32_t(slots.size());
        slots.emplace_back();
      }
      slots[idx].nr = {idx};
      return {idx};
    }

    void free(peer_nr_t nr) {
      slots[nr.index] = peer_slot_t {};
      free_stack.push_back(nr.index);
    }

    peer_slot_t& operator[](peer_nr_t nr) { return slots[nr.index]; }

    bool valid(peer_nr_t nr) const {
      return nr.index < std::uint32_t(slots.size()) && slots[nr.index].connected;
    }
  };

  struct socket_t {
    template<typename ext_t, typename... args_t>
    ext_t* add_extension(args_t&&... args) {
      auto  e = std::make_unique<ext_t>(std::forward<args_t>(args)...);
      auto* raw = e.get();
      extensions.push_back(std::move(e));
      return raw;
    }

    peer_nr_t open_peer(const std::string& ip, std::uint16_t port) {
      peer_nr_t    nr = peers.alloc();
      peer_slot_t& peer = peers[nr];
      peer.owner = this;
      peer.nr = nr;
      peer.connected = true;

      peer.tcp.reset(new uv_tcp_t);
      uv_tcp_init(reinterpret_cast<uv_loop_t*>(fan::event::get_loop()), peer.tcp.get());
      peer.tcp->data = &peer;

      for (auto& ext : extensions) {
        ext->alloc_for(nr);
      }

      peer.add_write_cb(nullptr, &sentinel_ext_t::write_cb);

      auto* req = new uv_connect_t;
      req->data = this;

      struct sockaddr_in addr;
      uv_ip4_addr(ip.c_str(), port, &addr);
      uv_tcp_connect(req, peer.tcp.get(),
        reinterpret_cast<const sockaddr*>(&addr),
        [](uv_connect_t* req, int status) {
        auto* sock = static_cast<socket_t*>(req->data);
        auto* peer = static_cast<peer_slot_t*>(
          reinterpret_cast<uv_tcp_t*>(req->handle)->data
          );
        delete req;
        if (status != 0) {
          sock->close_peer(peer->nr);
          return;
        }
        sock->_start_read(*peer);
        sock->_begin_connect_chain(*peer);
      }
      );

      return nr;
    }

    void close_peer(peer_nr_t nr) {
      if (!peers.valid(nr)) { return; }
      peer_slot_t& peer = peers[nr];
      _run_disconnect_chain(peer);
      for (auto& ext : extensions) {
        ext->free_for(nr);
      }
      peers.free(nr);
    }

    void _advance_connect_chain(peer_slot_t& peer) {
      peer.blocked = false;
      _begin_connect_chain(peer);
    }

    peer_slot_t& operator[](peer_nr_t nr) { return peers[nr]; }
    bool         valid(peer_nr_t nr) { return peers.valid(nr); }

    std::vector<std::unique_ptr<extension_base_t>> extensions;
    peer_bll_t                                     peers;

  private:
    void _begin_connect_chain(peer_slot_t& peer) {
      while (peer.connect_idx < std::uint32_t(extensions.size())) {
        auto& ext = *extensions[peer.connect_idx];
        ++peer.connect_idx;
        if (ext.on_connect) {
          ext.on_connect(ext.peer_data(peer.nr), peer);
          if (peer.blocked) { return; }
        }
      }
    }

    void _run_disconnect_chain(peer_slot_t& peer) {
      for (auto& ext : extensions) {
        if (ext->on_disconnect) {
          ext->on_disconnect(ext->peer_data(peer.nr), peer);
        }
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
        auto* peer = static_cast<peer_slot_t*>(stream->data);
        if (nread <= 0) { return; }
        peer->dispatch_read({peer->read_buf.data(), std::size_t(nread)});
      }
      );
    }
  };

  inline void peer_slot_t::resume_connect() {
    if (owner) { owner->_advance_connect_chain(*this); }
  }

} // namespace fan::network
