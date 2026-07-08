module;

#if defined(FAN_NETWORK)
#include <coroutine>
#include <uv.h>
#undef min
#undef max
#undef NO_ERROR
#if defined(__clang__) || !defined(__GNUC__) || defined(FAN_NETWORK_ENABLE_HTTP_ON_GCC)
#define FAN_NETWORK_HTTP_ENABLED
#endif
#if defined(FAN_NETWORK_HTTP_ENABLED) && \
  (defined(__clang__) || !defined(__GNUC__) || defined(FAN_NETWORK_ENABLE_CURL_ON_GCC))
#define FAN_NETWORK_CURL_ENABLED
#endif
#ifdef FAN_NETWORK_CURL_ENABLED
  #include <curl/curl.h>
  #include <curl/multi.h>
#endif
#include <openssl/sha.h>
#endif

module fan.network;

#if defined(FAN_NETWORK)
namespace fan {
  namespace network {

    getaddrinfo_t::getaddrinfo_t(const char* node, const char* service, struct addrinfo* hints) :
      data(std::make_unique<getaddrinfo_data_t>()) {
      data->getaddrinfo_handle.data = data.get();
      uv_getaddrinfo((uv_loop_t*)fan::event::get_loop(), &data->getaddrinfo_handle, [](uv_getaddrinfo_t* getaddrinfo_handle, int status, struct addrinfo* res) {
        auto data = static_cast<getaddrinfo_data_t*>(getaddrinfo_handle->data);
        if (status == UV_ECANCELED) {
          delete data;
          return;
        }
        data->ready = true;
        data->status = status;

        [[likely]] if (data->co_handle) {
          data->co_handle();
        }
      }, node, service, hints);
    }

    getaddrinfo_t::~getaddrinfo_t() {
      if (data) {
        uv_freeaddrinfo(data->getaddrinfo_handle.addrinfo);
        if (uv_cancel(reinterpret_cast<uv_req_t*>(&data->getaddrinfo_handle)) == 0) {
          data.release();
        }
      }
    }

    connector_t::~connector_t() {
      [[unlikely]] if (data && data->status == 1) {
        data->req.cb = [](uv_connect_t* req, int) {
          delete static_cast<connector_data_t*>(req->data);
        };
        data.release();
      }
    }

    int connector_t::await_resume() {
      if (data->status != 0) {
        fan::throw_error(std::string("connection failed with:") + uv_strerror(data->status));
      }
      return data->status;
    }

    void raw_reader_t::stop() {
      uv_read_stop(stream.get());
    }

    int raw_reader_t::start() noexcept {
      return uv_read_start(stream.get(),
        [](uv_handle_t* handle, std::size_t suggested_size, uv_buf_t* buf) {
          auto self = static_cast<raw_reader_t*>(handle->data);
          self->buf.resize(suggested_size);
          *buf = uv_buf_init(self->buf.data(), suggested_size);
        },
        [](uv_stream_t* req, ssize_t nread, const uv_buf_t* buf) {
          auto self = static_cast<raw_reader_t*>(req->data);
          self->nread = nread;
          [[likely]] if (self->co_handle && nread != 0) {
            self->co_handle();
          }
        }
      );
    }

    data_t raw_reader_t::await_resume() {
      data_t data;
      data.status = nread < 0 ? nread : error_code::ok;
      if (nread > 0) {
        data.buffer = buffer_t(buf.begin(), buf.begin() + nread);
      }
      nread = 0;
      co_handle = nullptr;
      return data;
    }

    raw_reader_t::~raw_reader_t() {
      if (stream) {
        uv_read_stop(stream.get());
      }
      buf.clear();
    }

    int raw_writer_t::write(const buffer_t& some_data) {
      if (!uv_is_writable(data->stream.get())) { 
        fan::throw_error("tcp write failed: socket not writable (not connected)");
      }

      data->to_write = some_data;
      data->status = 1;
      uv_buf_t buf = uv_buf_init(data->to_write.data(), data->to_write.size());
      int r = uv_write(&data->write_handle, data->stream.get(), &buf, 1,
        [](uv_write_t* write_handle, int status) {
          auto data = static_cast<writer_data_t*>(write_handle->data);
          data->status = status;
          if (data->co_handle) {
            data->co_handle();
          }
        });
      if (r < 0) {
        std::string msg = std::string("tcp write failed: ") + uv_err_name(r) + " (" + uv_strerror(r) + ")";
        fan::throw_error(msg);
      }
      return r;
    }

    raw_writer_t::~raw_writer_t() {
      if (data && data->status == 1) {
        data->write_handle.cb = [](uv_write_t* write_handle, int) {
          delete static_cast<writer_data_t*>(write_handle->data);
        };
        data.release();
      }
    }

    void ring_buffer_t::push_back(const char* data, std::size_t len) {
      if (size_ + len > capacity_) {
        grow(size_ + len);
      }
      for (std::size_t i = 0; i < len; ++i) {
        buffer_[tail_] = data[i];
        tail_ = (tail_ + 1) % capacity_;
      }
      size_ += len;
    }

    void ring_buffer_t::consume(std::size_t len) {
      if (len > size_) len = size_;
      head_ = (head_ + len) % capacity_;
      size_ -= len;
    }

    void ring_buffer_t::peek(char* dest, std::size_t len) const {
      std::size_t actual_len = std::min(len, size_);
      std::size_t h = head_;
      for (std::size_t i = 0; i < actual_len; ++i) {
        dest[i] = buffer_[h];
        h = (h + 1) % capacity_;
      }
    }

    std::pair<const char*, std::size_t> ring_buffer_t::get_contiguous() const {
      if (size_ == 0) return { nullptr, 0 };
      std::size_t available = std::min(size_, capacity_ - head_);
      return { &buffer_[head_], available };
    }

    void ring_buffer_t::grow(std::size_t new_capacity) {
      std::vector<char> new_buffer(new_capacity * 2);
      std::size_t h = head_;
      for (std::size_t i = 0; i < size_; ++i) {
        new_buffer[i] = buffer_[h];
        h = (h + 1) % capacity_;
      }
      buffer_ = std::move(new_buffer);
      head_ = 0;
      tail_ = size_;
      capacity_ = new_capacity * 2;
    }

    void reader_t::setup_fixed_size_read(ssize_t len) {
      reading_header = false;
      expected_size = len;
      is_raw_read = false;
      is_fixed_size_read = true;
    }

    void reader_t::setup_raw_read() {
      reading_header = false;
      expected_size = 0;
      is_raw_read = true;
      is_fixed_size_read = false;
    }

    void reader_t::setup_header_read() {
      reading_header = true;
      expected_size = 0;
      is_raw_read = false;
      is_fixed_size_read = false;
    }

    void reader_t::stop() {
      is_reading = false;
      uv_read_stop(stream.get());
    }

    int reader_t::start() noexcept {
      is_reading = true;
      return uv_read_start(stream.get(),
        [](uv_handle_t* handle, std::size_t suggested_size, uv_buf_t* buf) {
          auto self = static_cast<reader_t*>(handle->data);
          std::size_t buf_size = std::min(suggested_size, self->temp_buf.size());
          *buf = uv_buf_init(self->temp_buf.data(), buf_size);
        },
        [](uv_stream_t* req, ssize_t nread, const uv_buf_t* buf) {
          auto self = static_cast<reader_t*>(req->data);
          self->nread = nread;
          if (nread > 0) {
            self->accumulated_buf.push_back(self->temp_buf.data(), nread);
          }
          [[likely]] if (self->co_handle && nread != 0) {
            self->co_handle();
          }
        }
      );
    }

    bool reader_t::await_ready() const {
      if (nread < 0) return true;
      if (is_raw_read) {
        return nread > 0 || !accumulated_buf.empty();
      }
      else if (is_fixed_size_read) {
        return accumulated_buf.size() >= expected_size;
      }
      else {
        if (reading_header) {
          return accumulated_buf.size() >= header_size;
        }
        else {
          return accumulated_buf.size() >= expected_size;
        }
      }
    }

    message_t reader_t::await_resume() {
      if (nread < 0) {
        co_handle = nullptr;
        auto error_status = nread;
        nread = 0;
        return { .status = error_status, .done = true };
      }

      co_handle = nullptr;
      if (is_raw_read) {
        if (!accumulated_buf.empty()) {
          auto [data_ptr, size] = accumulated_buf.get_contiguous();
          buffer_t buffer;

          if (size == accumulated_buf.size()) {
            buffer.assign(data_ptr, data_ptr + size);
            accumulated_buf.consume(size);
          }
          else {
            buffer.reserve(accumulated_buf.size());
            std::size_t total_size = accumulated_buf.size();
            buffer.resize(total_size);
            accumulated_buf.peek(buffer.data(), total_size);
            accumulated_buf.consume(total_size);
          }
          nread = 0;
          return { .buffer = std::move(buffer), .status = error_code::ok, .done = false };
        }
        nread = 0;
        return { .status = error_code::ok, .done = false };
      }
      else if (is_fixed_size_read) {
        if (accumulated_buf.size() >= expected_size) {
          buffer_t buffer(expected_size);
          accumulated_buf.peek(buffer.data(), expected_size);
          accumulated_buf.consume(expected_size);
          nread = 0;
          return { .buffer = std::move(buffer), .status = error_code::ok, .done = true };
        }
        nread = 0;
        return { .status = error_code::ok, .done = false };
      }
      else {
        if (reading_header && accumulated_buf.size() >= header_size) {
          accumulated_buf.peek(reinterpret_cast<char*>(&expected_size), header_size);
          accumulated_buf.consume(header_size);
          reading_header = false;
          bytes_read = 0;
        }

        if (!reading_header && accumulated_buf.size() >= expected_size) {
          buffer_t buffer(expected_size);
          accumulated_buf.peek(buffer.data(), expected_size);
          accumulated_buf.consume(expected_size);
          reading_header = true;
          expected_size = bytes_read = 0;
          nread = 0;
          return { .buffer = std::move(buffer), .status = error_code::ok, .done = true };
        }
        nread = 0;
        return { .status = error_code::ok, .done = false };
      }
    }

    reader_t::~reader_t() {
      if (stream) {
        uv_read_stop(stream.get());
      }
    }

    int writer_t::write(const buffer_t& user_data) {
      std::uint64_t data_size = user_data.size();
      buffer_t message_data;
      message_data.reserve(sizeof(std::uint64_t) + data_size);
      message_data.insert(
        message_data.end(), 
        reinterpret_cast<const char*>(&data_size), 
        reinterpret_cast<const char*>(&data_size) + sizeof(std::uint64_t)
      );
      message_data.insert(message_data.end(), user_data.begin(), user_data.end());
      return raw_writer.write(message_data);
    }

    client_handler_t::nr_t client_handler_t::add_client() {
      nr_t nr = client_list.NewNodeLast();
      client_list[nr] = new fan::network::tcp_t;
      client_list[nr]->nr = nr;
      return nr;
    }

    client_handler_t& get_client_handler() {
      static client_handler_t client_handler;
      return client_handler;
    }

    void tcp_t::tcp_deleter_t::operator()(void* p) const {
      uv_close(static_cast<uv_handle_t*>(p), [](uv_handle_t* req) {
        try {
          delete reinterpret_cast<uv_tcp_t*>(req);
        }
        catch (std::exception e) {
          fan::throw_error("failed to delete tcp:", e.what());
        }
      });
    }

    tcp_t::tcp_t() : socket(new uv_tcp_t, tcp_deleter_t{}) {
      client_handler_t::nr_t nr_val = get_client_handler().client_list.NewNodeLast();
      get_client_handler().client_list[nr_val] = this;
      get_client_handler().client_list[nr_val]->nr = nr_val;
      uv_tcp_init((uv_loop_t*)fan::event::get_loop(), socket.get());
    }

    tcp_t::~tcp_t() {
      if (nr.iic()) {
        return;
      }
      get_client_handler().remove_client(nr);
      nr.sic();
    }

    fan::event::error_code_t tcp_t::accept(tcp_t& client) noexcept {
      return uv_accept(reinterpret_cast<uv_stream_t*>(socket.get()),
        reinterpret_cast<uv_stream_t*>(client.socket.get()));
    }

    fan::event::error_code_t tcp_t::bind(const std::string& ip, int port) noexcept {
      struct sockaddr_in bind_addr;
      uv_ip4_addr(ip.c_str(), port, &bind_addr);
      return uv_tcp_bind(socket.get(), reinterpret_cast<sockaddr*>(&bind_addr), 0);
    }

    fan::event::task_t tcp_t::listen(const listen_address_t& address, listen_cb_t lambda, bool bind_) {
      if (bind_) {
        if (address.port == 0) {
          fan::throw_error("invalid port");
        }
        auto bind_result = bind(address.ip, address.port);
        if (bind_result != 0) {
          fan::throw_error("UDP bind failed:"_str + uv_strerror(bind_result));
        }
      }

      listener_t listener(*this, get_client_handler().amount_of_connections);
      std::list<fan::event::task_t> tasks;
      while (true) {
        if (co_await listener != 0) {
          continue;
        }
        auto client_id = get_client_handler().add_client();
        tcp_t& client = get_client_handler()[client_id];
        // simple cleanup
        for (auto it = tasks.begin(); it != tasks.end();) {
          if (it->owner->h.done()) {
            it = tasks.erase(it);
          }
          else {
            ++it;
          }
        }
        if (accept(client) == 0) {
          tasks.emplace_back([client_id, lambda]() -> fan::event::task_t {
            co_await lambda(get_client_handler()[client_id]);
          }());
        }
      }
      co_return;
    }

    reader_t& tcp_t::get_reader() const {
      if (!reader) {
        reader = std::make_unique<reader_t>(*this);
        if (int err = reader->start(); err != 0) {
          fan::throw_error("start failed with:" + fan::event::strerror(err));
        }
      }
      return *reader;
    }

    fan::event::task_t tcp_server_listen(listen_address_t address, tcp_t::listen_cb_t lambda) {
      tcp_t tcp;
      co_await tcp.listen(address, lambda, true);
    }

    fan::event::task_t tcp_listen(listen_address_t address, tcp_t::listen_cb_t lambda) {
      tcp_t tcp;
      co_await tcp.listen(address, lambda, false);
    }

    socket_address_t::socket_address_t() {
      std::memset(&addr, 0, sizeof(addr));
    }

    socket_address_t::socket_address_t(const std::string& ip, int port) {
      std::memset(&addr, 0, sizeof(addr));
      if (uv_ip4_addr(ip.c_str(), port, &addr.ipv4) == 0) {
      }
      else if (uv_ip6_addr(ip.c_str(), port, &addr.ipv6) == 0) {
      }
      else {
        fan::throw_error("Invalid IP address: " + ip);
      }
    }

    socket_address_t::socket_address_t(const struct sockaddr* sa) {
      std::memset(&addr, 0, sizeof(addr));
      if (sa->sa_family == AF_INET) {
        std::memcpy(&addr.ipv4, sa, sizeof(struct sockaddr_in));
      }
      else if (sa->sa_family == AF_INET6) {
        std::memcpy(&addr.ipv6, sa, sizeof(struct sockaddr_in6));
      }
    }

    std::string socket_address_t::get_ip() const {
      char ip_str[INET6_ADDRSTRLEN];
      if (addr.generic.sa_family == AF_INET) {
        uv_ip4_name(&addr.ipv4, ip_str, sizeof(ip_str));
      }
      else if (addr.generic.sa_family == AF_INET6) {
        uv_ip6_name(&addr.ipv6, ip_str, sizeof(ip_str));
      }
      else {
        return "unknown";
      }
      return std::string(ip_str);
    }

    int socket_address_t::get_port() const {
      if (addr.generic.sa_family == AF_INET) {
        return ntohs(addr.ipv4.sin_port);
      }
      else if (addr.generic.sa_family == AF_INET6) {
        return ntohs(addr.ipv6.sin6_port);
      }
      return 0;
    }

    const struct sockaddr* socket_address_t::sockaddr_ptr() const {
      return &addr.generic;
    }

    int socket_address_t::sockaddr_len() const {
      if (addr.generic.sa_family == AF_INET) {
        return sizeof(struct sockaddr_in);
      }
      else if (addr.generic.sa_family == AF_INET6) {
        return sizeof(struct sockaddr_in6);
      }
      return sizeof(struct sockaddr_storage);
    }

    udp_send_t::~udp_send_t() {
      if (data && data->status == 1) {
        data->req.cb = [](uv_udp_send_t* req, int) {
          delete static_cast<send_data_t*>(req->data);
        };
        data.release();
      }
    }

    int udp_recv_t::start() {
      if (receiving) return 0;
      receiving = true;
      socket->data = this;
      return uv_udp_recv_start(socket.get(),
        [](uv_handle_t* handle, std::size_t suggested_size, uv_buf_t* buf) {
          auto self = static_cast<udp_recv_t*>(handle->data);
          self->datagram.data.resize(2000);
          *buf = uv_buf_init(self->datagram.data.data(), 2000);
        },
        [](uv_udp_t* handle, ssize_t nread, const uv_buf_t* buf, const struct sockaddr* addr, unsigned flags) {
          auto self = static_cast<udp_recv_t*>(handle->data);
          self->datagram.status = nread;
          self->ready = true;
          if (nread > 0 && addr) {
            self->datagram.data.resize(nread);
            self->datagram.sender = socket_address_t(addr);
          }
          else if (nread <= 0) {
            self->datagram.data.clear();
          }
          if (self->co_handle) {
            self->co_handle();
          }
        }
      );
    }

    void udp_recv_t::stop() {
      if (receiving) {
        uv_udp_recv_stop(socket.get());
        receiving = false;
      }
    }

    udp_recv_t::~udp_recv_t() {
      stop();
    }

    void udp_recvfrom_t::set_expected_sender(const socket_address_t& sender) {
      expected_sender = sender;
      filter_sender = true;
    }

    void udp_recvfrom_t::set_expected_sender(const std::string& ip, int port) {
      expected_sender = socket_address_t(ip, port);
      filter_sender = true;
    }

    bool udp_recvfrom_t::matches_expected_sender(const socket_address_t& actual_sender) const {
      if (!filter_sender) return true;
      return (expected_sender.get_ip() == actual_sender.get_ip() &&
        expected_sender.get_port() == actual_sender.get_port());
    }

    int udp_recvfrom_t::start() {
      if (receiving) return 0;
      receiving = true;
      socket->data = this;

      return uv_udp_recv_start(socket.get(),
        [](uv_handle_t* handle, std::size_t suggested_size, uv_buf_t* buf) {
          auto self = static_cast<udp_recvfrom_t*>(handle->data);
          self->datagram.data.resize(2000);
          *buf = uv_buf_init(self->datagram.data.data(), 2000);
        },
        [](uv_udp_t* handle, ssize_t nread, const uv_buf_t* buf, const struct sockaddr* addr, unsigned flags) {
          auto self = static_cast<udp_recvfrom_t*>(handle->data);
          if (nread > 0 && addr) {
            socket_address_t sender_addr(addr);
            if (self->filter_sender && !self->matches_expected_sender(sender_addr)) {
              return;
            }
            self->datagram.status = nread;
            self->datagram.data.resize(nread);
            self->datagram.sender = sender_addr;
          }
          else {
            self->datagram.status = nread;
            self->datagram.data.clear();
          }
          self->ready = true;
          if (self->co_handle) {
            self->co_handle();
          }
        }
      );
    }

    void udp_recvfrom_t::stop() {
      if (receiving) {
        uv_udp_recv_stop(socket.get());
        receiving = false;
      }
    }

    void udp_recvfrom_t::await_suspend(std::coroutine_handle<> h) {
      co_handle = h;
      if (int err = start(); err != 0) {
        fan::throw_error("start failed with:" + fan::event::strerror(err));
      }
    }

    udp_datagram_t udp_recvfrom_t::await_resume() {
      ready = false;
      co_handle = nullptr;
      return std::move(datagram);
    }

    udp_recvfrom_t::~udp_recvfrom_t() {
      stop();
    }

    void udp_t::udp_deleter_t::operator()(void* p) const {
      uv_close(static_cast<uv_handle_t*>(p), [](uv_handle_t* req) {
        delete reinterpret_cast<uv_udp_t*>(req);
      });
    }

    udp_t::udp_t() : socket(new uv_udp_t, udp_deleter_t{}) {
      int result = uv_udp_init((uv_loop_t*)fan::event::get_loop(), socket.get());
      if (result != 0) {
        fan::throw_error("Failed to initialize UDP socket:"_str + uv_strerror(result));
      }
    }

    fan::event::error_code_t udp_t::bind(const std::string& ip, int port, unsigned int flags) noexcept {
      socket_address_t addr(ip, port);
      return uv_udp_bind(socket.get(), addr.sockaddr_ptr(), flags);
    }

    fan::event::task_t udp_t::listen(const listen_address_t& address, recv_cb_t callback, bool bind_) {
      std::string ip = address.ip;
      int port = address.port;
      if (port == 0) {
        fan::throw_error("invalid port");
      }

      if (bind_) {
        auto bind_result = bind(ip, port);
        if (bind_result != 0) {
          fan::throw_error("UDP bind failed:"_str + uv_strerror(bind_result));
        }
      }

      while (true) {
        auto datagram = co_await recvfrom(ip, port);
        if (datagram.is_error()) {
          if (datagram.status == UV_EOF) {
            break;
          }
          continue;
        }

        if (!datagram.is_empty()) {
          co_await callback(*this, datagram);
        }
      }
      co_return;
    }

    socket_address_t udp_t::get_sockname() const {
      struct sockaddr_storage addr;
      int namelen = sizeof(addr);
      int result = uv_udp_getsockname(socket.get(), reinterpret_cast<struct sockaddr*>(&addr), &namelen);
      if (result != 0) {
        return socket_address_t();
      }
      return socket_address_t(reinterpret_cast<struct sockaddr*>(&addr));
    }

    fan::event::error_code_t udp_t::set_broadcast(bool enable) noexcept {
      return uv_udp_set_broadcast(socket.get(), enable ? 1 : 0);
    }
    fan::event::error_code_t udp_t::set_ttl(int ttl) noexcept {
      return uv_udp_set_ttl(socket.get(), ttl);
    }
    fan::event::error_code_t udp_t::join_multicast(const std::string& multicast_addr, const std::string& interface_addr) noexcept {
      return uv_udp_set_membership(socket.get(), multicast_addr.c_str(),
        interface_addr.empty() ? nullptr : interface_addr.c_str(),
        UV_JOIN_GROUP);
    }
    fan::event::error_code_t udp_t::leave_multicast(const std::string& multicast_addr, const std::string& interface_addr) noexcept {
      return uv_udp_set_membership(socket.get(), multicast_addr.c_str(),
        interface_addr.empty() ? nullptr : interface_addr.c_str(),
        UV_LEAVE_GROUP);
    }
    fan::event::error_code_t udp_t::set_multicast_ttl(int ttl) noexcept {
      return uv_udp_set_multicast_ttl(socket.get(), ttl);
    }
    fan::event::error_code_t udp_t::set_multicast_interface(const std::string& interface_addr) noexcept {
      return uv_udp_set_multicast_interface(socket.get(), interface_addr.c_str());
    }
    fan::event::error_code_t udp_t::set_multicast_loop(bool enable) noexcept {
      return uv_udp_set_multicast_loop(socket.get(), enable ? 1 : 0);
    }

    fan::event::task_t udp_listen(const listen_address_t& address, udp_t::recv_cb_t callback) {
      udp_t udp;
      co_await udp.listen(address, callback);
    }

#if defined(FAN_NETWORK_HTTP_ENABLED)
    namespace http {

      response_t& response_t::status(int code) {
        status_code = code;
        return *this;
      }

      response_t& response_t::header(const std::string& key, const std::string& value) {
        headers[key] = value;
        return *this;
      }

      response_t& response_t::json(const fan::json& data) {
        body = data.dump();
        headers["Content-Type"] = "application/json";
        return *this;
      }

      response_t& response_t::text(const std::string& data) {
        body = data;
        headers["Content-Type"] = "text/plain";
        return *this;
      }

      response_t& response_t::html(const std::string& data) {
        body = data;
        headers["Content-Type"] = "text/html";
        return *this;
      }

      response_t& response_t::ok(const fan::json& data) {
        status_code = status_t::ok;
        if (!data.is_null()) {
          json(data);
        }
        return *this;
      }

      response_t& response_t::created(const fan::json& data) {
        status_code = status_t::created;
        if (!data.is_null()) {
          json(data);
        }
        return *this;
      }

      response_t& response_t::not_found(const std::string& message) {
        status_code = status_t::not_found;
        return text(message);
      }

      response_t& response_t::bad_request(const std::string& message) {
        status_code = status_t::bad_request;
        return text(message);
      }

      response_t& response_t::internal_error(const std::string& message) {
        status_code = status_t::internal_server_error;
        return text(message);
      }

      response_t& response_t::error(const error_t& err) {
        switch (err.code) {
        case error_t::invalid_param:
        case error_t::invalid_json:
        case error_t::validation_error:
          return bad_request(err.message);
        case error_t::not_found_error:
          return not_found(err.message);
        case error_t::database_error:
        case error_t::connection_failed:
        case error_t::timeout:
        default:
          return internal_error(err.message);
        }
      }

      std::string response_t::to_string() const {
        std::string response = "HTTP/1.1 " + std::to_string(status_code) + " ";
        switch (status_code) {
        case status_t::ok: response += "OK"; break;
        case status_t::created: response += "Created"; break;
        case status_t::bad_request: response += "Bad Request"; break;
        case status_t::not_found: response += "Not Found"; break;
        case status_t::internal_server_error: response += "Internal Server Error"; break;
        default: response += "Unknown"; break;
        }
        response += "\r\n";
        for (const auto& [key, value] : headers) {
          response += key + ": " + value + "\r\n";
        }

        if (!body.empty()) {
          response += "Content-Length: " + std::to_string(body.length()) + "\r\n";
        }

        response += "\r\n" + body;
        return response;
      }

      bool route_t::matches(int req_method, const std::string& path, std::unordered_map<std::string, std::string>& params) const {
        if (req_method != method) return false;
        auto pattern_parts = split_path(pattern);
        auto path_parts = split_path(path);

        if (pattern_parts.size() != path_parts.size()) return false;

        params.clear();
        for (std::size_t i = 0; i < pattern_parts.size(); ++i) {
          if (pattern_parts[i].starts_with('{') && pattern_parts[i].ends_with('}')) {
            std::string param_name = pattern_parts[i].substr(1, pattern_parts[i].length() - 2);
            params[param_name] = path_parts[i];
          }
          else if (pattern_parts[i].starts_with('{') && pattern_parts[i].find('}') != std::string::npos) {
            auto end_brace = pattern_parts[i].find('}');
            std::string param_name = pattern_parts[i].substr(1, end_brace - 1);
            std::string suffix = pattern_parts[i].substr(end_brace + 1);
            std::string value = path_parts[i];
            if (!suffix.empty()) {
              if (!value.ends_with(suffix)) return false;
              value.erase(value.size() - suffix.size());
            }
            params[param_name] = value;
          }
          else if (pattern_parts[i] != path_parts[i]) {
            return false;
          }
        }
        return true;
      }

      std::vector<std::string> route_t::split_path(const std::string& path) const {
        std::vector<std::string> parts;
        std::string current;
        for (char c : path) {
          if (c == '/') {
            if (!current.empty()) {
              parts.push_back(current);
              current.clear();
            }
          }
          else {
            current += c;
          }
        }
        if (!current.empty()) {
          parts.push_back(current);
        }
        return parts;
      }

      fan::event::runv_t<response_t> router_t::handle(const request_t& req) {
        for (const auto& route : routes) {
          request_t modified_req = req;
          if (route.matches(req.method, req.path, modified_req.params)) {
            response_t res;
            try {
              co_await route.handler(modified_req, res);
            }
            catch (const std::exception& e) {
              res.internal_error("Handler exception: " + std::string(e.what()));
            }
            co_return res;
          }
        }

        response_t res;
        co_return res.not_found();
      }

      fan::event::task_t server_t::listen(const fan::network::listen_address_t& address) {
        co_await fan::network::tcp_server_listen(address, [this](fan::network::tcp_t& client) -> fan::event::task_t {
          bool keep_alive = true;

          while (keep_alive) {
            std::string request_data;
            bool headers_complete = false;
            std::size_t content_length = 0;
            response_t error_response;
            bool has_error = false;

            try {
              while (!headers_complete) {
                auto msg = co_await client.read_raw();
                if (msg.status < 0) {
                  co_return;
                }

                std::string chunk(msg.buffer.begin(), msg.buffer.end());
                request_data += chunk;

                std::size_t header_end = request_data.find("\r\n\r\n");
                if (header_end != std::string::npos) {
                  headers_complete = true;

                  std::size_t cl_pos = request_data.find("Content-Length:");
                  if (cl_pos != std::string::npos) {
                    std::size_t cl_start = cl_pos + 15;
                    std::size_t cl_end = request_data.find("\r\n", cl_start);
                    std::string cl_str = request_data.substr(cl_start, cl_end - cl_start);
                    cl_str.erase(0, cl_str.find_first_not_of(" \t"));
                    content_length = std::stoull(cl_str);
                  }

                  std::size_t current_body_length = request_data.length() - (header_end + 4);
                  while (current_body_length < content_length) {
                    auto body_msg = co_await client.read_raw();
                    if (body_msg.status < 0) {
                      co_return;
                    }
                    std::string body_chunk(body_msg.buffer.begin(), body_msg.buffer.end());
                    request_data += body_chunk;
                    current_body_length += body_chunk.length();
                  }
                }
              }

              request_t request = parse_request(request_data);
              auto conn_header = request.header("Connection");
              if (conn_header.find("close") != std::string::npos) {
                keep_alive = false;
              }

              response_t response = co_await router.handle(request);
              if (keep_alive) {
                response.header("Connection", "keep-alive");
              }
              else {
                response.header("Connection", "close");
              }

              std::string response_str = response.to_string();
              co_await client.write_raw(response_str);

            }
            catch (const std::exception& e) {
              error_response.internal_error();
              error_response.header("Connection", "close");
              has_error = true;
              keep_alive = false;
            }

            if (has_error) {
              std::string response_str = error_response.to_string();
              co_await client.write_raw(response_str);
            }
          }
          co_return;
        });
      }

#ifdef FAN_NETWORK_CURL_ENABLED
      async_request_t::async_request_t(const std::string& u, const config_t& cfg)
        : url(u), config(cfg) {
        easy_handle = curl_easy_init();
        if (!easy_handle) {
          error_message = "curl_easy_init failed";
          return;
        }
        curl_easy_setopt(easy_handle, CURLOPT_URL, url.c_str());
        curl_easy_setopt(easy_handle, CURLOPT_WRITEFUNCTION, &async_request_t::write_cb);
        curl_easy_setopt(easy_handle, CURLOPT_WRITEDATA, this);
        curl_easy_setopt(easy_handle, CURLOPT_FOLLOWLOCATION, config.follow_redirects ? 1L : 0L);
        curl_easy_setopt(easy_handle, CURLOPT_SSL_VERIFYPEER, config.verify_ssl ? 1L : 0L);
        curl_easy_setopt(easy_handle, CURLOPT_SSL_VERIFYHOST, config.verify_ssl ? 2L : 0L);
        curl_easy_setopt(easy_handle, CURLOPT_TIMEOUT, config.timeout_seconds);
        curl_easy_setopt(easy_handle, CURLOPT_USERAGENT, config.user_agent.c_str());
        curl_easy_setopt(easy_handle, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(easy_handle, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_1_1);
        curl_easy_setopt(easy_handle, CURLOPT_CAINFO, "curl-ca-bundle.crt");
      }

      async_request_t::~async_request_t() {
        if (curl_headers) {
          curl_slist_free_all(curl_headers);
          curl_headers = nullptr;
        }
        if (easy_handle) {
          curl_easy_cleanup(easy_handle);
          easy_handle = nullptr;
        }
      }

      void async_request_t::await_suspend(std::coroutine_handle<> h) {
        awaiting = h;
        async_context_t::instance().add_request(shared_from_this());
      }

      std::expected<response_t, std::string> async_request_t::await_resume() {
        if (!error_message.empty()) {
          return std::unexpected(error_message);
        }
        return std::move(response);
      }

      std::size_t async_request_t::write_cb(char* ptr, std::size_t size, std::size_t nmemb, void* userdata) {
        auto* self = static_cast<async_request_t*>(userdata);
        std::size_t total = size * nmemb;
        if (total) {
          self->response.body.append(ptr, total);
        }
        return total;
      }

      async_context_t::async_context_t() {
        curl_global_init(CURL_GLOBAL_DEFAULT);
        multi_handle = curl_multi_init();
        if (multi_handle) {
          curl_multi_setopt(multi_handle, CURLMOPT_SOCKETFUNCTION, &async_context_t::socket_cb);
          curl_multi_setopt(multi_handle, CURLMOPT_SOCKETDATA, this);
          curl_multi_setopt(multi_handle, CURLMOPT_TIMERFUNCTION, &async_context_t::timer_cb);
          curl_multi_setopt(multi_handle, CURLMOPT_TIMERDATA, this);
        }
        uv_loop_t* loop = (uv_loop_t*)fan::event::get_loop();
        uv_timer_init(loop, &timeout_timer);
        timeout_timer.data = this;
      }

      async_context_t::~async_context_t() {
        for (auto& kv : polls) {
          auto* c = kv.second;
          if (c) {
            uv_poll_stop(&c->poll);
            uv_close(reinterpret_cast<uv_handle_t*>(&c->poll), [](uv_handle_t* h) {
              auto* cctx = reinterpret_cast<sock_ctx*>(h->data);
              delete cctx;
            });
          }
        }
        polls.clear();
        if (timeout_timer_active) {
          uv_timer_stop(&timeout_timer);
          timeout_timer_active = false;
        }
        uv_close(reinterpret_cast<uv_handle_t*>(&timeout_timer), nullptr);
        if (multi_handle) {
          curl_multi_cleanup(multi_handle);
          multi_handle = nullptr;
        }
        curl_global_cleanup();
      }

      async_context_t& async_context_t::instance() {
        static async_context_t ctx;
        return ctx;
      }

      int async_context_t::socket_cb(CURL*, curl_socket_t s, int what, void* userp, void* socketp) {
        auto* ctx = static_cast<async_context_t*>(userp);
        auto* cctx = static_cast<sock_ctx*>(socketp);
        switch (what) {
        case CURL_POLL_IN:
        case CURL_POLL_OUT:
        case CURL_POLL_INOUT:
        {
          if (!cctx) {
            cctx = ctx->create_sock_ctx(s);
            if (!cctx) {
              curl_multi_assign(ctx->multi_handle, s, nullptr);
              return 0;
            }
            curl_multi_assign(ctx->multi_handle, s, cctx);
          }
          int uv_events = 0;
          if (what != CURL_POLL_OUT) {
            uv_events |= UV_READABLE;
          }
          if (what != CURL_POLL_IN) {
            uv_events |= UV_WRITABLE;
          }
          uv_poll_start(&cctx->poll, uv_events, &async_context_t::poll_cb);
          break;
        }
        case CURL_POLL_REMOVE:
        {
          if (cctx) {
            ctx->destroy_sock_ctx(cctx);
            curl_multi_assign(ctx->multi_handle, s, nullptr);
          }
          break;
        }
        default: break;
        }
        return 0;
      }

      int async_context_t::timer_cb(CURLM*, long timeout_ms, void* userp) {
        auto* ctx = static_cast<async_context_t*>(userp);
        if (timeout_ms < 0) {
          if (ctx->timeout_timer_active) {
            uv_timer_stop(&ctx->timeout_timer);
            ctx->timeout_timer_active = false;
          }
          return 0;
        }
        if (!ctx->timeout_timer_active) {
          ctx->timeout_timer.data = ctx;
          ctx->timeout_timer_active = true;
        }
        if (timeout_ms == 0) {
          timeout_ms = 1;
        }
        uv_timer_start(&ctx->timeout_timer, &async_context_t::timeout_cb, static_cast<std::uint64_t>(timeout_ms), 0);
        return 0;
      }

      void async_context_t::timeout_cb(uv_timer_t* handle) {
        auto* ctx = static_cast<async_context_t*>(handle->data);
        ctx->timeout_timer_active = false;
        int still_running = 0;
        curl_multi_socket_action(ctx->multi_handle, CURL_SOCKET_TIMEOUT, 0, &still_running);
        ctx->drain_multi();
      }

      void async_context_t::poll_cb(uv_poll_t* req, int status, int events) {
        auto* cctx = static_cast<sock_ctx*>(req->data);
        auto* ctx = cctx->ctx;
        int flags = 0;
        if (events & UV_READABLE) {
          flags |= CURL_CSELECT_IN;
        }
        if (events & UV_WRITABLE) {
          flags |= CURL_CSELECT_OUT;
        }
        if (status < 0) {
          flags |= CURL_CSELECT_ERR;
        }
        int still_running = 0;
        curl_multi_socket_action(ctx->multi_handle, cctx->sock, flags, &still_running);
        ctx->drain_multi();
      }

      async_context_t::sock_ctx* async_context_t::create_sock_ctx(curl_socket_t s) {
        auto* c = new sock_ctx {};
        c->sock = s;
        c->ctx = this;
        int rc = uv_poll_init_socket((uv_loop_t*)fan::event::get_loop(), &c->poll, s);
        if (rc != 0) {
          delete c;
          return nullptr;
        }
        c->poll.data = c;
        polls[s] = c;
        return c;
      }

      void async_context_t::destroy_sock_ctx(sock_ctx* c) {
        if (!c) {
          return;
        }
        polls.erase(c->sock);
        uv_poll_stop(&c->poll);
        uv_close(reinterpret_cast<uv_handle_t*>(&c->poll), [](uv_handle_t* h) {
          auto* cctx = reinterpret_cast<sock_ctx*>(h->data);
          delete cctx;
        });
      }

      void async_context_t::add_request(const std::shared_ptr<async_request_t>& req) {
        if (!req || !req->easy_handle) {
          return;
        }
        for (const auto& kv : req->headers_map) {
          std::string h = kv.first + ": " + kv.second;
          req->curl_headers = curl_slist_append(req->curl_headers, h.c_str());
        }
        if (req->curl_headers) {
          curl_easy_setopt(req->easy_handle, CURLOPT_HTTPHEADER, req->curl_headers);
        }
        curl_easy_setopt(req->easy_handle, CURLOPT_PRIVATE, req.get());
        CURLMcode rc = curl_multi_add_handle(multi_handle, req->easy_handle);
        if (rc != CURLM_OK) {
          req->error_message = curl_multi_strerror(rc);
          if (req->awaiting) {
            auto h = req->awaiting;
            req->awaiting = {};
            h.resume();
          }
          return;
        }
        active_requests.push_back(req);
        int still_running = 0;
        curl_multi_socket_action(multi_handle, CURL_SOCKET_TIMEOUT, 0, &still_running);
        drain_multi();
      }

      void async_context_t::drain_multi() {
        CURLMsg* msg = nullptr;
        int msgs_left = 0;
        while ((msg = curl_multi_info_read(multi_handle, &msgs_left))) {
          if (msg->msg != CURLMSG_DONE) {
            continue;
          }
          CURL* easy = msg->easy_handle;
          async_request_t* raw_req = nullptr;
          curl_easy_getinfo(easy, CURLINFO_PRIVATE, &raw_req);

          auto it = std::find_if(active_requests.begin(), active_requests.end(), [raw_req](const std::shared_ptr<async_request_t>& p) {
            return p.get() == raw_req;
          });
          if (it == active_requests.end()) {
            curl_multi_remove_handle(multi_handle, easy);
            continue;
          }

          auto sp = *it;
          active_requests.erase(it);
          long code = 0;
          curl_easy_getinfo(easy, CURLINFO_RESPONSE_CODE, &code);
          sp->response.status_code = static_cast<int>(code);
          sp->completed.store(true);
          curl_multi_remove_handle(multi_handle, easy);
          if (msg->data.result != CURLE_OK && sp->error_message.empty()) {
            sp->error_message = curl_easy_strerror(msg->data.result);
          }

          if (sp->awaiting) {
            auto h = sp->awaiting;
            sp->awaiting = {};
            h.resume();
          }
        }
      }

      fan::event::runv_t<std::expected<response_t, std::string>>
        get(const std::string& url, const config_t& cfg) {
        co_return co_await *std::make_shared<async_request_t>(url, cfg);
      }

      fan::event::runv_t<std::expected<response_t, std::string>>
        post(const std::string& url,
          const std::string& body,
          const std::unordered_map<std::string, std::string>& headers,
          const config_t& cfg
        ) {
        auto req = std::make_shared<async_request_t>(url, cfg);
        req->headers_map = headers;

        curl_easy_setopt(req->easy_handle, CURLOPT_POST, 1L);
        curl_easy_setopt(req->easy_handle, CURLOPT_POSTFIELDS, body.c_str());
        curl_easy_setopt(req->easy_handle, CURLOPT_POSTFIELDSIZE, body.size());

        co_return co_await *req;
      }

      fan::event::runv_t<std::expected<response_t, std::string>>
        client_t::get(const std::string& path) {
        co_return co_await http::get(base_url + path, config);
      }

      fan::event::runv_t<std::expected<response_t, std::string>>
        client_t::post(const std::string& path, const fan::json& body) {
        co_return co_await http::post(base_url + path, body.dump(),
          {{"Content-Type", "application/json"}}, config);
      }

      fan::event::runv_t<std::expected<response_t, std::string>>
        client_t::post(const std::string& path, const std::string& body,
          const std::unordered_map<std::string, std::string>& headers) {
        co_return co_await http::post(base_url + path, body, headers, config);
      }

      fan::event::runv_t<std::expected<response_t, std::string>>
        client_t::put(const std::string& path, const fan::json& body) {
        auto req = std::make_shared<async_request_t>(base_url + path, config);
        req->headers_map = {{"Content-Type", "application /json"}};
        std::string body_str = body.dump();
        curl_easy_setopt(req->easy_handle, CURLOPT_CUSTOMREQUEST, "PUT");
        curl_easy_setopt(req->easy_handle, CURLOPT_POSTFIELDS, body_str.c_str());
        curl_easy_setopt(req->easy_handle, CURLOPT_POSTFIELDSIZE, body_str.size());
        co_return co_await *req;
      }

      fan::event::runv_t<std::expected<response_t, std::string>>
        client_t::delete_(const std::string& path) {
        auto req = std::make_shared<async_request_t>(base_url + path, config);
        curl_easy_setopt(req->easy_handle, CURLOPT_CUSTOMREQUEST, "DELETE");
        co_return co_await *req;
      }
#endif
    } // namespace http
#endif

  } // namespace network
}
#endif