module;

#include <uv.h>
#include <stdexcept>
#include <memory>
#include <coroutine>
#undef min
#undef max

export module fan:network;

import :print;
import :event;

export namespace fan {
  namespace network {
    struct tcp_t;
    struct getaddrinfo_t {
      struct getaddrinfo_data_t {
        uv_getaddrinfo_t getaddrinfo_handle;
        std::coroutine_handle<> co_handle;
        bool ready{ false };
        int status;
      };
      std::unique_ptr<getaddrinfo_data_t> data;

      getaddrinfo_t(const char* node, const char* service, struct addrinfo* hints = nullptr) :
        data(std::make_unique<getaddrinfo_data_t>()) {
        data->getaddrinfo_handle.data = data.get();
        uv_getaddrinfo(fan::event::event_loop, &data->getaddrinfo_handle, [](uv_getaddrinfo_t* getaddrinfo_handle, int status, struct addrinfo* res) {
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

      bool await_ready() const { return data->ready; }
      void await_suspend(std::coroutine_handle<> h) { data->co_handle = h; }
      int await_resume() const { return data->status; };

      ~getaddrinfo_t() {
        if (data) {
          uv_freeaddrinfo(data->getaddrinfo_handle.addrinfo);
          if (uv_cancel(reinterpret_cast<uv_req_t*>(&data->getaddrinfo_handle)) == 0) {
            data.release();
          }
        }
      }
    };

    struct connector_t {
      struct connector_data_t {
        uv_connect_t req;
        std::coroutine_handle<> co_handle;
        int status;
      };

      std::unique_ptr<connector_data_t> data;

      template <typename T>
      requires (std::is_same_v<T, tcp_t>)
      connector_t(const T& tcp, const char* ip, int port) :
        data{ std::make_unique<connector_data_t>() }
      {
        data->req.data = data.get();
        struct sockaddr_in client_addr;
        auto result = uv_ip4_addr(ip, port, &client_addr);

        if (result != 0) {
          fan::throw_error("failed to resolve address");
        }

        data->status = uv_tcp_connect(&data->req, tcp.socket.get(), reinterpret_cast<sockaddr*>(&client_addr),
          [](uv_connect_t* req, int status) {
            auto* data = static_cast<connector_data_t*>(req->data);
            data->status = status;
            [[likely]] if (data->co_handle) {
              data->co_handle();
            }
        });
        if (data->status == 0) {
          data->status = 1;
        }
      }

      template <typename T>
      requires (std::is_same_v<T, tcp_t>)
      connector_t(const T& tcp, const getaddrinfo_t& info) : data(std::make_unique<connector_data_t>()) {
        data->req.data = data.get();
        data->status = uv_tcp_connect(
          &data->req, 
          tcp.socket.get(), 
          (const struct sockaddr*)info.data->getaddrinfo_handle.addrinfo->ai_addr,
          [](uv_connect_t* req, int status) {
            auto* data = static_cast<connector_data_t*>(req->data);
            data->status = status;
            [[likely]] if (data->co_handle) {
              data->co_handle();
            }
          }
        );

        if (data->status == 0) {
          data->status = 1;
        }
      }

      connector_t(connector_t&&) = default;
      connector_t& operator=(connector_t&&) = default;
      ~connector_t() {
        [[unlikely]] if (data && data->status == 1) {
          data->req.cb = [](uv_connect_t* req, int) {
            delete static_cast<connector_data_t*>(req->data);
            };
          data.release();
        }
      }

      bool await_ready() { return data->status <= 0; }
      void await_suspend(std::coroutine_handle<> h) { data->co_handle = h; }
      int await_resume() {
        if (data->status != 0) {
          throw std::runtime_error(std::string("connection failed with") + uv_strerror(data->status));
        }
        return data->status;
      }
    };

    struct listener_t {
      std::shared_ptr<uv_stream_t> stream;
      std::coroutine_handle<> co_handle;
      int status;
      bool ready;


      template <typename T>
      requires (std::is_same_v<T, tcp_t>)
      listener_t(const T& tcp, int backlog) :
        stream{ std::reinterpret_pointer_cast<uv_stream_t>(tcp.socket) },
        ready{ false }
      {
        stream->data = this;
        auto r = uv_listen(stream.get(), backlog, [](uv_stream_t* req, int status) {
          auto self = static_cast<listener_t*>(req->data);
          self->status = status;
          self->ready = true;
          if (self->co_handle)
            self->co_handle();
          });
        if (r != 0) {
          fan::throw_error("listen error", r);
        }
      }

      listener_t(listener_t&& l) :
        stream(std::move(l.stream)),
        co_handle(std::move(l.co_handle)),
        status(l.status),
        ready(l.ready) {
        stream->data = this;
      }

      bool await_ready() const { return ready; }
      void await_suspend(std::coroutine_handle<> h) { co_handle = h; }
      int await_resume() {
        ready = false;
        co_handle = nullptr;
        return status;
      }
    };

    struct raw_reader_t {
      std::shared_ptr<uv_stream_t> stream;
      std::string buf;
      ssize_t nread{ 0 };
      std::coroutine_handle<> co_handle;

      template <typename T>
      requires (std::is_same_v<T, tcp_t>)
      raw_reader_t(const T& tcp) : stream{ std::reinterpret_pointer_cast<uv_stream_t>(tcp.socket) } {
        stream->data = this;
      }
      raw_reader_t(const raw_reader_t&) = delete;
      raw_reader_t(raw_reader_t&& r) noexcept :
        stream{ std::move(r.stream) },
        buf{ std::move(r.buf) },
        nread{ std::move(r.nread) },
        co_handle{ std::move(r.co_handle) } {
        stream->data = this;
      }

      void stop() {
        uv_read_stop(stream.get());
      }
      int start() noexcept {
        return uv_read_start(stream.get(),
          [](uv_handle_t* handle, size_t suggested_size, uv_buf_t* buf) {
            auto self = static_cast<raw_reader_t*>(handle->data);
            self->buf.resize(suggested_size);
            *buf = uv_buf_init(self->buf.data(), suggested_size);
          },
          [](uv_stream_t* req, ssize_t nread, const uv_buf_t* buf) {
            auto self = static_cast<raw_reader_t*>(req->data);
            self->nread = nread;
            [[likely]] if (self->co_handle)
              self->co_handle();
          }
        );
      }
      bool await_ready() const { return nread > 0; }
      void await_suspend(std::coroutine_handle<> h) { co_handle = h; }
      const char* await_resume() {
        const char* out = nullptr;
        if (nread > 0) {
          out = buf.c_str();
          nread = 0;
        }
        co_handle = nullptr;
        return out;
      }

      ~raw_reader_t() {
        if (stream) {
          uv_read_stop(stream.get());
        }
        buf.clear();
      }
    };

    struct raw_writer_t {
      struct writer_data_t {
        std::shared_ptr<uv_stream_t> stream;
        uv_write_t write_handle;
        std::string to_write;
        std::coroutine_handle<> co_handle;
        int status{ 0 };
      };

      std::unique_ptr<writer_data_t> data;

      template <typename T>
        requires (std::is_same_v<T, tcp_t>)
      raw_writer_t(const T& tcp) :
        data(new writer_data_t{ std::reinterpret_pointer_cast<uv_stream_t>(tcp.socket) }) {
        data->write_handle.data = data.get();
      }
      raw_writer_t(raw_writer_t&&) = default;
      raw_writer_t& operator=(raw_writer_t&&) = default;

      int write(const std::string& some_data) {
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
          fan::throw_error("tcp write failed", r);
        }
        return r;
      }

      bool await_ready() const noexcept { return data->status <= 0; }
      void await_suspend(std::coroutine_handle<> h) noexcept { data->co_handle = h; }
      int await_resume() noexcept {
        data->co_handle = nullptr;
        return data->status;
      }

      ~raw_writer_t() {
        if (data && data->status == 1)
        {
          data->write_handle.cb = [](uv_write_t* write_handle, int) {
            delete static_cast<writer_data_t*>(write_handle->data);
            };
          data.release();
        }
      }
    };

    struct message_t {
      bool done;
      std::string buffer;
    };

    struct reader_t {
      std::shared_ptr<uv_stream_t> stream;
      std::string accumulated_buf;
      std::string temp_buf;
      ssize_t nread{ 0 };
      std::coroutine_handle<> co_handle;

      uint64_t expected_size{ 0 };
      uint64_t bytes_read{ 0 };
      bool reading_header{ true };
      static constexpr size_t header_size = sizeof(uint64_t);

      template <typename T>
      requires (std::is_same_v<T, tcp_t>)
      reader_t(const T& tcp) : stream{ std::reinterpret_pointer_cast<uv_stream_t>(tcp.socket) } {
        stream->data = this;
      }

      reader_t(const reader_t&) = delete;
      reader_t(reader_t&& r) noexcept :
        stream{ std::move(r.stream) },
        accumulated_buf{ std::move(r.accumulated_buf) },
        temp_buf{ std::move(r.temp_buf) },
        nread{ std::move(r.nread) },
        co_handle{ std::move(r.co_handle) },
        expected_size{ r.expected_size },
        bytes_read{ r.bytes_read },
        reading_header{ r.reading_header } {
        stream->data = this;
      }

      void stop() {
        uv_read_stop(stream.get());
      }

      int start() noexcept {
        return uv_read_start(stream.get(),
          [](uv_handle_t* handle, size_t suggested_size, uv_buf_t* buf) {
            auto self = static_cast<reader_t*>(handle->data);
            self->temp_buf.resize(suggested_size);
            *buf = uv_buf_init(self->temp_buf.data(), suggested_size);
          },
          [](uv_stream_t* req, ssize_t nread, const uv_buf_t* buf) {
            auto self = static_cast<reader_t*>(req->data);
            self->nread = nread;
            if (nread > 0) {
              self->accumulated_buf.append(self->temp_buf.data(), nread);
            }
            [[likely]] if (self->co_handle)
              self->co_handle();
          }
        );
      }

      bool await_ready() const {
        if (nread < 0) return true;

        if (reading_header) {
          return accumulated_buf.size() >= header_size;
        }
        else {
          return accumulated_buf.size() >= expected_size;
        }
      }

      void await_suspend(std::coroutine_handle<> h) {
        co_handle = h;
      }

      message_t* await_resume() {
        static message_t msg;
        msg.done = false;
        msg.buffer.clear();

        if (nread < 0) {
          msg.done = true;
          co_handle = nullptr;
          return &msg;
        }

        if (reading_header && accumulated_buf.size() >= header_size) {
          std::memcpy(&expected_size, accumulated_buf.data(), header_size);

          accumulated_buf.erase(0, header_size);
          reading_header = false;
          bytes_read = 0;

          if (accumulated_buf.size() < expected_size) {
            co_handle = nullptr;
            nread = 0;
            return nullptr;
          }
        }

        if (!reading_header && accumulated_buf.size() >= expected_size) {
          msg.buffer = accumulated_buf.substr(0, expected_size);
          msg.done = true;

          accumulated_buf.erase(0, expected_size);

          reading_header = true;
          expected_size = 0;
          bytes_read = 0;
        }

        co_handle = nullptr;
        nread = 0;
        return msg.done ? &msg : nullptr;
      }

      ~reader_t() {
        if (stream) {
          uv_read_stop(stream.get());
        }
        accumulated_buf.clear();
        temp_buf.clear();
      }
    };

    struct writer_t {
      raw_writer_t raw_writer;

      template <typename T>
        requires (std::is_same_v<T, tcp_t>)
      writer_t(const T& tcp) : raw_writer(tcp) {}

      writer_t(const writer_t&) = delete;
      writer_t& operator=(const writer_t&) = delete;
      writer_t(writer_t&&) = default;
      writer_t& operator=(writer_t&&) = default;

      int write(const std::string& user_data) {
        uint64_t data_size = user_data.size();
        std::string message_data;
        message_data.reserve(sizeof(uint64_t) + data_size);

        message_data.append(reinterpret_cast<const char*>(&data_size), sizeof(uint64_t));

        message_data.append(user_data);

        return raw_writer.write(message_data);
      }

      int write(const message_t& msg) {
        return write(msg.buffer);
      }

      bool await_ready() const noexcept {
        return raw_writer.await_ready();
      }

      void await_suspend(std::coroutine_handle<> h) noexcept {
        raw_writer.await_suspend(h);
      }

      int await_resume() noexcept {
        return raw_writer.await_resume();
      }
    };

    struct listen_address_t {
      std::string ip = "0.0.0.0";
      uint16_t port = 0;
    };

    struct tcp_t;

    using listen_cb_t = std::function<fan::event::task_t(tcp_t&&)>;

    struct tcp_t {
      std::shared_ptr<uv_tcp_t> socket;
      reader_t reader;

      struct tcp_deleter_t {
        void operator()(void* p) const {
          uv_close(static_cast<uv_handle_t*>(p), [](uv_handle_t* req) {
            delete reinterpret_cast<uv_tcp_t*>(req);
          });
        }
      };

      tcp_t() : socket(new uv_tcp_t, tcp_deleter_t{}), reader(*this) {
        uv_tcp_init(fan::event::event_loop, socket.get());
      }
      tcp_t(const tcp_t&) = delete;
      tcp_t& operator=(const tcp_t&) = delete;
      tcp_t(tcp_t&&) = default;
      tcp_t& operator=(tcp_t&&) = delete;

      fan::event::error_code_t accept(tcp_t& client) noexcept {
        return uv_accept(reinterpret_cast<uv_stream_t*>(socket.get()),
          reinterpret_cast<uv_stream_t*>(client.socket.get()));
      }
      fan::event::error_code_t bind(const std::string& ip, int port) noexcept {
        struct sockaddr_in bind_addr;
        uv_ip4_addr(ip.c_str(), port, &bind_addr);
        return uv_tcp_bind(socket.get(), reinterpret_cast<sockaddr*>(&bind_addr), 0);
      }

      fan::event::task_t listen(const listen_address_t& address, listen_cb_t lambda);
      connector_t connect(const char* ip, int port) {
        return connector_t{ *this, ip, port };
      }
      connector_t connect(const getaddrinfo_t& info) {
        return connector_t{ *this, info };
      }
      reader_t read() {
        reader_t r{ *this };
        if (int result = r.start() != 0) {
          fan::throw_error("failed to start reading:", result);
        }
        return r;
      }
      writer_t write(const std::string& data) {
        writer_t w{ *this };
        w.write(data);
        return w;
      }
    };

    fan::event::task_t tcp_t::listen(const listen_address_t& address, listen_cb_t lambda) {
      if (address.port == 0) {
        fan::throw_error("invalid port");
      }
      bind(address.ip, address.port);
      static constexpr auto amount_of_connections = 128;
      listener_t listener = listener_t{ *this, amount_of_connections };
      while (true) {
        fan::network::tcp_t client;
        if (co_await listener != 0) {
          continue;
        }
        if (accept(client) == 0) {
          co_await lambda(std::move(client));
        }
      }
      co_return;
    }
    fan::event::task_t tcp_listen(listen_address_t address, listen_cb_t lambda) {
      tcp_t tcp;
      co_await tcp.listen(address, lambda);
    }
  }
}