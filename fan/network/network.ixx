module;

#include <uv.h>
#undef min
#undef max
#include <cstring>
#include <stdexcept>
#include <memory>
#include <coroutine>
#include <functional>
#include <unordered_map>
#include <array>
#include <string>

#include <openssl/ssl.h>
#include <openssl/err.h>
#include <openssl/bio.h>
#include <openssl/dtls1.h>
#include <openssl/rand.h>
#include <openssl/hmac.h>
#include <openssl/evp.h>

#include <fan/types/types.h>

export module fan.network;

export import fan.event;
export import fan.print;

export namespace fan {
  namespace network {
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
        uv_getaddrinfo(fan::event::get_event_loop(), &data->getaddrinfo_handle, [](uv_getaddrinfo_t* getaddrinfo_handle, int status, struct addrinfo* res) {
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

    // -------------------------------TCP-------------------------------
    struct tcp_t;
    struct connector_t {
      struct connector_data_t {
        uv_connect_t req;
        std::coroutine_handle<> co_handle;
        int status;
      };

      std::unique_ptr<connector_data_t> data;

      template <typename T>
      requires (std::is_same_v<T, tcp_t>)
      connector_t(const T& tcp, const std::string& ip, int port) :
        data{ std::make_unique<connector_data_t>() }
      {
        data->req.data = data.get();
        struct sockaddr_in client_addr;
        auto result = uv_ip4_addr(ip.c_str(), port, &client_addr);

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
          throw std::runtime_error(std::string("connection failed with:") + uv_strerror(data->status));
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
          fan::throw_error("listen error:"_str + uv_strerror(r));
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

    namespace error_code {
      enum {
        ok = 0,
        unknown = UV_UNKNOWN,
      };
    }

    using buffer_t = std::vector<char>;

    struct data_t {
      buffer_t buffer;
      ssize_t status = error_code::unknown;
      operator bool() {
        return status == error_code::ok;
      }
      operator ssize_t() const {
        return status;
      }
      operator const std::string() const {
        return { buffer.begin(), buffer.end() };
      }
      operator const buffer_t&() const {
        return buffer;
      }
    };

    struct raw_reader_t {
      std::shared_ptr<uv_stream_t> stream;
      buffer_t buf;
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
            [[likely]] if (self->co_handle && nread != 0) {
              self->co_handle();
            }
          }
        );
      }
      bool await_ready() const { return nread > 0; }
      void await_suspend(std::coroutine_handle<> h) { co_handle = h; }
      data_t await_resume() {
        data_t data;
        data.status = nread < 0 ? nread : error_code::ok;
        if (nread > 0) {
          data.buffer = buffer_t(buf.begin(), buf.begin() + nread);
        }
        nread = 0;
        co_handle = nullptr;
        return data;
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
        buffer_t to_write;
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

      int write(const buffer_t& some_data) {
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
          fan::throw_error("tcp write failed:", uv_strerror(r));
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
        if (data && data->status == 1) {
          data->write_handle.cb = [](uv_write_t* write_handle, int) {
            delete static_cast<writer_data_t*>(write_handle->data);
            };
          data.release();
        }
      }
    };

    struct message_t {
      buffer_t buffer;
      ssize_t status;
      bool done;

      operator bool() {
        return status == error_code::ok;
      }
      operator ssize_t() const {
        return status;
      }
      operator const std::string () const {
        return { buffer.begin(), buffer.end() };
      }
      operator const buffer_t& () const {
        return buffer;
      }
    };

    struct reader_t {
      std::shared_ptr<uv_stream_t> stream;
      buffer_t accumulated_buf;
      buffer_t temp_buf;
      ssize_t nread{ 0 };
      std::coroutine_handle<> co_handle;

      uint64_t expected_size{ 0 };
      uint64_t bytes_read{ 0 };
      bool reading_header{ true };
      bool is_raw_read{ false };
      bool is_fixed_size_read{ false };
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
        reading_header{ r.reading_header },
        is_raw_read{ r.is_raw_read },
        is_fixed_size_read{ r.is_fixed_size_read } {
        stream->data = this;
      }

      void setup_fixed_size_read(ssize_t len) {
        reading_header = false;
        expected_size = len;
        is_raw_read = false;
        is_fixed_size_read = true;
      }
      void setup_raw_read() {
        reading_header = false;
        expected_size = 0;
        is_raw_read = true;
        is_fixed_size_read = false;
      }
      void setup_header_read() {
        reading_header = true;
        expected_size = 0;
        is_raw_read = false;
        is_fixed_size_read = false;
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
              self->accumulated_buf.insert(self->accumulated_buf.end(), self->temp_buf.begin(), self->temp_buf.begin() + nread);
            }
            [[likely]] if (self->co_handle && nread != 0) {
              self->co_handle();
            }
          }
        );
      }
      bool await_ready() const {
        if (nread < 0) return true;
      //  if (nread == 0) return false;

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
      void await_suspend(std::coroutine_handle<> h) {
        co_handle = h;
      }
      message_t await_resume() {
        if (nread < 0) {
          co_handle = nullptr;
          auto error_status = nread;
          nread = 0;
          return { .status = error_status, .done = true };
        }

        co_handle = nullptr;

        if (is_raw_read) {
          buffer_t buffer;
          if (!accumulated_buf.empty()) {
            buffer = std::move(accumulated_buf);
            accumulated_buf.clear();
          }
          nread = 0;
          return { .buffer = std::move(buffer), .status = error_code::ok, .done = false };
        }
        else if (is_fixed_size_read) {
          if (accumulated_buf.size() >= expected_size) {
            buffer_t buffer(accumulated_buf.begin(), accumulated_buf.begin() + expected_size);
            accumulated_buf.erase(accumulated_buf.begin(), accumulated_buf.begin() + expected_size);
            nread = 0;
            return { .buffer = buffer, .status = error_code::ok, .done = true };
          }
          nread = 0;
          return { .status = error_code::ok, .done = false };
        }
        else {
          if (reading_header && accumulated_buf.size() >= header_size) {
            std::memcpy(&expected_size, accumulated_buf.data(), header_size);
            accumulated_buf.erase(accumulated_buf.begin(), accumulated_buf.begin() + header_size);
            reading_header = false;
            bytes_read = 0;
          }

          if (!reading_header && accumulated_buf.size() >= expected_size) {
            buffer_t buffer(accumulated_buf.begin(), accumulated_buf.begin() + expected_size);
            accumulated_buf.erase(accumulated_buf.begin(), accumulated_buf.begin() + expected_size);
            reading_header = true;
            expected_size = bytes_read = 0;
            nread = 0;
            return { .buffer = buffer, .status = error_code::ok, .done = true };
          }

          nread = 0;
          return { .status = error_code::ok, .done = false };
        }
      }

      ~reader_t() {
        if (stream) {
          uv_read_stop(stream.get());
        }
        accumulated_buf.clear();
        temp_buf.clear();
      }
    };

    template<typename T>
    struct typed_message_t {
      T data;
      ssize_t status;
      bool done;

      operator bool() const {
        return status == error_code::ok;
      }
      operator ssize_t() const {
        return status;
      }
      const T& operator*() const {
        return data;
      }
      T& operator*() {
        return data;
      }
      const T* operator->() const {
        return &data;
      }
      T* operator->() {
        return &data;
      }
    };

    template<typename T>
    struct typed_reader_t {
      reader_t& base_reader;

      typed_reader_t(reader_t& reader) : base_reader(reader) {
        base_reader.setup_fixed_size_read(sizeof(T));
      }

      bool await_ready() const {
        return base_reader.await_ready();
      }

      void await_suspend(std::coroutine_handle<> h) {
        base_reader.await_suspend(h);
      }

      typed_message_t<T> await_resume() {
        auto msg = base_reader.await_resume();
        typed_message_t<T> result;
        result.status = msg.status;
        result.done = msg.done;

        if (msg.status == error_code::ok && msg.buffer.size() >= sizeof(T)) {
          std::memcpy(&result.data, msg.buffer.data(), sizeof(T));
        }
        else {
          result.data = T{};
        }

        return result;
      }
    };

    struct writer_t {
      raw_writer_t raw_writer;
      std::vector<fan::event::task_t> client_tasks;

      template <typename T>
      requires (std::is_same_v<T, tcp_t>)
      writer_t(const T& tcp) : raw_writer(tcp) {}

      writer_t(const writer_t&) = delete;
      writer_t& operator=(const writer_t&) = delete;
      writer_t(writer_t&&) = default;
      writer_t& operator=(writer_t&&) = default;

      int write(const buffer_t& user_data) {
        uint64_t data_size = user_data.size();
        buffer_t message_data;
        message_data.reserve(sizeof(uint64_t) + data_size);

        message_data.insert(
          message_data.end(), 
          reinterpret_cast<const char*>(&data_size), 
          reinterpret_cast<const char*>(&data_size) + sizeof(uint64_t)
        );
        message_data.insert(message_data.end(), user_data.begin(), user_data.end());

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

    struct client_handler_t {
#define BLL_set_SafeNext 1
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_prefix client_list
#include <fan/fan_bll_preset.h>
#define BLL_set_Link 1
#define BLL_set_type_node uint32_t
#define BLL_set_NodeDataType fan::network::tcp_t*
#define BLL_set_CPP_CopyAtPointerChange 1 // maybe not necessary since holding *
#include <BLL/BLL.h>

      using nr_t = client_list_t::nr_t;
      client_list_t client_list;

      nr_t add_client();
      void remove_client(nr_t nr) {
        client_list.unlrec(nr);
      }

      tcp_t& get_client(nr_t id) {
        return *client_list[id];
      }
      const tcp_t& get_client(nr_t id) const {
        return get_client(id);
      }

      tcp_t& operator[](nr_t id) {
        return get_client(id);
      }
      const tcp_t& operator[](nr_t id) const {
        return get_client(id);
      }
      uint32_t amount_of_connections = 128;
    };
    client_handler_t& get_client_handler() {
      static client_handler_t client_handler;
      return client_handler;
    }

    struct tcp_t {
      client_handler_t::nr_t nr;
      fan::event::task_t task;
      using listen_cb_t = std::function<fan::event::task_t(tcp_t&)>;

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
        client_handler_t::nr_t nr = get_client_handler().client_list.NewNodeLast();
        get_client_handler().client_list[nr] = this;
        get_client_handler().client_list[nr]->nr = nr;
        uv_tcp_init(fan::event::get_event_loop(), socket.get());
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

      fan::event::task_t listen(const listen_address_t& address, listen_cb_t lambda, bool bind = false);
      connector_t connect(const std::string& ip, int port) {
        return connector_t{ *this, ip, port };
      }
      connector_t connect(const getaddrinfo_t& info) {
        return connector_t{ *this, info };
      }
      
      mutable std::unique_ptr<reader_t> persistent_reader;
      reader_t& get_reader() const {
        if (!persistent_reader) {
          persistent_reader = std::make_unique<reader_t>(*this);
          if (int err = persistent_reader->start(); err != 0) {
            fan::throw_error("start failed with:" + fan::event::strerror(err));
          }
        }
        return *persistent_reader;
      }
      reader_t& read(ssize_t len) const {
        auto& r = get_reader();
        r.setup_fixed_size_read(len);
        return r;
      }
      reader_t& read_raw() const {
        auto& r = get_reader();
        r.setup_raw_read();
        return r;
      }
      reader_t& read() const {
        auto& r = get_reader();
        r.setup_header_read();
        return r;
      }
      template <typename T>
      typed_reader_t<T> read() const {
        auto& r = get_reader();
        return typed_reader_t<T>(r);
      }

      writer_t write(const buffer_t& data) const {
        writer_t w{ *this };
        w.write(data);
        return w;
      }
      writer_t write(const std::string& data) const {
        return write(buffer_t(data.begin(), data.end()));
      }
      raw_writer_t write_raw(const buffer_t& data) const {
        raw_writer_t w{ *this };
        w.write(data);
        return w;
      }
      raw_writer_t write_raw(const std::string& data) const {
        return write_raw(buffer_t(data.begin(), data.end()));
      }
      template <typename T>
      raw_writer_t write_raw(T* data, ssize_t length) const {
        return write_raw(std::string((const char*)data, (const char*)data + length));
      }

      void broadcast(auto lambda) const {
        auto it = get_client_handler().client_list.GetNodeFirst();
        while (it != get_client_handler().client_list.dst) {
          get_client_handler().client_list.StartSafeNext(it);
          tcp_t* node = get_client_handler().client_list[it];
          if (node->nr != nr) {
            lambda(*node);
          }
          it = get_client_handler().client_list.EndSafeNext();
        }
      }
    };

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
      while (true) {
        if (co_await listener != 0) {
          continue;
        }
        auto client_id = get_client_handler().add_client();
        tcp_t& client = get_client_handler()[client_id];
        if (accept(client) == 0) {
          try {
            client.task = [client_id] (const auto& lambda) -> fan::event::task_t { co_await lambda(get_client_handler()[client_id]); }(lambda);
          }
          catch (const fan::exception_t& e) {
            fan::print("Client NRI:", client_id.NRI, "disconnected with error:", e.reason);
          }
        }
        fan::print("Removing client NRI:", client_id.NRI, "from client list");
        get_client_handler().remove_client(client_id);
      }
      co_return;
    }
    fan::event::task_t tcp_server_listen(listen_address_t address, tcp_t::listen_cb_t lambda) {
      tcp_t tcp;
      co_await tcp.listen(address, lambda, true);
    }
    fan::event::task_t tcp_listen(listen_address_t address, tcp_t::listen_cb_t lambda) {
      tcp_t tcp;
      co_await tcp.listen(address, lambda, false);
    }
    // -------------------------------TCP-------------------------------


    // -------------------------------UDP-------------------------------
    struct udp_t;
    struct socket_address_t {
      union {
        struct sockaddr_storage storage;
        struct sockaddr_in ipv4;
        struct sockaddr_in6 ipv6;
        struct sockaddr generic;
      } addr;

      socket_address_t() {
        std::memset(&addr, 0, sizeof(addr));
      }

      socket_address_t(const std::string& ip, int port) {
        std::memset(&addr, 0, sizeof(addr));
        if (uv_ip4_addr(ip.c_str(), port, &addr.ipv4) == 0) {
        }
        else if (uv_ip6_addr(ip.c_str(), port, &addr.ipv6) == 0) {
        }
        else {
          fan::throw_error("Invalid IP address: " + ip);
        }
      }

      socket_address_t(const struct sockaddr* sa) {
        std::memset(&addr, 0, sizeof(addr));
        if (sa->sa_family == AF_INET) {
          std::memcpy(&addr.ipv4, sa, sizeof(struct sockaddr_in));
        }
        else if (sa->sa_family == AF_INET6) {
          std::memcpy(&addr.ipv6, sa, sizeof(struct sockaddr_in6));
        }
      }

      std::string get_ip() const {
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

      int get_port() const {
        if (addr.generic.sa_family == AF_INET) {
          return ntohs(addr.ipv4.sin_port);
        }
        else if (addr.generic.sa_family == AF_INET6) {
          return ntohs(addr.ipv6.sin6_port);
        }
        return 0;
      }

      const struct sockaddr* sockaddr_ptr() const {
        return &addr.generic;
      }

      int sockaddr_len() const {
        if (addr.generic.sa_family == AF_INET) {
          return sizeof(struct sockaddr_in);
        }
        else if (addr.generic.sa_family == AF_INET6) {
          return sizeof(struct sockaddr_in6);
        }
        return sizeof(struct sockaddr_storage);
      }
    };

    struct udp_datagram_t {
      buffer_t data;
      socket_address_t sender;
      ssize_t status;

      udp_datagram_t() : status(0) {}

      udp_datagram_t(const buffer_t& msg, const socket_address_t& addr, ssize_t stat = 0)
        : data(msg), sender(addr), status(stat) {
      }

      operator bool() const {
        return status >= 0;
      }
      operator const std::string& () const {
        return { data.begin(), data.end() };
      }
      operator const buffer_t() const {
        return data;
      }

      const buffer_t& message() const { return data; }
      std::string sender_ip() const { return sender.get_ip(); }
      int sender_port() const { return sender.get_port(); }
      bool is_error() const { return status < 0; }
      bool is_empty() const { return data.empty(); }
      size_t size() const { return data.size(); }
    };

    struct udp_send_t {
      struct send_data_t {
        uv_udp_send_t req;
        buffer_t data_buffer;
        socket_address_t destination;
        std::coroutine_handle<> co_handle;
        int status{ 1 }; // 1 = pending, 0 = success, <0 = error
      };

      std::unique_ptr<send_data_t> data;

      template <typename T>
      requires (std::is_same_v<T, udp_t>)
      udp_send_t(const T& udp, const buffer_t& message, const socket_address_t& addr);
      template <typename T>
      requires (std::is_same_v<T, udp_t>)
      udp_send_t(const T& udp, const buffer_t& message, const std::string& ip, int port);

      template <typename T>
      requires (std::is_same_v<T, udp_t>)
      udp_send_t(const T& udp, const std::string& message, const std::string& ip, int port);
      template <typename T>
      requires (std::is_same_v<T, udp_t>)
      udp_send_t(const T& udp, const std::string& message, const socket_address_t& addr);
      udp_send_t(udp_send_t&&) = default;
      udp_send_t& operator=(udp_send_t&&) = default;

      bool await_ready() const { return data->status <= 0; }
      void await_suspend(std::coroutine_handle<> h) { data->co_handle = h; }
      int await_resume() {
        if (data->status < 0) {
          throw std::runtime_error(std::string("UDP send failed: ") + uv_strerror(data->status));
        }
        return data->status;
      }

      ~udp_send_t() {
        if (data && data->status == 1) {
          data->req.cb = [](uv_udp_send_t* req, int) {
            delete static_cast<send_data_t*>(req->data);
            };
          data.release();
        }
      }
    };

    struct udp_recv_t {
      std::shared_ptr<uv_udp_t> socket;
      std::coroutine_handle<> co_handle;
      udp_datagram_t datagram;
      bool ready{ false };
      bool receiving{ false };

      template <typename T>
        requires (std::is_same_v<T, udp_t>)
      udp_recv_t(const T& udp);
      udp_recv_t(const udp_recv_t&) = delete;
      udp_recv_t(udp_recv_t&& r) noexcept :
        socket{ std::move(r.socket) },
        co_handle{ std::move(r.co_handle) },
        datagram{ std::move(r.datagram) },
        ready{ r.ready },
        receiving{ r.receiving } {
        if (socket) {
          socket->data = this;
        }
      }

      int start() {
        if (receiving) return 0;
        receiving = true;
        socket->data = this;
        return uv_udp_recv_start(socket.get(),
          [](uv_handle_t* handle, size_t suggested_size, uv_buf_t* buf) {
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

            //uv_udp_recv_stop(handle);
            //self->receiving = false;

            if (self->co_handle) {
              self->co_handle();
            }
          }
        );
      }
      void stop() {
        if (receiving) {
          uv_udp_recv_stop(socket.get());
          receiving = false;
        }
      }
      bool await_ready() const { return ready; }
      void await_suspend(std::coroutine_handle<> h) {
        co_handle = h;
        start();
      }
      udp_datagram_t await_resume() {
        ready = false;
        co_handle = nullptr;
        return std::move(datagram);
      }

      ~udp_recv_t() {
        stop();
      }
    };

    struct udp_recvfrom_t {
      std::shared_ptr<uv_udp_t> socket;
      std::coroutine_handle<> co_handle;
      udp_datagram_t datagram;
      bool ready{ false };
      bool receiving{ false };
      bool filter_sender{ false };
      socket_address_t expected_sender;

      template<typename T>
        requires(std::is_same_v<T, udp_t>)
      udp_recvfrom_t(const T& udp) : socket(udp.socket) {
        socket->data = this;
      }

      void set_expected_sender(const socket_address_t& sender) {
        expected_sender = sender;
        filter_sender = true;
      }

      void set_expected_sender(const std::string& ip, int port) {
        expected_sender = socket_address_t(ip, port);
        filter_sender = true;
      }

      bool matches_expected_sender(const socket_address_t& actual_sender) const {
        if (!filter_sender) return true;

        return (expected_sender.get_ip() == actual_sender.get_ip() &&
          expected_sender.get_port() == actual_sender.get_port());
      }

      udp_recvfrom_t(const udp_recvfrom_t&) = delete;
      udp_recvfrom_t(udp_recvfrom_t&& r) noexcept
        : socket{ std::move(r.socket) },
        co_handle{ std::move(r.co_handle) },
        datagram{ std::move(r.datagram) },
        ready{ r.ready },
        receiving{ r.receiving } {
        if (socket) {
          socket->data = this;
        }
      }

      int start() {
        if (receiving) return 0;
        receiving = true;
        socket->data = this;

        return uv_udp_recv_start(socket.get(),
          [](uv_handle_t* handle, size_t suggested_size, uv_buf_t* buf) {
            auto self = static_cast<udp_recvfrom_t*>(handle->data);
            self->datagram.data.resize(2000);
            *buf = uv_buf_init(self->datagram.data.data(), 2000);
          },
          [](uv_udp_t* handle, ssize_t nread, const uv_buf_t* buf,
            const struct sockaddr* addr, unsigned flags) 
          {
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
            //uv_udp_recv_stop(handle);
            //self->receiving = false;

            if (self->co_handle) {
              self->co_handle();
            }
          }
        );
      }

      void stop() {
        if (receiving) {
          uv_udp_recv_stop(socket.get());
          receiving = false;
        }
      }

      bool await_ready() const {
        return ready;
      }

      void await_suspend(std::coroutine_handle<> h) {
        co_handle = h;
        if (int err = start(); err != 0) {
          fan::throw_error("start failed with:" + fan::event::strerror(err));
        }
      }

      udp_datagram_t await_resume() {
        ready = false;
        co_handle = nullptr;
        return std::move(datagram);
      }

      ~udp_recvfrom_t() {
        stop();
      }
    };


    struct udp_t {
      using recv_cb_t = std::function<fan::event::task_t(udp_t&, const udp_datagram_t&)>;

      std::shared_ptr<uv_udp_t> socket;
      fan::event::task_t task;

      struct udp_deleter_t {
        void operator()(void* p) const {
          uv_close(static_cast<uv_handle_t*>(p), [](uv_handle_t* req) {
            delete reinterpret_cast<uv_udp_t*>(req);
          });
        }
      };

      udp_t() : socket(new uv_udp_t, udp_deleter_t{}) {
        int result = uv_udp_init(fan::event::get_event_loop(), socket.get());
        if (result != 0) {
          fan::throw_error("Failed to initialize UDP socket:"_str + uv_strerror(result));
        }
      }

      udp_t(const udp_t&) = delete;
      udp_t& operator=(const udp_t&) = delete;
      udp_t(udp_t&&) = default;
      udp_t& operator=(udp_t&&) = delete;

      fan::event::error_code_t bind(const std::string& ip, int port, unsigned int flags = 0) noexcept {
        socket_address_t addr(ip, port);
        return uv_udp_bind(socket.get(), addr.sockaddr_ptr(), flags);
      }
      udp_send_t send(const std::string& data, const std::string& ip, int port) {
        return udp_send_t{ *this, data, ip, port };
      }
      udp_send_t send(const std::string& data, const socket_address_t& addr) {
        return udp_send_t{ *this, data, addr };
      }
      udp_send_t send(const buffer_t& data, const std::string& ip, int port) {
        return udp_send_t{ *this, data, ip, port };
      }
      udp_send_t send(const buffer_t& data, const socket_address_t& addr) {
        return udp_send_t{ *this, data, addr };
      }
      udp_send_t reply(const udp_datagram_t& received, const std::string& response) {
        return udp_send_t{ *this, response, received.sender };
      }
      udp_recv_t recv() {
        return udp_recv_t{ *this };
      }
      udp_recvfrom_t recvfrom() {
        return udp_recvfrom_t{ *this };
      }
      fan::event::task_value_resume_t<udp_datagram_t> recvfrom(const std::string& ip, int port) {
        auto recvfrom = udp_recvfrom_t{ *this };
        recvfrom.set_expected_sender(ip, port);
        co_return co_await recvfrom;
      }
      fan::event::task_value_resume_t<udp_datagram_t> recvfrom(const socket_address_t& addr) {
        auto recvfrom = udp_recvfrom_t{ *this };
        recvfrom.set_expected_sender(addr);
        co_return co_await recvfrom;
      }

      fan::event::task_t listen(const listen_address_t& address, recv_cb_t callback, bool bind_ = false) {
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

      socket_address_t get_sockname() const {
        struct sockaddr_storage addr;
        int namelen = sizeof(addr);
        int result = uv_udp_getsockname(socket.get(), reinterpret_cast<struct sockaddr*>(&addr), &namelen);

        if (result != 0) {
          return socket_address_t();
        }

        return socket_address_t(reinterpret_cast<struct sockaddr*>(&addr));
      }

      fan::event::error_code_t set_broadcast(bool enable) noexcept {
        return uv_udp_set_broadcast(socket.get(), enable ? 1 : 0);
      }
      fan::event::error_code_t set_ttl(int ttl) noexcept {
        return uv_udp_set_ttl(socket.get(), ttl);
      }
      fan::event::error_code_t join_multicast(const std::string& multicast_addr, const std::string& interface_addr = "") noexcept {
        return uv_udp_set_membership(socket.get(), multicast_addr.c_str(),
          interface_addr.empty() ? nullptr : interface_addr.c_str(),
          UV_JOIN_GROUP);
      }
      fan::event::error_code_t leave_multicast(const std::string& multicast_addr, const std::string& interface_addr = "") noexcept {
        return uv_udp_set_membership(socket.get(), multicast_addr.c_str(),
          interface_addr.empty() ? nullptr : interface_addr.c_str(),
          UV_LEAVE_GROUP);
      }
      fan::event::error_code_t set_multicast_ttl(int ttl) noexcept {
        return uv_udp_set_multicast_ttl(socket.get(), ttl);
      }
      fan::event::error_code_t set_multicast_interface(const std::string& interface_addr) noexcept {
        return uv_udp_set_multicast_interface(socket.get(), interface_addr.c_str());
      }

      fan::event::error_code_t set_multicast_loop(bool enable) noexcept {
        return uv_udp_set_multicast_loop(socket.get(), enable ? 1 : 0);
      }

      bool is_bound() const {
        auto addr = get_sockname();
        return addr.get_port() != 0;
      }
    };

    template <typename T>
    requires (std::is_same_v<T, udp_t>)
    udp_send_t::udp_send_t(const T& udp, const buffer_t& message, const socket_address_t& addr) :
      data(std::make_unique<send_data_t>()) {
      data->req.data = data.get();
      data->data_buffer = message;
      data->destination = addr;
      uv_buf_t buf = uv_buf_init(data->data_buffer.data(), data->data_buffer.size());
      data->status = uv_udp_send(&data->req, udp.socket.get(), &buf, 1,
        data->destination.sockaddr_ptr(),
        [](uv_udp_send_t* req, int status) {
          auto* data = static_cast<send_data_t*>(req->data);
          data->status = status;
          if (data->co_handle) {
            data->co_handle();
          }
        }
      );

      if (data->status == 0) {
        data->status = 1;
      }
    }

    template <typename T>
    requires (std::is_same_v<T, udp_t>)
    udp_send_t::udp_send_t(const T& udp, const buffer_t& message, const std::string& ip, int port) :
      udp_send_t(udp, message, socket_address_t(ip, port)) {}
    template <typename T>
      requires (std::is_same_v<T, udp_t>)
    udp_send_t::udp_send_t(const T& udp, const std::string& message, const socket_address_t& addr) :
      udp_send_t(udp, buffer_t{ message.begin(), message.end() }, addr) {
    }
    template <typename T>
      requires (std::is_same_v<T, udp_t>)
    udp_send_t::udp_send_t(const T& udp, const std::string& message, const std::string& ip, int port) :
      udp_send_t(udp, message, socket_address_t(ip, port)) {
    }

    template <typename T>
    requires (std::is_same_v<T, udp_t>)
    udp_recv_t::udp_recv_t(const T& udp) : socket(udp.socket) {
      socket->data = this;
    }
    fan::event::task_t udp_listen(const listen_address_t& address, udp_t::recv_cb_t callback) {
      udp_t udp;
      co_await udp.listen(address, callback);
    }
    // -------------------------------UDP-------------------------------



    struct keep_alive_config_t {
      static constexpr size_t timer_step_limit = 7;
      std::array<int, timer_step_limit> timer_steps{ 30, 1, 2, 2, 4, 4, 4 };
      size_t current_step = 0;
      bool is_running = false;
    };

    template<typename ConnectionType>
    struct keep_alive_timer_t {
    public:
      using timeout_callback_t = std::function<void()>;
      using send_keepalive_callback_t = std::function<fan::event::task_t(ConnectionType&)>;
    private:
      ConnectionType& connection;
      keep_alive_config_t config;
      fan::event::task_t timer_task;
      send_keepalive_callback_t send_callback;
      timeout_callback_t timeout_callback;
      bool should_stop = false;
      bool should_reset = false;

      fan::event::task_t timer_coroutine() {
        while (!should_stop && config.is_running) {
          co_await fan::co_sleep(config.timer_steps[config.current_step] * 1000);

          if (should_reset) {
            should_reset = false;
            config.current_step = 0;
            continue;
          }

          if (send_callback) {
            try {
              co_await send_callback(connection);
            }
            catch (const std::exception& e) {
              fan::print("Keep alive send failed: ", e.what());
            }
          }

          if (config.current_step == config.timer_step_limit - 1) {
            if (timeout_callback) {
              timeout_callback();
            }
            break;
          }
          else {
            config.current_step++;
          }
        }
        co_return;
      }

    public:
      keep_alive_timer_t(ConnectionType& conn,
        send_keepalive_callback_t send_cb,
        timeout_callback_t timeout_cb = nullptr)
        : connection(conn), send_callback(send_cb), timeout_callback(timeout_cb) {
      }

      ~keep_alive_timer_t() {
        stop();
      }

      void start() {
        if (config.is_running) {
          return;
        }
        config.current_step = 0;
        config.is_running = true;
        should_stop = false;
        should_reset = false;
        timer_task = timer_coroutine();
      }

      void stop() {
        should_stop = true;
        config.is_running = false;
      }

      void reset() {
        if (config.is_running) {
          should_reset = true;
        }
        else {
          start();
        }
      }
      bool is_running() const {
        return config.is_running;
      }
      size_t get_current_step() const {
        return config.current_step;
      }
    };

    class tcp_keep_alive_t {
    private:
      tcp_t& tcp_connection;
      std::unique_ptr<keep_alive_timer_t<tcp_t>> timer;

      fan::event::task_t send_keep_alive(tcp_t& tcp, const buffer_t& payload) {
        try {
          co_await tcp.write_raw(payload);
        }
        catch (const std::exception& e) {
          throw;
        }
        co_return;
      }

      void on_timeout() {
        fan::print("TCP keep alive timeout");
      }

    public:
      explicit tcp_keep_alive_t(tcp_t& tcp_conn, auto send_keep_alive_cb)
        : tcp_connection(tcp_conn) {
        timer = std::make_unique<keep_alive_timer_t<tcp_t>>(
          tcp_connection,
          [this, send_keep_alive_cb](tcp_t& tcp) -> fan::event::task_t {
            co_await send_keep_alive_cb(tcp);
          },
          [this]() { on_timeout(); }
        );
      }
      void start() {
        timer->start();
      }
      void stop() {
        timer->stop();
      }
      void reset() {
        timer->reset();
      }
      bool is_running() const {
        return timer->is_running();
      }
    };

    struct udp_keep_alive_t {
    private:
      udp_t& udp_connection;
      socket_address_t server_address;
      std::unique_ptr<keep_alive_timer_t<udp_t>> timer;

      fan::event::task_t send_keep_alive(udp_t& udp, const buffer_t& payload) {
        try {
          co_await udp.send(payload, server_address);
        }
        catch (const std::exception& e) {
          throw;
        }
        co_return;
      }

      void on_timeout() {
        fan::print("UDP keep alive timeout");
      }

    public:
      udp_keep_alive_t(udp_t& udp_conn) : udp_connection(udp_conn) {}

      void set_server(const socket_address_t& server_addr, auto send_keep_alive_cb) {
        this->server_address = server_addr;
        timer = std::make_unique<keep_alive_timer_t<udp_t>>(
          udp_connection,
          [this, send_keep_alive_cb](udp_t& udp) -> fan::event::task_t {
            co_await send_keep_alive_cb(udp);
          },
          [this]() { on_timeout(); }
        );
      }

      udp_keep_alive_t(udp_t& udp_conn, const socket_address_t& server_addr, auto send_keep_alive_cb)
        : udp_connection(udp_conn) {
        set_server(server_addr, send_keep_alive_cb);
      }
      udp_keep_alive_t(udp_t& udp_conn, const std::string& server_ip, int server_port, auto send_keep_alive_cb)
        : udp_keep_alive_t(udp_conn, socket_address_t(server_ip, server_port), send_keep_alive_cb) {}
      void start() {
        timer->start();
      }
      void stop() {
        timer->stop();
      }
      void reset() {
        timer->reset();
      }
      bool is_running() const {
        return timer->is_running();
      }
      void update_server_address(const std::string& server_ip, int server_port) {
        server_address = socket_address_t(server_ip, server_port);
      }
      void update_server_address(const socket_address_t& server_addr) {
        server_address = server_addr;
      }
    };
    /*
    class stream_cipher_t {
    private:
      unsigned char key[32];  // 256-bit key
      bool key_set = false;
      std::string key_hex_string;

    public:
      std::string get_key_hex() const {
        if (!key_set) return "";
        return key_hex_string;
      }

      // Set encryption key (32 bytes for AES-256)
      bool set_key(const unsigned char* new_key, size_t key_len) {
        if (key_len != 32) return false;
        memcpy(key, new_key, 32);
        key_set = true;
        return true;
      }

      // Set key from hex string
      bool set_key_hex(const std::string& hex_key) {
        if (hex_key.length() != 64) return false; // 32 bytes = 64 hex chars

        unsigned char binary_key[32];
        for (size_t i = 0; i < 32; ++i) {
          std::string byte_str = hex_key.substr(i * 2, 2);
          binary_key[i] = (unsigned char)std::stoul(byte_str, nullptr, 16);
        }

        bool result = set_key(binary_key, 32);
        if (result) {
          key_hex_string = hex_key; // Store the hex string
        }
        return result;
      }

      std::string generate_key() {
        unsigned char new_key[32];
        if (RAND_bytes(new_key, 32) != 1) {
          return "";
        }

        set_key(new_key, 32);

        // Return as hex string and store it
        std::string hex_key;
        for (size_t i = 0; i < 32; ++i) {
          char buf[3];
          snprintf(buf, sizeof(buf), "%02x", new_key[i]);
          hex_key += buf;
        }

        key_hex_string = hex_key; // Store the hex string
        return hex_key;
      }

      // Encrypt data using AES-256-GCM
      std::vector<uint8_t> encrypt(const std::vector<uint8_t>& plaintext) {
        if (!key_set || plaintext.empty()) return {};

        EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
        if (!ctx) return {};

        // Generate random IV (12 bytes for GCM)
        unsigned char iv[12];
        if (RAND_bytes(iv, 12) != 1) {
          EVP_CIPHER_CTX_free(ctx);
          return {};
        }

        // Initialize encryption
        if (EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), nullptr, nullptr, nullptr) != 1) {
          EVP_CIPHER_CTX_free(ctx);
          return {};
        }

        // Set IV length
        if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, 12, nullptr) != 1) {
          EVP_CIPHER_CTX_free(ctx);
          return {};
        }

        // Initialize key and IV
        if (EVP_EncryptInit_ex(ctx, nullptr, nullptr, key, iv) != 1) {
          EVP_CIPHER_CTX_free(ctx);
          return {};
        }

        // Prepare output buffer: IV(12) + ciphertext + tag(16)
        std::vector<uint8_t> output;
        output.resize(12 + plaintext.size() + 16);

        // Copy IV to output
        memcpy(output.data(), iv, 12);

        // Encrypt
        int len;
        if (EVP_EncryptUpdate(ctx, output.data() + 12, &len, plaintext.data(), (int)plaintext.size()) != 1) {
          EVP_CIPHER_CTX_free(ctx);
          return {};
        }
        int ciphertext_len = len;

        // Finalize
        if (EVP_EncryptFinal_ex(ctx, output.data() + 12 + len, &len) != 1) {
          EVP_CIPHER_CTX_free(ctx);
          return {};
        }
        ciphertext_len += len;

        // Get authentication tag
        if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, 16, output.data() + 12 + ciphertext_len) != 1) {
          EVP_CIPHER_CTX_free(ctx);
          return {};
        }

        EVP_CIPHER_CTX_free(ctx);

        // Resize to actual length
        output.resize(12 + ciphertext_len + 16);
        return output;
      }

      // Decrypt data using AES-256-GCM
      std::vector<uint8_t> decrypt(const std::vector<uint8_t>& ciphertext) {
        if (!key_set || ciphertext.size() < 28) return {}; // IV(12) + tag(16) = 28 minimum

        EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
        if (!ctx) return {};

        // Extract IV, ciphertext, and tag
        const unsigned char* iv = ciphertext.data();
        const unsigned char* encrypted_data = ciphertext.data() + 12;
        int encrypted_len = (int)ciphertext.size() - 28;
        const unsigned char* tag = ciphertext.data() + 12 + encrypted_len;

        // Initialize decryption
        if (EVP_DecryptInit_ex(ctx, EVP_aes_256_gcm(), nullptr, nullptr, nullptr) != 1) {
          EVP_CIPHER_CTX_free(ctx);
          return {};
        }

        // Set IV length
        if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, 12, nullptr) != 1) {
          EVP_CIPHER_CTX_free(ctx);
          return {};
        }

        // Initialize key and IV
        if (EVP_DecryptInit_ex(ctx, nullptr, nullptr, key, iv) != 1) {
          EVP_CIPHER_CTX_free(ctx);
          return {};
        }

        // Prepare output buffer
        std::vector<uint8_t> plaintext(encrypted_len);

        // Decrypt
        int len;
        if (EVP_DecryptUpdate(ctx, plaintext.data(), &len, encrypted_data, encrypted_len) != 1) {
          EVP_CIPHER_CTX_free(ctx);
          return {};
        }
        int plaintext_len = len;

        // Set expected tag
        if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, 16, const_cast<unsigned char*>(tag)) != 1) {
          EVP_CIPHER_CTX_free(ctx);
          return {};
        }

        // Finalize and verify tag
        if (EVP_DecryptFinal_ex(ctx, plaintext.data() + len, &len) != 1) {
          EVP_CIPHER_CTX_free(ctx);
          return {}; // Authentication failed or corruption detected
        }
        plaintext_len += len;

        EVP_CIPHER_CTX_free(ctx);

        plaintext.resize(plaintext_len);
        return plaintext;
      }

      // Test the encryption/decryption
      bool test() {
        std::string test_key = generate_key();
        if (test_key.empty()) return false;

        std::string test_data = "Hello, secure stream data!";
        std::vector<uint8_t> original(test_data.begin(), test_data.end());

        auto encrypted = encrypt(original);
        if (encrypted.empty()) return false;

        auto decrypted = decrypt(encrypted);
        if (decrypted.size() != original.size()) return false;

        return memcmp(original.data(), decrypted.data(), original.size()) == 0;
      }
    };

    // Global cipher instance
    stream_cipher_t& get_stream_cipher() {
      static stream_cipher_t cipher;
      return cipher;
    }*/
  }
}

fan::network::client_handler_t::nr_t fan::network::client_handler_t::add_client() {
  nr_t nr = client_list.NewNodeLast();
  client_list[nr] = new fan::network::tcp_t;
  client_list[nr]->nr = nr;
  return nr;
}