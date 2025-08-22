module;

#include <fan/utility.h>

#include <uv.h>
#undef min
#undef max
#undef NO_ERROR

#include <curl/curl.h>
#include <curl/multi.h>
#include <cstring>
#include <stdexcept>
#include <memory>
#include <coroutine>
#include <functional>
#include <unordered_map>
#include <array>
#include <string>

#include <mutex>
#include <cstdint>


export module fan.network;

import fan.utility;
export import fan.event;
import fan.print;
import fan.types.json;

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
        uv_getaddrinfo(fan::event::get_loop(), &data->getaddrinfo_handle, [](uv_getaddrinfo_t* getaddrinfo_handle, int status, struct addrinfo* res) {
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

      struct tcp_deleter_t {
        void operator()(void* p) const {
          uv_close(static_cast<uv_handle_t*>(p), [](uv_handle_t* req) {
            try {
              delete reinterpret_cast<uv_tcp_t*>(req);
            }
            catch (std::exception e) {
              fan::print("failed to delete tcp:", e.what());
            }
          });
        }
      };

      tcp_t() : socket(new uv_tcp_t, tcp_deleter_t{}) {
        client_handler_t::nr_t nr = get_client_handler().client_list.NewNodeLast();
        get_client_handler().client_list[nr] = this;
        get_client_handler().client_list[nr]->nr = nr;
        uv_tcp_init(fan::event::get_loop(), socket.get());
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
      
      mutable std::unique_ptr<reader_t> reader;
      reader_t& get_reader() const {
        if (!reader) {
          reader = std::make_unique<reader_t>(*this);
          if (int err = reader->start(); err != 0) {
            fan::throw_error("start failed with:" + fan::event::strerror(err));
          }
        }
        return *reader;
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
            client.task = [client_id] (const auto& lambda) -> fan::event::task_t { 
              co_await lambda(get_client_handler()[client_id]);
              get_client_handler().remove_client(client_id);
            }(lambda);
          }
          catch (const fan::exception_t& e) {
            fan::print("Client NRI:", client_id.NRI, "disconnected with error:", e.reason);
            get_client_handler().remove_client(client_id);
          }
        }
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
        int result = uv_udp_init(fan::event::get_loop(), socket.get());
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
      std::array<int, timer_step_limit> timer_steps{ 1, 1, 2, 2, 4, 4, 4 };
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

      fan::event::task_t timer_coroutine() {
        while (!should_stop && config.is_running) {
          co_await fan::co_sleep(config.timer_steps[config.current_step] * 1000);

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
        timer_task = timer_coroutine();
      }

      void stop() {
        should_stop = true;
        config.is_running = false;
      }

      void reset() {
        if (config.is_running) {
          stop();
          start();
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
        if (timer) {
          timer->stop();
          timer.reset();
          timer = nullptr;
        }
        this->server_address = server_addr;
        timer = std::make_unique<keep_alive_timer_t<udp_t>>(
          udp_connection,
          [this, send_keep_alive_cb](udp_t& udp) -> fan::event::task_t {
            co_await send_keep_alive_cb(udp);
          },
          [this]() { on_timeout(); }
        );
        timer->start();
      }

      udp_keep_alive_t(udp_t& udp_conn, const socket_address_t& server_addr, auto send_keep_alive_cb)
        : udp_connection(udp_conn) {
        set_server(server_addr, send_keep_alive_cb);
      }
      udp_keep_alive_t(udp_t& udp_conn, const std::string& server_ip, int server_port, auto send_keep_alive_cb)
        : udp_keep_alive_t(udp_conn, socket_address_t(server_ip, server_port), send_keep_alive_cb) {}
      ~udp_keep_alive_t() {
        if (timer) {
          timer->stop();
          timer.reset();
        }
      }
      void start() {
        if (timer == nullptr) {
          return;
        }
        timer->start();
      }
      void stop() {
        if (timer == nullptr) {
          return;
        }
        timer->stop();
      }
      void reset() {
        if (timer == nullptr) {
          return;
        }
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


    // -------------------------------HTTP/REST-------------------------------
    
    struct http_method_t {
      enum {
        get = 0,
        post,
        put,
        delete_,
        patch,
        head,
        options
      };
    };

    struct http_status_t {
      enum {
        ok = 200,
        created = 201,
        no_content = 204,
        bad_request = 400,
        unauthorized = 401,
        forbidden = 403,
        not_found = 404,
        method_not_allowed = 405,
        internal_server_error = 500,
        not_implemented = 501,
        service_unavailable = 503
      };
    };

    struct http_error_t {
      enum {
        invalid_json = 1,
        invalid_param,
        connection_failed,
        timeout,
        parse_failed,
        not_found_error,
        database_error,
        validation_error
      };

      int code;
      std::string message;

      http_error_t(int c, std::string msg) : code(c), message(std::move(msg)) {}
    };

    constexpr const char* HTTP_CRLF = "\r\n";
    constexpr const char* HTTP_HEADER_END = "\r\n\r\n";
    constexpr const char* HTTP_VERSION = "HTTP/1.1";

    std::string build_get_request(const std::string& path, const std::string& host, bool keep_alive = false) {
      return "GET " + path + " " + HTTP_VERSION + HTTP_CRLF +
        "Host: " + host + HTTP_CRLF +
        "Connection: " + (keep_alive ? "keep-alive" : "close") + HTTP_CRLF + HTTP_CRLF;
    }

    std::string build_post_request(const std::string& path, const std::string& host, const std::string& body, bool keep_alive = false) {
      return "POST " + path + " " + HTTP_VERSION + HTTP_CRLF +
        "Host: " + host + HTTP_CRLF +
        "Content-Type: application/json" + HTTP_CRLF +
        "Content-Length: " + std::to_string(body.length()) + HTTP_CRLF +
        "Connection: " + (keep_alive ? "keep-alive" : "close") + HTTP_CRLF + HTTP_CRLF + body;
    }

    std::string build_get_request_with_headers(const std::string& path, const std::string& host,
      const std::unordered_map<std::string, std::string>& headers,
      bool keep_alive = false) {
      std::string request = "GET " + path + " " + HTTP_VERSION + HTTP_CRLF +
        "Host: " + host + HTTP_CRLF;

      // custom headers
      for (const auto& [key, value] : headers) {
        request += key + ": " + value + HTTP_CRLF;
      }

      request += "Connection: " + (keep_alive ? std::string("keep-alive") : std::string("close")) + HTTP_CRLF + HTTP_CRLF;
      return request;
    }

    inline std::expected<fan::json, http_error_t> parse_json_simple(const std::string& json_str) {
      if (json_str.empty()) {
        return std::unexpected(http_error_t{ http_error_t::invalid_json, "Empty JSON string" });
      }

      try {
        return fan::json::parse(json_str);
      }
      catch (...) {
        return std::unexpected(http_error_t{ http_error_t::invalid_json, "Invalid JSON format" });
      }
    }

    struct http_request_t {
      int method;
      std::string path;
      std::string raw_path;
      std::unordered_map<std::string, std::string> headers;
      std::unordered_map<std::string, std::string> params;
      std::unordered_map<std::string, std::string> query;
      std::string body;

      template<typename T>
      std::expected<T, http_error_t> param(const std::string& name) const {
        auto it = params.find(name);
        if (it == params.end()) {
          return std::unexpected(http_error_t{ http_error_t::invalid_param, "Parameter '" + name + "' not found" });
        }

        try {
          if constexpr (std::is_same_v<T, int>) {
            return std::stoi(it->second);
          }
          else if constexpr (std::is_same_v<T, double>) {
            return std::stod(it->second);
          }
          else if constexpr (std::is_same_v<T, std::string>) {
            return it->second;
          }
        }
        catch (...) {
          return std::unexpected(http_error_t{ http_error_t::invalid_param, "Invalid parameter format for '" + name + "'" });
        }

        return std::unexpected(http_error_t{ http_error_t::invalid_param, "Unsupported parameter type" });
      }

      std::expected<fan::json, http_error_t> json() const {
        if (body.empty()) {
          return std::unexpected(http_error_t{ http_error_t::invalid_json, "Empty request body" });
        }
        return parse_json_simple(body);
      }

      std::string header(const std::string& name) const {
        auto it = headers.find(name);
        return it != headers.end() ? it->second : "";
      }

      template<typename T>
      std::expected<T, http_error_t> query_param(const std::string& name) const {
        auto it = query.find(name);
        if (it == query.end()) {
          return std::unexpected(http_error_t{ http_error_t::invalid_param, "Query parameter '" + name + "' not found" });
        }

        try {
          if constexpr (std::is_same_v<T, int>) {
            return std::stoi(it->second);
          }
          else if constexpr (std::is_same_v<T, double>) {
            return std::stod(it->second);
          }
          else if constexpr (std::is_same_v<T, std::string>) {
            return it->second;
          }
        }
        catch (...) {
          return std::unexpected(http_error_t{ http_error_t::invalid_param, "Invalid query parameter format for '" + name + "'" });
        }

        return std::unexpected(http_error_t{ http_error_t::invalid_param, "Unsupported query parameter type" });
      }
    };

    struct http_response_t {
      int status_code = http_status_t::ok;
      std::unordered_map<std::string, std::string> headers;
      std::string body;

      http_response_t& status(int code) {
        status_code = code;
        return *this;
      }

      http_response_t& header(const std::string& key, const std::string& value) {
        headers[key] = value;
        return *this;
      }

      http_response_t& json(const fan::json& data) {
        body = data.dump();
        headers["Content-Type"] = "application/json";
        return *this;
      }

      http_response_t& text(const std::string& data) {
        body = data;
        headers["Content-Type"] = "text/plain";
        return *this;
      }

      http_response_t& html(const std::string& data) {
        body = data;
        headers["Content-Type"] = "text/html";
        return *this;
      }

      http_response_t& ok(const fan::json& data = nullptr) {
        status_code = http_status_t::ok;
        if (!data.is_null()) {
          json(data);
        }
        return *this;
      }

      http_response_t& created(const fan::json& data = nullptr) {
        status_code = http_status_t::created;
        if (!data.is_null()) {
          json(data);
        }
        return *this;
      }

      http_response_t& not_found(const std::string& message = "Not Found") {
        status_code = http_status_t::not_found;
        return text(message);
      }

      http_response_t& bad_request(const std::string& message = "Bad Request") {
        status_code = http_status_t::bad_request;
        return text(message);
      }

      http_response_t& internal_error(const std::string& message = "Internal Server Error") {
        status_code = http_status_t::internal_server_error;
        return text(message);
      }

      http_response_t& error(const http_error_t& err) {
        switch (err.code) {
        case http_error_t::invalid_param:
        case http_error_t::invalid_json:
        case http_error_t::validation_error:
          return bad_request(err.message);
        case http_error_t::not_found_error:
          return not_found(err.message);
        case http_error_t::database_error:
        case http_error_t::connection_failed:
        case http_error_t::timeout:
        default:
          return internal_error(err.message);
        }
      }

      std::string to_string() const {
        std::string response = "HTTP/1.1 " + std::to_string(status_code) + " ";

        switch (status_code) {
        case http_status_t::ok: response += "OK"; break;
        case http_status_t::created: response += "Created"; break;
        case http_status_t::bad_request: response += "Bad Request"; break;
        case http_status_t::not_found: response += "Not Found"; break;
        case http_status_t::internal_server_error: response += "Internal Server Error"; break;
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
    };

    using async_handler_t = std::function<fan::event::task_t(const http_request_t&, http_response_t&)>;

    struct route_t {
      int method;
      std::string pattern;
      async_handler_t handler;

      bool matches(int req_method, const std::string& path, std::unordered_map<std::string, std::string>& params) const {
        if (req_method != method) return false;

        auto pattern_parts = split_path(pattern);
        auto path_parts = split_path(path);

        if (pattern_parts.size() != path_parts.size()) return false;

        params.clear();
        for (size_t i = 0; i < pattern_parts.size(); ++i) {
          if (pattern_parts[i].starts_with('{') && pattern_parts[i].ends_with('}')) {
            std::string param_name = pattern_parts[i].substr(1, pattern_parts[i].length() - 2);
            params[param_name] = path_parts[i];
          }
          else if (pattern_parts[i] != path_parts[i]) {
            return false;
          }
        }
        return true;
      }

    private:
      std::vector<std::string> split_path(const std::string& path) const {
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
    };

    struct router_t {
      std::vector<route_t> routes;

      template<typename Handler>
      void add_route(int method, const std::string& pattern, Handler&& handler) {
        route_t route;
        route.method = method;
        route.pattern = pattern;
        route.handler = [handler = std::forward<Handler>(handler)](const http_request_t& req, http_response_t& res) -> fan::event::task_t {
          co_await handler(req, res);
          };

        routes.push_back(std::move(route));
      }

      template<typename Handler>
      void get(const std::string& pattern, Handler&& handler) {
        add_route(http_method_t::get, pattern, std::forward<Handler>(handler));
      }

      template<typename Handler>
      void post(const std::string& pattern, Handler&& handler) {
        add_route(http_method_t::post, pattern, std::forward<Handler>(handler));
      }

      template<typename Handler>
      void put(const std::string& pattern, Handler&& handler) {
        add_route(http_method_t::put, pattern, std::forward<Handler>(handler));
      }

      template<typename Handler>
      void delete_(const std::string& pattern, Handler&& handler) {
        add_route(http_method_t::delete_, pattern, std::forward<Handler>(handler));
      }

      fan::event::task_value_resume_t<http_response_t> handle(const http_request_t& req) {
        for (const auto& route : routes) {
          http_request_t modified_req = req;
          if (route.matches(req.method, req.path, modified_req.params)) {
            http_response_t res;
            try {
              co_await route.handler(modified_req, res);
            }
            catch (const std::exception& e) {
              res.internal_error("Handler exception: " + std::string(e.what()));
            }
            co_return res;
          }
        }

        http_response_t res;
        co_return res.not_found();
      }
    };

    inline std::unordered_map<std::string, std::string> parse_query_string(const std::string& query_string) {
      std::unordered_map<std::string, std::string> params;
      std::string current_param;
      std::string current_value;
      bool reading_value = false;

      for (char c : query_string) {
        if (c == '=') {
          reading_value = true;
        }
        else if (c == '&') {
          if (!current_param.empty()) {
            params[current_param] = current_value;
          }
          current_param.clear();
          current_value.clear();
          reading_value = false;
        }
        else {
          if (reading_value) {
            current_value += c;
          }
          else {
            current_param += c;
          }
        }
      }

      if (!current_param.empty()) {
        params[current_param] = current_value;
      }

      return params;
    }

    http_request_t parse_http_request(const std::string& raw_request) {
      http_request_t req;
      std::istringstream stream(raw_request);
      std::string line;

      if (!std::getline(stream, line)) {
        return req;
      }

      line.erase(line.find_last_not_of("\r\n") + 1);

      std::istringstream request_line(line);
      std::string method_str, full_path, version;
      request_line >> method_str >> full_path >> version;

      if (method_str == "GET") req.method = http_method_t::get;
      else if (method_str == "POST") req.method = http_method_t::post;
      else if (method_str == "PUT") req.method = http_method_t::put;
      else if (method_str == "DELETE") req.method = http_method_t::delete_;

      req.raw_path = full_path;
      size_t query_pos = full_path.find('?');
      if (query_pos != std::string::npos) {
        req.path = full_path.substr(0, query_pos);
        std::string query_string = full_path.substr(query_pos + 1);
        req.query = parse_query_string(query_string);
      }
      else {
        req.path = full_path;
      }

      while (std::getline(stream, line) && !line.empty() && line != "\r") {
        line.erase(line.find_last_not_of("\r\n") + 1);
        size_t colon_pos = line.find(':');
        if (colon_pos != std::string::npos) {
          std::string key = line.substr(0, colon_pos);
          std::string value = line.substr(colon_pos + 1);
          value.erase(0, value.find_first_not_of(" \t"));
          req.headers[key] = value;
        }
      }

      std::string body_line;
      while (std::getline(stream, body_line)) {
        req.body += body_line + "\n";
      }
      if (!req.body.empty()) {
        req.body.pop_back();
      }

      return req;
    }

    struct http_server_t {
      router_t router;

      template<typename Handler>
      void get(const std::string& pattern, Handler&& handler) {
        router.get(pattern, std::forward<Handler>(handler));
      }

      template<typename Handler>
      void post(const std::string& pattern, Handler&& handler) {
        router.post(pattern, std::forward<Handler>(handler));
      }

      template<typename Handler>
      void put(const std::string& pattern, Handler&& handler) {
        router.put(pattern, std::forward<Handler>(handler));
      }

      template<typename Handler>
      void delete_(const std::string& pattern, Handler&& handler) {
        router.delete_(pattern, std::forward<Handler>(handler));
      }

      fan::event::task_t listen(const listen_address_t& address) {
        co_await tcp_server_listen(address, [this](tcp_t& client) -> fan::event::task_t {
          bool keep_alive = true;

          while (keep_alive) {
            std::string request_data;
            bool headers_complete = false;
            size_t content_length = 0;
            http_response_t error_response;
            bool has_error = false;

            try {
              while (!headers_complete) {
                auto msg = co_await client.read_raw();
                if (msg.status < 0) {
                  co_return;
                }

                std::string chunk(msg.buffer.begin(), msg.buffer.end());
                request_data += chunk;

                size_t header_end = request_data.find("\r\n\r\n");
                if (header_end != std::string::npos) {
                  headers_complete = true;

                  size_t cl_pos = request_data.find("Content-Length:");
                  if (cl_pos != std::string::npos) {
                    size_t cl_start = cl_pos + 15;
                    size_t cl_end = request_data.find("\r\n", cl_start);
                    std::string cl_str = request_data.substr(cl_start, cl_end - cl_start);
                    cl_str.erase(0, cl_str.find_first_not_of(" \t"));
                    content_length = std::stoull(cl_str);
                  }

                  size_t current_body_length = request_data.length() - (header_end + 4);
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

              http_request_t request = parse_http_request(request_data);

              auto conn_header = request.header("Connection");
              if (conn_header.find("close") != std::string::npos) {
                keep_alive = false;
              }

              http_response_t response = co_await router.handle(request);

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
    };

    struct response_t {
      int status_code = 200;
      std::unordered_map<std::string, std::string> headers;
      std::string body;

      response_t& status(int code) {
        status_code = code;
        return *this;
      }

      response_t& header(const std::string& key, const std::string& value) {
        headers[key] = value;
        return *this;
      }

      response_t& json(const fan::json& data) {
        body = data.dump();
        headers["Content-Type"] = "application/json";
        return *this;
      }

      response_t& text(const std::string& data) {
        body = data;
        headers["Content-Type"] = "text/plain";
        return *this;
      }
    };

    struct request_t {
      int method;
      std::string path;
      std::unordered_map<std::string, std::string> headers;
      std::unordered_map<std::string, std::string> params;
      std::unordered_map<std::string, std::string> query;
      std::string body;

      std::string header(const std::string& name) const {
        auto it = headers.find(name);
        return it != headers.end() ? it->second : "";
      }

      std::expected<fan::json, CURLcode> json() const {
        if (body.empty()) {
          return std::unexpected(CURLE_BAD_CONTENT_ENCODING);
        }
        try {
          return fan::json::parse(body);
        }
        catch (...) {
          return std::unexpected(CURLE_BAD_CONTENT_ENCODING);
        }
      }
    };

    struct http_config_t {
      bool verify_ssl = true;
      bool follow_redirects = true;
      long timeout_seconds = 30;
      bool enable_http2 = true;
      bool keep_alive = false;
      std::string user_agent = "libcurl-client/1.0";
      std::string ca_cert_path = "curl-ca-bundle.crt";
    };

    struct curl_data_t {
      std::string body;
      std::string headers;
      long status_code = 0;
    };

    struct async_http_context_t {
      CURLM* multi_handle;
      uv_timer_t timeout_timer;
      int active_handles;
      bool timer_started;
      bool callbacks_set;

      static async_http_context_t& instance() {
        static async_http_context_t ctx;
        return ctx;
      }

      async_http_context_t() : multi_handle(nullptr), active_handles(0),
        timer_started(false), callbacks_set(false) {
        multi_handle = curl_multi_init();
        uv_timer_init(fan::event::get_loop(), &timeout_timer);
        timeout_timer.data = this;
      }

      ~async_http_context_t() {
        if (multi_handle) {
          curl_multi_cleanup(multi_handle);
        }
      }
    };

    struct async_http_request_t {
      CURL* easy_handle;
      struct curl_slist* headers_list;
      std::coroutine_handle<> co_handle;
      curl_data_t response_data;
      CURLcode result;
      bool completed;
      std::string post_data; // Keep post data alive

      async_http_request_t() : easy_handle(nullptr), headers_list(nullptr),
        completed(false), result(CURLE_OK) {
        easy_handle = curl_easy_init();
        if (easy_handle) {
          curl_easy_setopt(easy_handle, CURLOPT_WRITEFUNCTION, write_body);
          curl_easy_setopt(easy_handle, CURLOPT_WRITEDATA, &response_data);
          curl_easy_setopt(easy_handle, CURLOPT_HEADERFUNCTION, write_headers);
          curl_easy_setopt(easy_handle, CURLOPT_HEADERDATA, &response_data);
          curl_easy_setopt(easy_handle, CURLOPT_PRIVATE, this);
        }
      }

      ~async_http_request_t() {
        if (headers_list) {
          curl_slist_free_all(headers_list);
        }
        if (easy_handle) {
          curl_easy_cleanup(easy_handle);
        }
      }

      async_http_request_t(const async_http_request_t&) = delete;
      async_http_request_t& operator=(const async_http_request_t&) = delete;
      async_http_request_t(async_http_request_t&& other) noexcept
        : easy_handle(other.easy_handle), headers_list(other.headers_list),
        co_handle(other.co_handle), response_data(std::move(other.response_data)),
        result(other.result), completed(other.completed),
        post_data(std::move(other.post_data)) {
        other.easy_handle = nullptr;
        other.headers_list = nullptr;
        if (easy_handle) {
          curl_easy_setopt(easy_handle, CURLOPT_WRITEFUNCTION, write_body);
          curl_easy_setopt(easy_handle, CURLOPT_WRITEDATA, &response_data);
          curl_easy_setopt(easy_handle, CURLOPT_HEADERFUNCTION, write_headers);
          curl_easy_setopt(easy_handle, CURLOPT_HEADERDATA, &response_data);
          curl_easy_setopt(easy_handle, CURLOPT_PRIVATE, this);
        }
      }

      static size_t write_body(void* data, size_t size, size_t nmemb, curl_data_t* response) {
        size_t total = size * nmemb;
        response->body.append(static_cast<char*>(data), total);
        return total;
      }

      static size_t write_headers(void* data, size_t size, size_t nmemb, curl_data_t* response) {
        size_t total = size * nmemb;
        response->headers.append(static_cast<char*>(data), total);
        return total;
      }

      bool await_ready() const { return completed; }

      void await_suspend(std::coroutine_handle<> h) {
        co_handle = h;
        auto& ctx = async_http_context_t::instance();

        // Set up the multi interface callbacks if not already done
        if (!ctx.callbacks_set) {
          curl_multi_setopt(ctx.multi_handle, CURLMOPT_SOCKETFUNCTION, socket_callback);
          curl_multi_setopt(ctx.multi_handle, CURLMOPT_SOCKETDATA, &ctx);
          curl_multi_setopt(ctx.multi_handle, CURLMOPT_TIMERFUNCTION, timer_function);
          curl_multi_setopt(ctx.multi_handle, CURLMOPT_TIMERDATA, &ctx);
          ctx.callbacks_set = true;
        }

        curl_multi_add_handle(ctx.multi_handle, easy_handle);

        // Kick off the multi interface
        int running_handles;
        curl_multi_socket_action(ctx.multi_handle, CURL_SOCKET_TIMEOUT, 0, &running_handles);
        check_multi_info(&ctx);
      }

      std::expected<response_t, CURLcode> await_resume() {
        if (result != CURLE_OK) {
          return std::unexpected(result);
        }

        response_t response;
        curl_easy_getinfo(easy_handle, CURLINFO_RESPONSE_CODE, &response_data.status_code);
        response.status_code = static_cast<int>(response_data.status_code);
        response.body = std::move(response_data.body);
        parse_response_headers(response_data.headers, response);

        return response;
      }

    private:
      static void timer_callback(uv_timer_t* handle) {
        auto* ctx = static_cast<async_http_context_t*>(handle->data);
        ctx->timer_started = false;

        int running_handles;
        curl_multi_socket_action(ctx->multi_handle, CURL_SOCKET_TIMEOUT, 0, &running_handles);
        check_multi_info(ctx);
      }

      static void poll_callback(uv_poll_t* handle, int status, int events) {
        auto* ctx = &async_http_context_t::instance();
        int flags = 0;
        if (events & UV_READABLE) flags |= CURL_CSELECT_IN;
        if (events & UV_WRITABLE) flags |= CURL_CSELECT_OUT;

        curl_socket_t sockfd = static_cast<curl_socket_t>(reinterpret_cast<intptr_t>(handle->data));

        int running_handles;
        curl_multi_socket_action(ctx->multi_handle, sockfd, flags, &running_handles);
        check_multi_info(ctx);
      }

      static int socket_callback(CURL* easy, curl_socket_t sockfd, int what, void* userp, void* socketp) {
        auto* ctx = static_cast<async_http_context_t*>(userp);

        if (what == CURL_POLL_REMOVE) {
          if (socketp) {
            uv_poll_t* poll_handle = static_cast<uv_poll_t*>(socketp);
            uv_poll_stop(poll_handle);
            uv_close(reinterpret_cast<uv_handle_t*>(poll_handle), [](uv_handle_t* handle) {
              delete reinterpret_cast<uv_poll_t*>(handle);
              });
          }
          return 0;
        }

        uv_poll_t* poll_handle = static_cast<uv_poll_t*>(socketp);
        if (!poll_handle) {
          poll_handle = new uv_poll_t;
          uv_poll_init_socket(fan::event::get_loop(), poll_handle, sockfd);
          poll_handle->data = reinterpret_cast<void*>(static_cast<intptr_t>(sockfd));
          curl_multi_assign(ctx->multi_handle, sockfd, poll_handle);
        }

        int events = 0;
        if (what & CURL_POLL_IN) events |= UV_READABLE;
        if (what & CURL_POLL_OUT) events |= UV_WRITABLE;

        uv_poll_start(poll_handle, events, poll_callback);
        return 0;
      }

      static int timer_function(CURLM* multi, long timeout_ms, void* userp) {
        auto* ctx = static_cast<async_http_context_t*>(userp);

        if (timeout_ms < 0) {
          if (ctx->timer_started) {
            uv_timer_stop(&ctx->timeout_timer);
            ctx->timer_started = false;
          }
        }
        else {
          if (timeout_ms == 0) timeout_ms = 1;  // minimum 1ms
          uv_timer_start(&ctx->timeout_timer, timer_callback, timeout_ms, 0);
          ctx->timer_started = true;
        }
        return 0;
      }

      static void check_multi_info(async_http_context_t* ctx) {
        CURLMsg* msg;
        int msgs_left;

        while ((msg = curl_multi_info_read(ctx->multi_handle, &msgs_left))) {
          if (msg->msg == CURLMSG_DONE) {
            CURL* easy_handle = msg->easy_handle;
            async_http_request_t* request;
            curl_easy_getinfo(easy_handle, CURLINFO_PRIVATE, &request);

            request->result = msg->data.result;
            request->completed = true;

            curl_multi_remove_handle(ctx->multi_handle, easy_handle);

            if (request->co_handle) {
              request->co_handle.resume();
            }
          }
        }
      }

      void parse_response_headers(const std::string& header_str, response_t& response) {
        size_t pos = 0;
        while (pos < header_str.length()) {
          size_t end = header_str.find('\n', pos);
          if (end == std::string::npos) {
            end = header_str.length();
          }

          std::string line = header_str.substr(pos, end - pos);
          if (!line.empty() && line.back() == '\r') {
            line.pop_back();
          }

          size_t colon = line.find(':');
          if (colon != std::string::npos && colon > 0) {
            std::string key = line.substr(0, colon);
            std::string value = line.substr(colon + 1);

            if (!value.empty() && value.front() == ' ') {
              value = value.substr(1);
            }

            response.headers[key] = value;
          }

          pos = end + 1;
        }
      }
    };

    struct curl_global_manager_t {
      curl_global_manager_t() {
        curl_global_init(CURL_GLOBAL_DEFAULT);
      }
      ~curl_global_manager_t() {
        curl_global_cleanup();
      }
      static void init() {
        static curl_global_manager_t curl_manager;
      }
    };
    std::string strerror(CURLcode code) {
      return curl_easy_strerror(code);
    }

    struct async_http_client_t {
      std::string base_url;
      http_config_t config;

      async_http_client_t(const std::string& url, const http_config_t& cfg = {})
        : base_url(url), config(cfg) {
        static std::once_flag curl_init_flag;
        std::call_once(curl_init_flag, []() {
          curl_global_manager_t::init();
          });
      }

      fan::event::task_value_resume_t<std::expected<response_t, std::string>> request(
        const std::string& method,
        const std::string& path,
        const std::unordered_map<std::string, std::string>& headers = {},
        const std::string& body = ""
      ) {
        async_http_request_t request;
        std::string url = base_url + path;

        if (curl_easy_setopt(request.easy_handle, CURLOPT_URL, url.c_str()) != CURLE_OK) {
          co_return std::unexpected("Failed to set URL");
        }

        curl_easy_setopt(request.easy_handle, CURLOPT_FOLLOWLOCATION, config.follow_redirects ? 1L : 0L);
        curl_easy_setopt(request.easy_handle, CURLOPT_TIMEOUT, config.timeout_seconds);
        curl_easy_setopt(request.easy_handle, CURLOPT_USERAGENT, config.user_agent.c_str());
        curl_easy_setopt(request.easy_handle, CURLOPT_SSL_VERIFYPEER, config.verify_ssl ? 1L : 0L);
        curl_easy_setopt(request.easy_handle, CURLOPT_SSL_VERIFYHOST, config.verify_ssl ? 2L : 0L);

        if (config.enable_http2) {
          curl_easy_setopt(request.easy_handle, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_2_0);
        }
        if (config.keep_alive) {
          curl_easy_setopt(request.easy_handle, CURLOPT_TCP_KEEPALIVE, 1L);
        }

        curl_easy_setopt(request.easy_handle, CURLOPT_CAINFO, config.ca_cert_path.c_str());
        curl_easy_setopt(request.easy_handle, CURLOPT_ACCEPT_ENCODING, "");

        for (const auto& [key, value] : headers) {
          std::string header_str = key + ": " + value;
          request.headers_list = curl_slist_append(request.headers_list, header_str.c_str());
        }
        if (request.headers_list) {
          curl_easy_setopt(request.easy_handle, CURLOPT_HTTPHEADER, request.headers_list);
        }

        if (method == "POST") {
          curl_easy_setopt(request.easy_handle, CURLOPT_POST, 1L);
          if (!body.empty()) {
            request.post_data = body;
            curl_easy_setopt(request.easy_handle, CURLOPT_POSTFIELDS, request.post_data.c_str());
            curl_easy_setopt(request.easy_handle, CURLOPT_POSTFIELDSIZE, request.post_data.length());
          }
        }
        else if (method == "PUT") {
          curl_easy_setopt(request.easy_handle, CURLOPT_CUSTOMREQUEST, "PUT");
          if (!body.empty()) {
            request.post_data = body;
            curl_easy_setopt(request.easy_handle, CURLOPT_POSTFIELDS, request.post_data.c_str());
            curl_easy_setopt(request.easy_handle, CURLOPT_POSTFIELDSIZE, request.post_data.length());
          }
        }
        else if (method == "DELETE") {
          curl_easy_setopt(request.easy_handle, CURLOPT_CUSTOMREQUEST, "DELETE");
        }
        else {
          curl_easy_setopt(request.easy_handle, CURLOPT_HTTPGET, 1L);
        }

        auto result = co_await request;
        if (!result) {
          co_return std::unexpected(fan::network::strerror(result.error()));
        }

        co_return result.value();
      }

      fan::event::task_value_resume_t<std::expected<response_t, std::string>> get(const std::string& path) {
        co_return co_await request("GET", path);
      }

      fan::event::task_value_resume_t<std::expected<response_t, std::string>> get(
        const std::string& path,
        const std::unordered_map<std::string, std::string>& headers
      ) {
        co_return co_await request("GET", path, headers);
      }

      fan::event::task_value_resume_t<std::expected<response_t, std::string>> post(
        const std::string& path,
        const fan::json& data
      ) {
        std::unordered_map<std::string, std::string> headers = {
            {"Content-Type", "application/json"}
        };
        co_return co_await request("POST", path, headers, data.dump());
      }

      fan::event::task_value_resume_t<std::expected<response_t, std::string>> put(
        const std::string& path,
        const fan::json& data
      ) {
        std::unordered_map<std::string, std::string> headers = {
            {"Content-Type", "application/json"}
        };
        co_return co_await request("PUT", path, headers, data.dump());
      }

      fan::event::task_value_resume_t<std::expected<response_t, std::string>> delete_(const std::string& path) {
        co_return co_await request("DELETE", path);
      }
    };

    using http_client_t = async_http_client_t;
    using https_client_t = async_http_client_t;
    using http2_client_t = async_http_client_t;

    // -------------------------------HTTP/REST-------------------------------

  }
}

fan::network::client_handler_t::nr_t fan::network::client_handler_t::add_client() {
  nr_t nr = client_list.NewNodeLast();
  client_list[nr] = new fan::network::tcp_t;
  client_list[nr]->nr = nr;
  return nr;
}