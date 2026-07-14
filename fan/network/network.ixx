module;

#include <fan/utility.h>

#if defined(FAN_NETWORK)
  #include <uv.h>
  #if defined(__clang__) || !defined(__GNUC__) || defined(FAN_NETWORK_ENABLE_HTTP_ON_GCC)
    #define FAN_NETWORK_HTTP_ENABLED
  #endif
  #if defined(FAN_NETWORK_HTTP_ENABLED) && \
    (defined(__clang__) || !defined(__GNUC__) || defined(FAN_NETWORK_ENABLE_CURL_ON_GCC))
    #define FAN_NETWORK_CURL_ENABLED
    #pragma comment(lib, "libcurl.a")
    #pragma comment(lib, "libcurl.dll.a")
  #endif
#endif


export module fan.network;

import std;

#if defined(FAN_NETWORK)

import fan.utility;
import fan.event.types;
import fan.print.error;
import fan.types.json;
import fan.memory;
import fan.event;

export namespace fan {
  namespace network {
    struct tcp_t;
    struct udp_t;

    struct getaddrinfo_t {
      struct getaddrinfo_data_t;
      std::unique_ptr<getaddrinfo_data_t> data;

      getaddrinfo_t(const char* node, const char* service, struct addrinfo* hints = nullptr);
      bool await_ready() const;
      void await_suspend(std::coroutine_handle<> h);
      int await_resume() const;
      ~getaddrinfo_t();
    };

    // -------------------------------TCP-------------------------------
    struct connector_t {
      struct connector_data_t;
      std::unique_ptr<connector_data_t> data;

      connector_t(const tcp_t& tcp, const std::string& ip, int port);
      connector_t(const tcp_t& tcp, const getaddrinfo_t& info);

      connector_t(connector_t&&) noexcept;
      connector_t& operator=(connector_t&&) noexcept;
      ~connector_t();
      
      bool await_ready() const;
      void await_suspend(std::coroutine_handle<> h);
      int await_resume();
    };

    struct listener_t {
      std::shared_ptr<uv_stream_t> stream;
      std::coroutine_handle<> co_handle;
      int status;
      bool ready;

      listener_t(const tcp_t& tcp, int backlog);
      listener_t(listener_t&& l);

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

      explicit operator bool() { return status == error_code::ok; }
      operator ssize_t() const { return status; }
      operator const std::string() const { return { buffer.begin(), buffer.end() }; }
      operator const buffer_t&() const { return buffer; }
    };

    struct raw_reader_t {
      std::shared_ptr<uv_stream_t> stream;
      buffer_t buf;
      ssize_t nread{ 0 };
      std::coroutine_handle<> co_handle;

      raw_reader_t(const tcp_t& tcp);
      raw_reader_t(const raw_reader_t&) = delete;
      raw_reader_t(raw_reader_t&& r) noexcept;

      void stop();
      int start() noexcept;
      bool await_ready() const { return nread > 0; }
      void await_suspend(std::coroutine_handle<> h) { co_handle = h; }
      data_t await_resume();
      ~raw_reader_t();
    };

    struct raw_writer_t {
      struct writer_data_t;
      std::unique_ptr<writer_data_t> data;

      raw_writer_t(const tcp_t& tcp);
      raw_writer_t(raw_writer_t&&) noexcept;
      raw_writer_t& operator=(raw_writer_t&&) noexcept;

      int write(const buffer_t& some_data);
      bool await_ready() const noexcept;
      void await_suspend(std::coroutine_handle<> h) noexcept;
      int await_resume() noexcept;
      ~raw_writer_t();
    };

    struct message_t {
      buffer_t buffer;
      ssize_t status;
      bool done;

      explicit operator bool() { return status == error_code::ok; }
      operator ssize_t() const { return status; }
      operator const std::string () const { return { buffer.begin(), buffer.end() }; }
      operator const buffer_t& () const { return buffer; }
    };

    class ring_buffer_t {
    private:
      std::vector<char> buffer_;
      std::size_t head_ = 0;
      std::size_t tail_ = 0;
      std::size_t size_ = 0;
      std::size_t capacity_;

    public:
      explicit ring_buffer_t(std::size_t capacity = 64 * 1024)
        : buffer_(capacity), capacity_(capacity) {
      }

      void push_back(const char* data, std::size_t len);
      void consume(std::size_t len);
      std::size_t size() const { return size_; }
      bool empty() const { return size_ == 0; }
      void peek(char* dest, std::size_t len) const;
      std::pair<const char*, std::size_t> get_contiguous() const;

    private:
      void grow(std::size_t new_capacity);
    };

    struct reader_t {
      std::shared_ptr<uv_stream_t> stream;
      ring_buffer_t accumulated_buf;
      std::vector<char> temp_buf;
      ssize_t nread{ 0 };
      std::coroutine_handle<> co_handle;
      bool is_reading = false;

      std::uint64_t expected_size{ 0 };
      std::uint64_t bytes_read{ 0 };
      bool reading_header{ true };
      bool is_raw_read{ false };
      bool is_fixed_size_read{ false };

      static constexpr std::size_t header_size = sizeof(std::uint64_t);
      static constexpr std::size_t default_buffer_size = 64 * 1024;

      reader_t(const tcp_t& tcp);
      reader_t(const reader_t&) = delete;
      reader_t(reader_t&& r) noexcept;

      void setup_fixed_size_read(ssize_t len);
      void setup_raw_read();
      void setup_header_read();
      void stop();
      int start() noexcept;
      bool await_ready() const;
      void await_suspend(std::coroutine_handle<> h) { co_handle = h; }
      message_t await_resume();
      ~reader_t();
    };

    template<typename T>
    struct typed_message_t {
      T data;
      ssize_t status;
      bool done;

      explicit operator bool() const { return status == error_code::ok; }
      operator ssize_t() const { return status; }
      const T& operator*() const { return data; }
      T& operator*() { return data; }
      const T* operator->() const { return &data; }
      T* operator->() { return &data; }
    };

    template<typename T>
    struct typed_reader_t {
      reader_t& base_reader;

      typed_reader_t(reader_t& reader) : base_reader(reader) {
        base_reader.setup_fixed_size_read(sizeof(T));
      }
      bool await_ready() const { return base_reader.await_ready(); }
      void await_suspend(std::coroutine_handle<> h) { base_reader.await_suspend(h); }
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

      writer_t(const tcp_t& tcp);
      writer_t(const writer_t&) = delete;
      writer_t& operator=(const writer_t&) = delete;
      writer_t(writer_t&&) = default;
      writer_t& operator=(writer_t&&) = default;

      int write(const buffer_t& user_data);
      int write(const message_t& msg) { return write(msg.buffer); }

      bool await_ready() const noexcept { return raw_writer.await_ready(); }
      void await_suspend(std::coroutine_handle<> h) noexcept { raw_writer.await_suspend(h); }
      int await_resume() noexcept { return raw_writer.await_resume(); }
    };

    struct listen_address_t {
      std::string ip = "0.0.0.0";
      std::uint16_t port = 0;
    };

    struct client_handler_t {
#define BLL_set_SafeNext 1
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_prefix client_list
#include <fan/fan_bll_preset.h>
#define BLL_set_Link 1
#define BLL_set_type_node uint32_t
#define BLL_set_NodeDataType fan::network::tcp_t*
#define BLL_set_CPP_CopyAtPointerChange 1
#include <BLL/BLL.h>

      using nr_t = client_list_t::nr_t;
      client_list_t client_list;

      nr_t add_client();
      void remove_client(nr_t nr) { client_list.unlrec(nr); }

      tcp_t& get_client(nr_t id) { return *client_list[id]; }
      const tcp_t& get_client(nr_t id) const { return const_cast<client_handler_t*>(this)->get_client(id); }

      tcp_t& operator[](nr_t id) { return get_client(id); }
      const tcp_t& operator[](nr_t id) const { return get_client(id); }
      std::uint32_t amount_of_connections = 128;
    };

    client_handler_t& get_client_handler();

    using tcp_id_t = client_handler_t::nr_t;

    struct tcp_t {
      tcp_id_t nr;
      std::string tag;
      fan::event::task_t task;
      using listen_cb_t = std::function<fan::event::task_t(tcp_t&)>;

      std::shared_ptr<uv_tcp_t> socket;

      operator tcp_id_t () const { return nr; }

      struct tcp_deleter_t {
        void operator()(void* p) const;
      };

      tcp_t();
      ~tcp_t();
      tcp_t(const tcp_t&) = delete;
      tcp_t& operator=(const tcp_t&) = delete;
      tcp_t(tcp_t&&) = default;
      tcp_t& operator=(tcp_t&&) = delete;

      fan::event::error_code_t accept(tcp_t& client) noexcept;
      fan::event::error_code_t bind(const std::string& ip, int port) noexcept;
      fan::event::task_t listen(const listen_address_t& address, listen_cb_t lambda, bool bind = false);

      connector_t connect(const std::string& ip, int port) { return connector_t{ *this, ip, port }; }
      connector_t connect(const getaddrinfo_t& info) { return connector_t{ *this, info }; }

      void set_tag(const std::string& t) { tag = t; }
      const std::string& get_tag() const { return tag; }

      mutable std::unique_ptr<reader_t> reader;
      reader_t& get_reader() const;

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

    fan::event::task_t tcp_server_listen(listen_address_t address, tcp_t::listen_cb_t lambda);
    fan::event::task_t tcp_listen(listen_address_t address, tcp_t::listen_cb_t lambda);

    // -------------------------------UDP-------------------------------
    struct socket_address_t {
      union {
        struct sockaddr_storage storage;
        struct sockaddr_in ipv4;
        struct sockaddr_in6 ipv6;
        struct sockaddr generic;
      } addr;

      socket_address_t();
      socket_address_t(const std::string& ip, int port);
      socket_address_t(const struct sockaddr* sa);

      std::string get_ip() const;
      int get_port() const;
      const struct sockaddr* sockaddr_ptr() const;
      int sockaddr_len() const;
    };

    struct udp_datagram_t {
      buffer_t data;
      socket_address_t sender;
      ssize_t status;

      udp_datagram_t() : status(0) {}
      udp_datagram_t(const buffer_t& msg, const socket_address_t& addr, ssize_t stat = 0)
        : data(msg), sender(addr), status(stat) {}

      operator bool() const { return status >= 0; }
      operator std::string() const { return std::string(data.begin(), data.end()); }
      operator const buffer_t() const { return data; }

      const buffer_t& message() const { return data; }
      std::string sender_ip() const { return sender.get_ip(); }
      int sender_port() const { return sender.get_port(); }
      bool is_error() const { return status < 0; }
      bool is_empty() const { return data.empty(); }
      std::size_t size() const { return data.size(); }
    };

    struct udp_send_t {
      struct send_data_t;
      std::unique_ptr<send_data_t> data;

      udp_send_t(const udp_t& udp, const buffer_t& message, const socket_address_t& addr);
      udp_send_t(const udp_t& udp, const buffer_t& message, const std::string& ip, int port);
      udp_send_t(const udp_t& udp, const std::string& message, const std::string& ip, int port);
      udp_send_t(const udp_t& udp, const std::string& message, const socket_address_t& addr);

      udp_send_t(udp_send_t&&) noexcept;
      udp_send_t& operator=(udp_send_t&&) noexcept;

      bool await_ready() const;
      void await_suspend(std::coroutine_handle<> h);
      int await_resume();
      ~udp_send_t();
    };

    struct udp_recv_t {
      std::shared_ptr<uv_udp_t> socket;
      std::coroutine_handle<> co_handle;
      udp_datagram_t datagram;
      bool ready{ false };
      bool receiving{ false };

      udp_recv_t(const udp_t& udp);
      udp_recv_t(const udp_recv_t&) = delete;
      udp_recv_t(udp_recv_t&& r) noexcept;

      int start();
      void stop();

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
      ~udp_recv_t();
    };

    struct udp_recvfrom_t {
      std::shared_ptr<uv_udp_t> socket;
      std::coroutine_handle<> co_handle;
      udp_datagram_t datagram;
      bool ready{ false };
      bool receiving{ false };
      bool filter_sender{ false };
      socket_address_t expected_sender;

      udp_recvfrom_t(const udp_t& udp);

      void set_expected_sender(const socket_address_t& sender);
      void set_expected_sender(const std::string& ip, int port);
      bool matches_expected_sender(const socket_address_t& actual_sender) const;

      udp_recvfrom_t(const udp_recvfrom_t&) = delete;
      udp_recvfrom_t(udp_recvfrom_t&& r) noexcept;

      int start();
      void stop();

      bool await_ready() const { return ready; }
      void await_suspend(std::coroutine_handle<> h);
      udp_datagram_t await_resume();
      ~udp_recvfrom_t();
    };

    struct udp_t {
      using recv_cb_t = std::function<fan::event::task_t(udp_t&, const udp_datagram_t&)>;
      std::shared_ptr<uv_udp_t> socket;
      fan::event::task_t task;

      struct udp_deleter_t {
        void operator()(void* p) const;
      };

      udp_t();
      udp_t(const udp_t&) = delete;
      udp_t& operator=(const udp_t&) = delete;
      udp_t(udp_t&&) = default;
      udp_t& operator=(udp_t&&) = delete;

      fan::event::error_code_t bind(const std::string& ip, int port, unsigned int flags = 0) noexcept;

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
      fan::event::runv_t<udp_datagram_t> recvfrom(const std::string& ip, int port) {
        auto recvfrom = udp_recvfrom_t{ *this };
        recvfrom.set_expected_sender(ip, port);
        co_return co_await recvfrom;
      }
      fan::event::runv_t<udp_datagram_t> recvfrom(const socket_address_t& addr) {
        auto recvfrom = udp_recvfrom_t{ *this };
        recvfrom.set_expected_sender(addr);
        co_return co_await recvfrom;
      }

      fan::event::task_t listen(const listen_address_t& address, recv_cb_t callback, bool bind_ = false);

      socket_address_t get_sockname() const;

      fan::event::error_code_t set_broadcast(bool enable) noexcept;
      fan::event::error_code_t set_ttl(int ttl) noexcept;
      fan::event::error_code_t join_multicast(const std::string& multicast_addr, const std::string& interface_addr = "") noexcept;
      fan::event::error_code_t leave_multicast(const std::string& multicast_addr, const std::string& interface_addr = "") noexcept;
      fan::event::error_code_t set_multicast_ttl(int ttl) noexcept;
      fan::event::error_code_t set_multicast_interface(const std::string& interface_addr) noexcept;
      fan::event::error_code_t set_multicast_loop(bool enable) noexcept;

      bool is_bound() const {
        auto addr = get_sockname();
        return addr.get_port() != 0;
      }
    };

    fan::event::task_t udp_listen(const listen_address_t& address, udp_t::recv_cb_t callback);

    // -------------------------------UDP-------------------------------

    struct keep_alive_config_t {
      static constexpr std::size_t timer_step_limit = 7;
      std::array<int, timer_step_limit> timer_steps{ 1, 1, 2, 2, 4, 4, 4 };
      std::size_t current_step = 0;
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
              fan::throw_error("Keep alive send failed: ", e.what());
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

      std::size_t get_current_step() const {
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
        fan::throw_error("TCP keep alive timeout");
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
      void start() { timer->start(); }
      void stop() { timer->stop(); }
      void reset() { timer->reset(); }
      bool is_running() const { return timer->is_running(); }
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
        fan::throw_error("UDP keep alive timeout");
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
        if (timer == nullptr) { return; }
        timer->start();
      }
      void stop() {
        if (timer == nullptr) { return; }
        timer->stop();
      }
      void reset() {
        if (timer == nullptr) { return; }
        timer->reset();
      }
      bool is_running() const { return timer->is_running(); }
      void update_server_address(const std::string& server_ip, int server_port) {
        server_address = socket_address_t(server_ip, server_port);
      }
      void update_server_address(const socket_address_t& server_addr) {
        server_address = server_addr;
      }
    };

#if defined(FAN_NETWORK_HTTP_ENABLED)
    // -------------------------------HTTP/REST-------------------------------
    namespace http {
      struct method_t {
        enum {
          get = 0, post, put, delete_, patch, head, options
        };
      };

      struct status_t {
        enum {
          ok = 200, created = 201, no_content = 204, bad_request = 400, unauthorized = 401,
          forbidden = 403, not_found = 404, method_not_allowed = 405, internal_server_error = 500,
          not_implemented = 501, service_unavailable = 503
        };
      };

      struct error_t {
        enum {
          invalid_json = 1, invalid_param, connection_failed, timeout, parse_failed,
          not_found_error, database_error, validation_error
        };
        int code;
        std::string message;

        error_t(int c, std::string msg) : code(c), message(std::move(msg)) {}
      };

      constexpr const char* CRLF = "\r\n";
      constexpr const char* HEADER_END = "\r\n\r\n";
      constexpr const char* VERSION = "HTTP/1.1";

      inline std::string build_get_request(const std::string& path, const std::string& host, bool keep_alive = false) {
        return "GET " + path + " " + VERSION + CRLF +
          "Host: " + host + CRLF +
          "Connection: " + (keep_alive ? "keep-alive" : "close") + CRLF + CRLF;
      }

      inline std::string build_post_request(const std::string& path, const std::string& host, const std::string& body, bool keep_alive = false) {
        return "POST " + path + " " + VERSION + CRLF +
          "Host: " + host + CRLF +
          "Content-Type: application/json" + CRLF +
          "Content-Length: " + std::to_string(body.length()) + CRLF +
          "Connection: " + 
          (keep_alive ? "keep-alive" : "close") + CRLF + CRLF + body;
      }

      inline std::string build_get_request_with_headers(const std::string& path, const std::string& host,
        const std::unordered_map<std::string, std::string>& headers,
        bool keep_alive = false) {
        std::string request = "GET " + path + " " + VERSION + CRLF +
          "Host: " + host + CRLF;
        for (const auto& [key, value] : headers) {
          request += key + ": " + value + CRLF;
        }

        request += "Connection: " + (keep_alive ? std::string("keep-alive") : std::string("close")) + CRLF + CRLF;
        return request;
      }

      inline std::expected<fan::json, error_t> parse_json_simple(const std::string& json_str) {
        if (json_str.empty()) {
          return std::unexpected(error_t {error_t::invalid_json, "Empty JSON string"});
        }

        try {
          return fan::json::parse(json_str);
        }
        catch (...) {
          return std::unexpected(error_t {error_t::invalid_json, "Invalid JSON format"});
        }
      }

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

      struct request_t {
        int method;
        std::string path;
        std::string raw_path;
        std::unordered_map<std::string, std::string> headers;
        std::unordered_map<std::string, std::string> params;
        std::unordered_map<std::string, std::string> query;
        std::string body;

        template<typename T>
        std::expected<T, error_t> param(const std::string& name) const {
          auto it = params.find(name);
          if (it == params.end()) {
            return std::unexpected(error_t {error_t::invalid_param, "Parameter '" + name + "' not found"});
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
            return std::unexpected(error_t {error_t::invalid_param, "Invalid parameter format for '" + name + "'"});
          }

          return std::unexpected(error_t {error_t::invalid_param, "Unsupported parameter type"});
        }

        std::expected<fan::json, error_t> json() const {
          if (body.empty()) {
            return std::unexpected(error_t {error_t::invalid_json, "Empty request body"});
          }
          return parse_json_simple(body);
        }

        std::string header(const std::string& name) const {
          auto it = headers.find(name);
          return it != headers.end() ? it->second : "";
        }

        template<typename T>
        std::expected<T, error_t> query_param(const std::string& name) const {
          auto it = query.find(name);
          if (it == query.end()) {
            return std::unexpected(error_t {error_t::invalid_param, "Query parameter '" + name + "' not found"});
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
            return std::unexpected(error_t {error_t::invalid_param, "Invalid query parameter format for '" + name + "'"});
          }

          return std::unexpected(error_t {error_t::invalid_param, "Unsupported query parameter type"});
        }
      };

      struct response_t {
        int status_code = status_t::ok;
        std::unordered_map<std::string, std::string> headers;
        std::string body;

        response_t& status(int code);
        response_t& header(const std::string& key, const std::string& value);
        response_t& json(const fan::json& data);
        response_t& text(const std::string& data);
        response_t& html(const std::string& data);
        response_t& ok(const fan::json& data = {});
        response_t& created(const fan::json& data = {});
        response_t& not_found(const std::string& message = "Not Found");
        response_t& bad_request(const std::string& message = "Bad Request");
        response_t& internal_error(const std::string& message = "Internal Server Error");
        response_t& error(const error_t& err);
        std::string to_string() const;
      };

      using async_handler_t = std::function<fan::event::task_t(const request_t&, response_t&)>;

      struct route_t {
        int method;
        std::string pattern;
        async_handler_t handler;

        bool matches(int req_method, const std::string& path, std::unordered_map<std::string, std::string>& params) const;

      private:
        std::vector<std::string> split_path(const std::string& path) const;
      };

      struct router_t {
        std::vector<route_t> routes;

        template<typename Handler>
        void add_route(int method, const std::string& pattern, Handler&& handler) {
          route_t route;
          route.method = method;
          route.pattern = pattern;
          route.handler = [handler = std::forward<Handler>(handler)](const request_t& req, response_t& res) -> fan::event::task_t {
            co_await handler(req, res);
          };

          routes.push_back(std::move(route));
        }

        template<typename Handler>
        void get(const std::string& pattern, Handler&& handler) {
          add_route(method_t::get, pattern, std::forward<Handler>(handler));
        }

        template<typename Handler>
        void post(const std::string& pattern, Handler&& handler) {
          add_route(method_t::post, pattern, std::forward<Handler>(handler));
        }

        template<typename Handler>
        void put(const std::string& pattern, Handler&& handler) {
          add_route(method_t::put, pattern, std::forward<Handler>(handler));
        }

        template<typename Handler>
        void delete_(const std::string& pattern, Handler&& handler) {
          add_route(method_t::delete_, pattern, std::forward<Handler>(handler));
        }

        fan::event::runv_t<response_t> handle(const request_t& req);
      };

      inline request_t parse_request(const std::string& raw_request) {
        request_t req;
        std::istringstream stream(raw_request);
        std::string line;

        if (!std::getline(stream, line)) {
          return req;
        }

        line.erase(line.find_last_not_of("\r\n") + 1);

        std::istringstream request_line(line);
        std::string method_str, full_path, version;
        request_line >> method_str >> full_path >> version;

        if (method_str == "GET") req.method = method_t::get;
        else if (method_str == "POST") req.method = method_t::post;
        else if (method_str == "PUT") req.method = method_t::put;
        else if (method_str == "DELETE") req.method = method_t::delete_;

        req.raw_path = full_path;
        std::size_t query_pos = full_path.find('?');

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
          std::size_t colon_pos = line.find(':');
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

      struct server_t {
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

        fan::event::task_t listen(const fan::network::listen_address_t& address);
      };

#ifdef FAN_NETWORK_CURL_ENABLED
      struct config_t {
        bool verify_ssl = true;
        bool follow_redirects = true;
        long timeout_seconds = 30;
        std::string user_agent = "libcurl-client/1.0";
      };

      struct async_request_t : std::enable_shared_from_this<async_request_t> {
        void* easy_handle = nullptr;
        void* curl_headers = nullptr;
        std::string url;
        config_t config;
        std::unordered_map<std::string, std::string> headers_map;
        response_t response;
        std::string error_message;
        std::coroutine_handle<> awaiting {};
        std::atomic<bool> completed {false};

        explicit async_request_t(const std::string& u, const config_t& cfg);
        ~async_request_t();

        bool await_ready() const noexcept { return false; }
        void await_suspend(std::coroutine_handle<> h);
        std::expected<response_t, std::string> await_resume();
        static std::size_t write_cb(char* ptr, std::size_t size, std::size_t nmemb, void* userdata);
      };

      fan::event::runv_t<std::expected<response_t, std::string>>
        get(const std::string& url, const config_t& cfg = {});

      fan::event::runv_t<std::expected<response_t, std::string>>
        post(const std::string& url,
          const std::string& body,
          const std::unordered_map<std::string, std::string>& headers = {},
          const config_t& cfg = {}
        );

      struct client_t {
        std::string base_url;
        config_t config;

        client_t(const std::string& url, const config_t& cfg = {})
          : base_url(url), config(cfg) {}

        fan::event::runv_t<std::expected<response_t, std::string>>
          get(const std::string& path);

        fan::event::runv_t<std::expected<response_t, std::string>>
          post(const std::string& path, const fan::json& body);

        fan::event::runv_t<std::expected<response_t, std::string>>
          post(const std::string& path, const std::string& body,
            const std::unordered_map<std::string, std::string>& headers = {});

        fan::event::runv_t<std::expected<response_t, std::string>>
          put(const std::string& path, const fan::json& body);

        fan::event::runv_t<std::expected<response_t, std::string>>
          delete_(const std::string& path);
      };
#endif
    } // namespace http
    // -------------------------------HTTP/REST-------------------------------
#endif
  } // namespace network
}
#endif