#pragma once

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #pragma comment(lib, "ws2_32.lib")
  typedef SOCKET socket_t;
#else
  #include <arpa/inet.h>
  #include <fcntl.h>
  #include <netinet/in.h>
  #include <sys/socket.h>
  #include <unistd.h>
  typedef int socket_t;
  #define INVALID_SOCKET -1
  #define SOCKET_ERROR -1
  #define closesocket close
#endif

#include <fan/ev/ev.h>

#include <unordered_set>
#include <memory>

namespace fan {
  namespace network {
    struct socket_awaitable_t {
      socket_t sock;
      const char* send_buffer = nullptr;
      char* recv_buffer = nullptr;
      size_t buffer_size = 0;
      struct operation_e { enum { accept, read, write }; };
      int op;
      sockaddr_in* client_addr = nullptr;
      socklen_t* addr_len = nullptr;

      socket_awaitable_t(socket_t s, int o) : sock(s), op(o) {}

      bool await_ready() { return true; }

      void await_suspend(std::coroutine_handle<> h) {
#ifdef _WIN32
        unsigned long mode = 1;
        ioctlsocket(sock, FIONBIO, &mode);
#else
        int flags = fcntl(sock, F_GETFL, 0);
        fcntl(sock, F_SETFL, flags | O_NONBLOCK);
#endif
      }

      uintptr_t await_resume() {
        switch (op) {
        case operation_e::accept: {
          auto client_sock = accept(sock, (sockaddr*)client_addr, addr_len);
          return client_sock;
        }
        case operation_e::read:
          return recv(sock, recv_buffer, buffer_size, 0);
        case operation_e::write:
          return send(sock, send_buffer, buffer_size, 0);
        }
        return -1;
      }
    };

    static task_t<void> read_data(socket_t sock, char* buffer, size_t buffer_size) {
      socket_awaitable_t size_read_op{ sock, socket_awaitable_t::operation_e::read };
      size_read_op.recv_buffer = buffer;
      size_read_op.buffer_size = buffer_size;
      size_t total_bytes_read = 0;
      while (total_bytes_read < buffer_size) {
        auto bytes_read = co_await size_read_op;
        if (bytes_read <= 0) {
          co_return{};
        }
        total_bytes_read += bytes_read;
        size_read_op.recv_buffer += bytes_read;
        size_read_op.buffer_size -= bytes_read;
      }
      co_return{};
    }
    static task_t<void> write_data(socket_t sock, const char* buffer, size_t buffer_size) {
      socket_awaitable_t size_write_op{ sock, socket_awaitable_t::operation_e::write };
      size_write_op.send_buffer = buffer;
      size_write_op.buffer_size = buffer_size;
      size_t total_bytes_sent = 0;
      while (total_bytes_sent < buffer_size) {
        auto bytes_read = co_await size_write_op;
        if ((int64_t)bytes_read <= 0) {
          co_return{};
        }
        total_bytes_sent += bytes_read;
        size_write_op.recv_buffer += bytes_read;
        size_write_op.buffer_size -= bytes_read;
      }
      co_return{};
    }

    class network_client_t {
      socket_t sock = INVALID_SOCKET;

    public:
      network_client_t() {
#ifdef _WIN32
        WSADATA wsaData;
        WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif
      }

      ~network_client_t() {
        if (sock != INVALID_SOCKET) {
          closesocket(sock);
        }
#ifdef _WIN32
        WSACleanup();
#endif
      }

      task_t<bool> connect(const char* host, int port) {
        sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock == INVALID_SOCKET) {
          co_return false;
        }

        sockaddr_in addr = {};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        inet_pton(AF_INET, host, &addr.sin_addr);

        if (::connect(sock, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
          co_return false;
        }

        co_return true;
      }

      std::vector<char> buffer;

      task_t<void> send_data(const std::string& data) {
        {
          int data_size = data.size();
          auto task = write_data(sock, reinterpret_cast<const char*>(&data_size), sizeof(data_size));
          task.handle.resume();
        }

        {
          auto task = write_data(sock, data.c_str(), data.size());
          task.handle.resume();
        }
        co_return {};
      }
      task_t<std::string> receive_data() {
        int response_size = 0;
        {
          auto task = read_data(sock, reinterpret_cast<char*>(&response_size), sizeof(response_size));
          task.handle.resume();
        }

        {
          buffer.resize(response_size);
          auto task = read_data(sock, buffer.data(), response_size);
          task.handle.resume();
        }

        co_return std::string(buffer.data(), response_size);
      }
    };

    struct client_t {
      socket_t sock;
      bool connected = true;
      std::vector<char> buffer;
      std::function<void()> loop;

      client_t(socket_t s) : sock(s), buffer(1024) {}

      ~client_t() {
        if (sock != INVALID_SOCKET) {
          closesocket(sock);
        }
      }

      bool is_connected() const { return connected; }

      task_t<void> handle_client() {
        while (connected) {
          if (loop) {
            loop();
         }
        }
        co_return{};
      }
    };

    struct server_t {
      socket_t listener = INVALID_SOCKET;
      bool running = true;
      std::vector<std::shared_ptr<client_t>> clients;
      std::function<void(client_t&)> on_connect;

      server_t() {
#ifdef _WIN32
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
          throw std::runtime_error("Failed to initialize Winsock");
        }
#endif
      }

      ~server_t() {
        if (listener != INVALID_SOCKET) {
          closesocket(listener);
        }
#ifdef _WIN32
        WSACleanup();
#endif
      }

      task_t<void> run(const char* host, int port) {
        // Create listener socket
        listener = socket(AF_INET, SOCK_STREAM, 0);
        if (listener == INVALID_SOCKET) {
          throw std::runtime_error("failed to create socket");
        }

        // Bind and listen
        sockaddr_in addr = {};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        inet_pton(AF_INET, host, &addr.sin_addr);

        if (bind(listener, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
          fan::throw_error("failed to bind socket");
        }

        if (listen(listener, SOMAXCONN) == SOCKET_ERROR) {
          fan::throw_error("failed to listen on socket");
        }

        fan::print("server listening on ", host, ":", port);

        while (running) {
          sockaddr_in client_addr = {};
          socklen_t addr_len = sizeof(client_addr);

          socket_awaitable_t accept_op{ listener, socket_awaitable_t::operation_e::accept };
          accept_op.client_addr = &client_addr;
          accept_op.addr_len = &addr_len;

          auto client_sock = co_await accept_op;

          if (client_sock == INVALID_SOCKET) {
            continue;
          }

          auto client = std::make_shared<client_t>(client_sock);
          clients.emplace_back(client);


          if (on_connect) {
            on_connect(*clients.back());
          }

          auto task = client->handle_client();
          task.handle.resume();
          // Clean up disconnected clients
          // clients.erase(
          //     std::remove_if(clients.begin(), clients.end(),
          //         [&](const auto& client) { return !client->is_connected(); }),
          //     clients.end()
          // );
        }
        co_return {};
      }

      void stop() { running = false; }
    };
  }
}