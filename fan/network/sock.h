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
        if (ioctlsocket(sock, FIONBIO, &mode) != NO_ERROR) {
          throw std::runtime_error("failed to set non-blocking mode on socket");
        }
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
              fan::print("Socket creation failed: ", strerror(errno));
              co_return false;
          }

          sockaddr_in addr = {};
          addr.sin_family = AF_INET;
          addr.sin_port = htons(port);
          if (inet_pton(AF_INET, host, &addr.sin_addr) <= 0) {
              fan::print("Invalid address/ Address not supported: ", strerror(errno));
              co_return false;
          }

          if (::connect(sock, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
              fan::print("Connection failed: ", strerror(errno));
              co_return false;
          }

          co_return true;
      }


      std::vector<char> buffer;
      task_t<std::string> echo(const std::string& msg) {
        buffer.resize(1024);

        int msg_size = msg.size();
        {
          socket_awaitable_t size_write_op{ sock, socket_awaitable_t::operation_e::write };
          size_write_op.send_buffer = reinterpret_cast<const char*>(&msg_size);
          size_write_op.buffer_size = sizeof(msg_size);
          size_t total_bytes_sent = 0;
          while (total_bytes_sent < sizeof(msg_size)) {
            auto bytes_sent = co_await size_write_op;
            if (bytes_sent == SOCKET_ERROR) {
              // Handle error
              co_return "";
            }
            total_bytes_sent += bytes_sent;
            size_write_op.send_buffer += bytes_sent;
            size_write_op.buffer_size -= bytes_sent;
          }
        }

        {
          socket_awaitable_t write_op{ sock, socket_awaitable_t::operation_e::write };
          write_op.send_buffer = msg.c_str();
          write_op.buffer_size = msg.size();
          size_t total_bytes_sent = 0;
          while (total_bytes_sent < msg.size()) {
            auto bytes_sent = co_await write_op;
            if (bytes_sent == SOCKET_ERROR) {
              // Handle error
              co_return "";
            }
            total_bytes_sent += bytes_sent;
            write_op.send_buffer += bytes_sent;
            write_op.buffer_size -= bytes_sent;
          }
        }

        int response_size = 0;
        {
          socket_awaitable_t size_read_op{ sock, socket_awaitable_t::operation_e::read };
          size_read_op.recv_buffer = reinterpret_cast<char*>(&response_size);
          size_read_op.buffer_size = sizeof(response_size);
          size_t total_bytes_read = 0;
          while (total_bytes_read < sizeof(response_size)) {
            auto bytes_read = co_await size_read_op;
            if (bytes_read == SOCKET_ERROR) {
              // Handle error
              co_return "";
            }
            total_bytes_read += bytes_read;
            size_read_op.recv_buffer += bytes_read;
            size_read_op.buffer_size -= bytes_read;
          }
        }

        {
          socket_awaitable_t read_op{ sock, socket_awaitable_t::operation_e::read };
          buffer.resize(response_size);
          read_op.recv_buffer = buffer.data();
          read_op.buffer_size = buffer.size();
          size_t total_bytes_read = 0;
          while (total_bytes_read < response_size) {
            auto bytes_read = co_await read_op;
            if (bytes_read == SOCKET_ERROR) {
              // Handle error
              co_return "";
            }
            total_bytes_read += bytes_read;
            read_op.recv_buffer += bytes_read;
            read_op.buffer_size -= bytes_read;
          }
        }

        co_return std::string(buffer.data(), response_size);
      }
    };

    class client_t {
      socket_t sock;
      bool connected = true;
      std::vector<char> buffer;

    public:
      client_t(socket_t s) : sock(s), buffer(1024) {}

      ~client_t() {
        if (sock != INVALID_SOCKET) {
          closesocket(sock);
        }
      }

      bool is_connected() const { return connected; }

      task_t<void> handle_client() {
        while (connected) {
          // Receive the size of the incoming message
          int msg_size = 0;
          {
            socket_awaitable_t size_read_op{ sock, socket_awaitable_t::operation_e::read };
            size_read_op.recv_buffer = reinterpret_cast<char*>(&msg_size);
            size_read_op.buffer_size = sizeof(msg_size);
            size_t total_bytes_read = 0;
            while (total_bytes_read < sizeof(msg_size)) {
              auto bytes_read = co_await size_read_op;
              if (bytes_read <= 0) {
                connected = false;
                co_return{};
              }
              total_bytes_read += bytes_read;
              size_read_op.recv_buffer += bytes_read;
              size_read_op.buffer_size -= bytes_read;
            }
          }

          // Receive the actual message
          buffer.resize(msg_size);
          {
            socket_awaitable_t read_op{ sock, socket_awaitable_t::operation_e::read };
            read_op.recv_buffer = buffer.data();
            read_op.buffer_size = buffer.size();
            size_t total_bytes_read = 0;
            while (total_bytes_read < msg_size) {
              auto bytes_read = co_await read_op;
              if (bytes_read <= 0) {
                connected = false;
                co_return{};
              }
              total_bytes_read += bytes_read;
              read_op.recv_buffer += bytes_read;
              read_op.buffer_size -= bytes_read;
            }
          }

          std::string received(buffer.data(), msg_size);
          std::string response;

          if (received == "ping") {
            response = "pong";
            fan::print("received ping, sending", response);
          }
          else {
            response = "unknown command";
            fan::print("Received unknown command:", received);
          }

          // Send the size of the response first
          int response_size = response.size();
          {
            socket_awaitable_t size_write_op{ sock, socket_awaitable_t::operation_e::write };
            size_write_op.send_buffer = reinterpret_cast<const char*>(&response_size);
            size_write_op.buffer_size = sizeof(response_size);
            size_t total_bytes_sent = 0;
            while (total_bytes_sent < sizeof(response_size)) {
              auto bytes_sent = co_await size_write_op;
              if (bytes_sent <= 0) {
                connected = false;
                co_return{};
              }
              total_bytes_sent += bytes_sent;
              size_write_op.send_buffer += bytes_sent;
              size_write_op.buffer_size -= bytes_sent;
            }
          }

          // Send the actual response
          {
            socket_awaitable_t write_op{ sock, socket_awaitable_t::operation_e::write };
            write_op.send_buffer = response.c_str();
            write_op.buffer_size = response.size();
            size_t total_bytes_sent = 0;
            while (total_bytes_sent < response.size()) {
              auto bytes_sent = co_await write_op;
              if (bytes_sent <= 0) {
                connected = false;
                co_return{};
              }
              total_bytes_sent += bytes_sent;
              write_op.send_buffer += bytes_sent;
              write_op.buffer_size -= bytes_sent;
            }
          }
        }
      }
    };

    class server_t {
      socket_t listener = INVALID_SOCKET;
      bool running = true;
      std::unordered_set<std::shared_ptr<client_t>> clients;

    public:
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
        listener = socket(AF_INET, SOCK_STREAM, 0);
        if (listener == INVALID_SOCKET) {
          throw std::runtime_error("Failed to create socket");
        }

        sockaddr_in addr = {};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        inet_pton(AF_INET, host, &addr.sin_addr);

        if (bind(listener, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
          throw std::runtime_error("Failed to bind socket");
        }

        if (listen(listener, SOMAXCONN) == SOCKET_ERROR) {
          throw std::runtime_error("Failed to listen on socket");
        }

        fan::print("Server listening on ", host, ":", port);

        // Accept loop
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
          clients.insert(client);

          auto task = client->handle_client();
          task.handle.resume();
          // Clean up disconnected clients
          // clients.erase(
          //     std::remove_if(clients.begin(), clients.end(),
          //         [&](const auto& client) { return !client->is_connected(); }),
          //     clients.end()
          // );
        }
      }

      void stop() { running = false; }
    };
  }
}