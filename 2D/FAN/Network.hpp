#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <iostream>
#include <functional>
#include <string>
#include <cstring>
#include <fstream>
#include <thread>
#include <vector>
#include <cstdint>
#include <map>

#include <FAN/Time.hpp>

#if defined(_WIN64) || defined(_WIN32)
#define FAN_WINDOWS
#endif

#ifdef FAN_WINDOWS
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib,"ws2_32.lib")
#else
using SOCKET = uintptr_t;
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#define SOCKET_ERROR -1
#define INVALID_SOCKET (uintptr_t)(~0)
#endif

struct File {
	File(const char* file_name) : name(file_name) {}
	void read() {
		std::ifstream file(name.c_str(), std::ifstream::ate | std::ifstream::binary);
		file.seekg(0, std::ios::end);
		data.resize(file.tellg());
		file.seekg(0, std::ios::beg);
		file.read(&data[0], data.size());
		file.close();
	}
	static inline bool file_exists(const std::string& name) {
		if (FILE* file = fopen(name.c_str(), "r")) {
			fclose(file);
			return true;
		}
		else {
			return false;
		}
	}
	std::string data;
	std::string name;
};

#ifdef FAN_WINDOWS
void init_winsock() {
	int err;
	WSADATA wsa;
	if ((err = WSAStartup(MAKEWORD(2, 2), &wsa)) != 0) {
		printf("WSAStartup failed with error: %d\n", err);
		exit(EXIT_FAILURE);
	}
}
#endif

class Username {
public:
	Username() : username() {}
	Username(std::string_view username) :
		username(username) {}
	inline std::string get_username() {
		return username;
	}
protected:
	inline void set_username(std::string_view new_username) {
		username = new_username;
	}
private:
	std::string username;
};

struct Message_Info {
	Message_Info(const Username& username, std::string_view message) :
		username(username), message(message) {}
	Username username;
	std::string message;
};

class tcp_server;
class client;

class data_handler {
protected:
	void m_get_file(SOCKET socket) const {
		std::string data;
		uint64_t file_size = *(size_t*)m_get_data(socket, sizeof(uint64_t)).data();
		uint64_t file_name_size = *(size_t*)m_get_data(socket, sizeof(uint64_t)).data();
		std::string file_name((const char*)m_get_data(socket, file_name_size).data());

		data.resize(file_size);

		uint64_t recvlen = 0;
		uint64_t totalsend = 0;
		uint64_t remaining = file_size;
		while (totalsend != file_size) {
			recvlen = recv(socket, &data[totalsend], remaining, 0);
			if (recvlen == INVALID_SOCKET) {
				puts("recv failed");
				return;
			}
			totalsend += recvlen;
			remaining = file_size - totalsend;
			printf("remaining: %llu\n", remaining);
		}

		int rename_count = 1;
		while (File::file_exists(file_name.c_str())) {
			std::string rename_str = std::to_string(rename_count);
			if (rename_count > 1) {
				for (int i = 0; i < rename_str.length(); i++) {
					file_name[file_name.find('(') + 1 + i] = rename_str[i];
				}
			}
			else {
				std::string file_type;
				for (int i = file_name.find('.'); i < file_name.length(); i++) {
					file_type.push_back(file_name[i]);
				}
				file_name.erase(file_name.find('.'), file_name.length());
				file_name.push_back(' ');
				file_name.push_back('(');
				file_name.append(rename_str);
				file_name.push_back(')');
				file_name.append(file_type);
			}
			rename_count++;
		}

		FILE* file = fopen(file_name.c_str(), "w+b");
		fwrite(data.data(), data.size(), 1, file);
		fclose(file);
		puts("received file");
		data.clear();
	}
	std::string m_get_message(SOCKET socket) const {
		uint64_t message_size = *(uint64_t*)m_get_data(
			socket,
			sizeof(uint64_t)
		).data();
		return m_get_data(socket, message_size);
	}
	Message_Info m_get_message_sender(SOCKET socket) const {
		std::string sender_username = m_get_message(socket);
		std::string message = m_get_message(socket);
		return Message_Info(Username(sender_username), message);
	}
	std::string m_get_data(SOCKET socket, uint64_t size) const {
		std::string data;
		data.resize(size);

		uint64_t recvlen = 0;
		uint64_t totalsend = 0;
		uint64_t remaining = size;

		while (recvlen != size) {
			recvlen = recv(socket, &data[totalsend], remaining, 0);
			if (recvlen == INVALID_SOCKET) {
				puts("recv failed");
				return std::string();
			}
			totalsend = totalsend + recvlen;
			remaining = recvlen - totalsend;
		}
		return data;
	}
	void m_send_file(SOCKET socket, File file) const {
		file.read();
		int64_t sendlen(0);
		uint64_t totalsend(0);
		uint64_t remaining = file.data.size();
		m_send_data(socket, (const char*)&remaining, sizeof(remaining));
		uint64_t file_name_size = file.name.size();
		m_send_message(socket, file.name.c_str());

		while (sendlen != file.data.size()) {
			sendlen = send(socket, &file.data[totalsend], remaining, 0);
			if (sendlen == SOCKET_ERROR) {
				printf("error\n");
			}
			totalsend += sendlen;
			remaining = file.data.size() - totalsend;
		}
	}
	void m_send_message(SOCKET socket, std::string message) const {
		uint64_t message_size = message.size();
		m_send_data(socket, (const char*)&message_size, sizeof(uint64_t));
		m_send_data(socket, message.c_str(), message_size);
	}
	void m_send_data(SOCKET socket, const char* data, uint64_t size) const {
		uint64_t sendlen = 0;
		uint64_t totalsend = 0;
		uint64_t remaining = size;

		while (sendlen != size) {
			sendlen = send(socket, &data[totalsend], remaining, 0);
			if (sendlen == SOCKET_ERROR) {
				puts("error");
			}
			totalsend = totalsend + sendlen;
			remaining = sendlen - totalsend;
		}
	}
};

class tcp_server : public Username, public data_handler {
public:
	tcp_server(unsigned short port) {
		std::thread listen_thread(&tcp_server::initialize, this, port);
		listen_thread.detach();
	}
	~tcp_server() {
		exit_program = true;
		for (int i = 0; i < socket_size(); i++) {
			send_data(
				sockets[i],
				"exit",
				strlen("exit")
			);
		}
#ifdef FAN_WINDOWS
		for (auto&& i : sockets) {
			closesocket(i);
		}
		WSACleanup();
#else
		for (auto&& i : sockets) {
			close(i);
		}
#endif
	}

	inline SOCKET get_socket(uint64_t socket) const { return sockets[socket]; }

	inline std::string get_message(SOCKET socket) const { return m_get_message(socket); }
	inline Message_Info get_message_sender(SOCKET socket) const { return m_get_message_sender(socket); }
	inline void send_message(SOCKET socket, std::string message) const { m_send_message(socket, message); }

	inline void get_file(SOCKET socket) const { m_get_file(socket); }
	inline void send_file(SOCKET socket, File file) const { m_send_file(socket, file); }

	inline std::string get_data(SOCKET socket, uint64_t size) const { return m_get_data(socket, size); }
	inline void send_data(SOCKET socket, const char* data, uint64_t size) const { m_send_data(socket, data, size); }

	unsigned short convert_endian(unsigned short port) {
		return (unsigned short)port << 8 | (unsigned short)port >> 8;
	}

	void redirect_message(SOCKET socket) {
		std::string sender_username = get_message(
			socket
		);
		std::string receiver_username = get_message(
			socket
		);
		if (users.find(receiver_username.c_str()) == users.end()) {
			printf("%s offline\n", receiver_username.c_str());
			return;
		}
		int receiver = users[receiver_username.c_str()];
		send_message(sockets[receiver], sender_username);
		send_message(sockets[receiver], m_get_message(socket));
		printf(
			"message redirected from %s to %s",
			sender_username.c_str(),
			receiver_username.c_str()
		);
	}

	constexpr bool quit() const {
		return exit_program;
	}

	inline uint64_t socket_size() const {
		return sockets.size();
	}

	std::vector<sockaddr_in> info;
	std::vector<std::thread> threads;
	std::map<std::string, int> users;
private:

	void initialize(unsigned short port) {
		int error = 0;
		uintptr_t server_socket;
		addrinfo hints, * result;
		memset(&hints, 0, sizeof(hints));
		hints.ai_family = AF_INET;
		hints.ai_socktype = SOCK_STREAM;
		hints.ai_protocol = IPPROTO_TCP;
		hints.ai_flags = AI_PASSIVE;

		server_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

		if (server_socket == INVALID_SOCKET) {
#ifdef FAN_WINDOWS
			printf("socket failed with error: %ld\n", WSAGetLastError());
			WSACleanup();
#endif
			exit(EXIT_FAILURE);
		}

		if ((error = getaddrinfo(NULL, std::to_string(port).c_str(), &hints, &result)) != 0) {
			printf("getaddrinfo failed with error: %d\n", error);
#ifdef FAN_WINDOWS
			WSACleanup();
#endif
			exit(EXIT_FAILURE);
		}

		if (bind(server_socket, result->ai_addr, (int)result->ai_addrlen) == SOCKET_ERROR) {
			printf("bind failed\n");
		}

		freeaddrinfo(result);

		if (listen(server_socket, SOMAXCONN) == SOCKET_ERROR) {
			printf("listen failed\n");

#ifdef FAN_WINDOWS
			closesocket(server_socket);
			WSACleanup();
#else
			close(server_socket);
#endif
			exit(EXIT_FAILURE);
		}

		for (int i = 0; ; i++) {
			SOCKET sock = accept(server_socket, NULL, NULL);
			sockets.push_back(sock);
			std::string username = m_get_message(sock).c_str();
			printf("%s joined\n", username.c_str());
			users[username] = i;

			if (exit_program) {
#ifdef FAN_WINDOWS
				closesocket(server_socket);
				WSACleanup();
#else
				printf("accept failed\n");
				close(server_socket);
#endif
				return;
			}
			if (sock == INVALID_SOCKET) {
#ifdef FAN_WINDOWS
				printf("accept failed %d\n", WSAGetLastError());
				closesocket(server_socket);
				WSACleanup();
#else
				printf("accept failed\n");
				close(server_socket);
#endif
				exit(EXIT_FAILURE);
			}
		}

#ifdef FAN_WINDOWS
		closesocket(server_socket);
#else
		close(server_socket);
#endif
	}
	std::vector<SOCKET> sockets;

	bool exit_program = false;
};

class client : public Username, public data_handler {
public:
	client(const char* ip, unsigned short port, std::string username = std::string()) {
		this->set_username(username);

		int error;
		sockaddr_in clientService;

		clientService.sin_family = AF_INET;
		clientService.sin_addr.s_addr = inet_addr(ip);
		clientService.sin_port = htons(port);

		connect_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

		if (connect_socket == INVALID_SOCKET) {
#ifdef FAN_WINDOWS
			wprintf(L"socket function failed with error: %ld\n", WSAGetLastError());
			WSACleanup();
#endif
			exit(EXIT_FAILURE);
		}

		error = connect(connect_socket, (sockaddr*)&clientService, sizeof(clientService));

		if (error == SOCKET_ERROR) {
#ifdef FAN_WINDOWS
			wprintf(L"connect function failed with error: %ld\n", WSAGetLastError());
			error = closesocket(connect_socket);
			if (error == SOCKET_ERROR)
				wprintf(L"closesocket function failed with error: %ld\n", WSAGetLastError());
			WSACleanup();
#else
			fprintf(stderr, "getaddr failed: %s\n", strerror(errno));
#endif
			exit(EXIT_FAILURE);
		}
		if (!username.empty()) {
			m_send_message(connect_socket, get_username());
		}
	}

	~client() {
		if (m_get_data(connect_socket, strlen("exit")) != "exit") {
			puts("failed to receive exit message");
			exit(EXIT_FAILURE);
		}
#ifdef FAN_WINDOWS
		closesocket(connect_socket);
		WSACleanup();
#else
		close(connect_socket);
#endif
	}

	inline std::string get_message() const { return m_get_message(connect_socket); }
	inline Message_Info get_message_sender() const { return m_get_message_sender(connect_socket); }
	inline void send_message(std::string message) const { m_send_message(connect_socket, message); }

	inline void get_file() const { m_get_file(connect_socket); }
	inline void send_file(File file) const { m_send_file(connect_socket, file); }

	inline std::string get_data(uint64_t size) const { return m_get_data(connect_socket, size); }
	inline void send_data(const char* data, uint64_t size) const { m_send_data(connect_socket, data, size); }

	void send_message_user(const std::string& message, const std::string& destination) {
		uint64_t my_username_size = get_username().size();
		send_data((const char*)&my_username_size, sizeof(uint64_t));
		send_data(get_username().c_str(), my_username_size);

		uint64_t destination_username_size = destination.size();
		send_data((const char*)&destination_username_size, sizeof(uint64_t));
		send_data(destination.c_str(), destination_username_size);

		uint64_t message_size = message.size();
		send_data((const char*)&message_size, sizeof(uint64_t));
		send_data(message.c_str(), message_size);
	}

private:
	SOCKET connect_socket;
	uint64_t send_size = 0;
	uint64_t sent_size = 0;
};