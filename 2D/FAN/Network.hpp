#ifdef _WIN64
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


#include <iostream>
#include <functional>
#include <string>
#include <cstring>
#include <fstream>
#include <thread>
#include <FAN/Time.hpp>

constexpr auto PORT = 43254;

class File {
public:
	File(const char* file_name) : name(file_name) {}
	void read() {
		std::ifstream file(name.c_str(), std::ifstream::ate | std::ifstream::binary);
		file.seekg(0, std::ios::end);
		data.resize(file.tellg());
		file.seekg(0, std::ios::beg);
		file.read(&data[0], data.size());
		file.close();
	}
	std::string data;
	std::string name;
};

inline bool file_exists(const std::string& name) {
	if (FILE* file = fopen(name.c_str(), "r")) {
		fclose(file);
		return true;
	}
	else {
		return false;
	}
}

#ifdef _WIN64
void init_winsock() {
	int err;
	WSADATA wsa;
	if ((err = WSAStartup(MAKEWORD(2, 2), &wsa)) != 0) {
		printf("WSAStartup failed with error: %d\n", err);
		exit(EXIT_FAILURE);
	}
}
#endif

class tcp_server {
public:
	tcp_server(unsigned short port) {
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
#ifdef _WIN64
			printf("socket failed with error: %ld\n", WSAGetLastError());
			WSACleanup();
#endif
			exit(EXIT_FAILURE);
		}

		if ((error = getaddrinfo(NULL, std::to_string(port).c_str(), &hints, &result)) != 0) {
			printf("getaddrinfo failed with error: %d\n", error);
#ifdef _WIN64
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

#ifdef _WIN64
			closesocket(server_socket);
			WSACleanup();
#else
			close(server_socket);
#endif
			exit(EXIT_FAILURE);
		}
		client_socket = accept(server_socket, NULL, NULL);
		if (client_socket == INVALID_SOCKET) {
#ifdef _WIN64
			printf("accept failed\n", WSAGetLastError());
			closesocket(server_socket);
			WSACleanup();
#else
			printf("accept failed\n");
			close(server_socket);
#endif
			exit(EXIT_FAILURE);
		}
		puts("connected");

#ifdef _WIN64
		closesocket(server_socket);
#else
		close(server_socket);
#endif
	}

	~tcp_server() {
		send_data("exit", strlen("exit"));
#ifdef _WIN64
		closesocket(client_socket);
		WSACleanup();
#else
		close(client_socket);
#endif
	}

	unsigned short convert_endian(unsigned short port) {
		return (unsigned short)port << 8 | (unsigned short)port >> 8;
	}

	void send_data(const char* packet, std::size_t size, std::size_t index = 0) {

		std::size_t sendlen = 0;
		std::size_t totalsend = 0;
		std::size_t remaining = size;

		while (sendlen != size) {
			sendlen = send(client_socket, &packet[totalsend], remaining, 0);
			if (sendlen == SOCKET_ERROR) {
				printf("error\n");
			}
			totalsend = totalsend + sendlen;
			remaining = sendlen - totalsend;
		}
	}

	std::string receive_data(std::size_t size) {
		std::string data;

		data.resize(size);

		std::size_t recvlen = 0;
		std::size_t totalsend = 0;
		std::size_t remaining = size;

		while (recvlen != size) {
			recvlen = recv(client_socket, &data[totalsend], remaining, 0);
			if (recvlen == INVALID_SOCKET) {
				printf("recv failed\n");
				return std::string();
			}
			totalsend = totalsend + recvlen;
			remaining = recvlen - totalsend;
		}
		return data;
	}

	void receive_file() {
		std::string data;
		std::size_t file_size = *(size_t*)receive_data(sizeof(std::size_t)).data();
		std::size_t file_name_size = *(size_t*)receive_data(sizeof(std::size_t)).data();
		std::string file_name((const char*)receive_data(file_name_size).data());

		data.resize(file_size);

		std::size_t recvlen = 0;
		std::size_t totalsend = 0;
		std::size_t remaining = file_size;

		while (totalsend != file_size) {
			recvlen = recv(client_socket, &data[totalsend], remaining, 0);
			if (recvlen == INVALID_SOCKET) {
				printf("recv failed\n");
				return;
			}
			totalsend += recvlen;
			remaining = file_size - totalsend;
			printf("remaining: %llu\n", remaining);
		}

		int rename_count = 1;
		while (file_exists(file_name.c_str())) {
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
		printf("finished\n");
		data.clear();
	}

private:
	SOCKET client_socket;
};

class client {
public:
	client(const char* ip, unsigned short port) {
		int error;
		sockaddr_in clientService;

		clientService.sin_family = AF_INET;
		clientService.sin_addr.s_addr = inet_addr(ip);
		clientService.sin_port = htons(port);

		connect_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

		if (connect_socket == INVALID_SOCKET) {
#ifdef _WIN64
			wprintf(L"socket function failed with error: %ld\n", WSAGetLastError());
			WSACleanup();
#endif
			exit(EXIT_FAILURE);
		}

		error = connect(connect_socket, (sockaddr*)&clientService, sizeof(clientService));

		if (error == SOCKET_ERROR) {
#ifdef _WIN64
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
	}

	void send_data(const char* data, std::size_t size) const {
		int64_t sendlen = 0;
		std::size_t totalsend = 0;
		std::size_t remaining = size;

		while (sendlen != size) {
			sendlen = send(connect_socket, &data[totalsend], remaining, 0);
			if (sendlen == SOCKET_ERROR) {
				printf("error\n");
			}
			totalsend += sendlen;
			remaining = size - totalsend;
		}
	}

	void send_file(File file) const {
		file.read();
		int64_t sendlen(0);
		std::size_t totalsend(0);
		std::size_t remaining = file.data.size();
		send_data((const char*)&remaining, sizeof(remaining));
		std::size_t file_name_size = file.name.size();
		send_data((const char*)&file_name_size, sizeof(file_name_size));
		send_data((const char*)&file.name[0], file_name_size);

		while (sendlen != file.data.size()) {
			sendlen = send(connect_socket, &file.data[totalsend], remaining, 0);
			if (sendlen == SOCKET_ERROR) {
				printf("error\n");
			}
			totalsend += sendlen;
			remaining = file.data.size() - totalsend;
		}
	}

	std::string receive_packet(std::size_t size) {
		std::string data;

		data.resize(size);

		std::size_t recvlen(0);
		std::size_t totalsend(0);
		std::size_t remaining(size);

		while (recvlen != size) {
			recvlen = recv(connect_socket, &data[totalsend], remaining, 0);
			if (recvlen == INVALID_SOCKET) {
				printf("recv failed\n");
				return std::string();
			}
			totalsend = totalsend + recvlen;
			remaining = recvlen - totalsend;
		}
		return data;
	}

	~client() {
		if (receive_packet(strlen("exit")) != "exit") {
			printf("estimated close time: inf\n");
			std::cin.get();
		}
#ifdef _WIN64
		closesocket(connect_socket);
		WSACleanup();
#else
		close(connect_socket);
#endif
	}

private:
	uintptr_t connect_socket;
	std::size_t send_size = 0;
	std::size_t sent_size = 0;
};