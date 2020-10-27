#pragma once
#ifdef _MSC_VER
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#endif

#include <FAN/file.hpp>
#include <FAN/types.h>

#include <functional>
#include <thread>
#include <cstring>
#include <cmath>
#include <map>

#ifdef FAN_WINDOWS
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib,"ws2_32.lib")
#define WIN32_LEAN_AND_MEAN

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

static WSADATA wsa;
static int wsa_error = WSAStartup(MAKEWORD(2, 2), &wsa);
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

namespace fan {
	namespace network {
		const int max_packet_size = 1024;

		#define stringify(name) #name


		enum class packet_type {
			get_file,
			send_file,
			get_message,
			send_message,
			send_message_user,

			get_registeration,
			send_registeration,
			get_username,
			set_username,
			get_password,
			set_password,
			exit
		};

		static const char* packet_type_str[]{
			stringify(get_file),
			stringify(send_file),
			stringify(get_message),
			stringify(send_message),
			stringify(send_message_user),

			stringify(get_registeration),
			stringify(send_registeration),
			stringify(get_username),
			stringify(set_username),
			stringify(get_password),
			stringify(set_password),
			stringify(exit)
		};

		class User {
		public:
			User() : username(), password() {}
			User(const std::string& username, const std::string& password) :
				username(username), password(password) {}
			User(const std::string& username) :
				username(username), password() {}
			inline std::string get_username() {
				return username;
			}
		protected:
			std::map<std::string, int> users;
			inline void set_username(const std::string& new_username) {
				username = new_username;
			}
			inline std::string get_password() const {
				return password;
			}
			inline void set_password(const std::string& new_password) {
				username = new_password;
			}
		private:
			std::string username;
			std::string password;
		};

		struct Message_Info : public User {
			Message_Info(const User& username, const std::string& message) :
				User(username), message(message) {}
			std::string message;
		};

		class tcp_server;
		class client;

		class data_handler : public User {
		protected:
			std::string m_get_file(SOCKET socket, bool overwrite, std::string path = std::string()) const {
				std::string data;
				uint64_t file_size = *(size_t*)m_get_data(socket, sizeof(uint64_t)).data();
				uint64_t file_name_size = *(size_t*)m_get_data(socket, sizeof(uint64_t)).data();
				std::string file_name((const char*)m_get_data(socket, file_name_size).data());

				data.resize(file_size);

				uint64_t recvlen = 0;
				uint64_t totalsend = 0;
				uint64_t remaining = file_size;
				puts("started receiving");
				while (totalsend != file_size) {
					recvlen = recv(socket, &data[totalsend], remaining, 0);
					if (recvlen == INVALID_SOCKET) {
						puts("recv failed");
						return (const char*)0;
					}
					totalsend += recvlen;
					remaining = file_size - totalsend;
				}

				if (overwrite) {
					goto skip;
				}
				{

					if (!path.empty()) {
						path.push_back('\\');
					}
					file_name.insert(0, path);
					int rename_count = 1;
					while (File::file_exists(file_name.c_str())) {
						std::string rename_str = std::to_string(rename_count);
						if (rename_count > 1) {
							for (uint_t i = 0; i < rename_str.length(); i++) {
								char& l_path = file_name[file_name.find_last_of('(') + 1 + i];
								auto found = file_name.find_last_of(l_path);
								if (file_name[found] == ')') {
									file_name.insert(file_name.begin() + found, ' ');
								}
								l_path = rename_str[i];
							}
						}
						else {
							std::string file_type;
							for (uint_t i = file_name.find_last_of('.'); i < file_name.length(); i++) {
								if (i != std::string::npos) {
									file_type.push_back(file_name[i]);
								}
							}
							auto found = file_name.find_last_of('.');
							if (found != std::string::npos) {
								file_name.erase(file_name.find_last_of('.'), file_name.length());
							}
							file_name.push_back(' ');
							file_name.push_back('(');
							file_name.append(rename_str);
							file_name.push_back(')');
							file_name.append(file_type);
						}
						rename_count++;
					}
				}
			skip:
				if (path != std::string()) {
					File::write(path.c_str(), data, std::ios_base::binary);
				}
				else if (overwrite) {
					File::write(file_name.c_str(), data, std::ios_base::binary);
				}
				else {
					File::write(file_name.c_str(), data);
				}

				std::cout << "received file " << file_name << '\n';
				return file_name;
			}
			std::string m_get_message(SOCKET socket) const {
				uint64_t message_size = *(uint64_t*)m_get_data(
					socket,
					sizeof(uint64_t)
				).data();
				return m_get_data(socket, message_size);
			}
			Message_Info m_get_message_user(SOCKET socket) const {
				std::string sender_username = m_get_message(socket);
				std::string message = m_get_message(socket);
				return Message_Info(User(sender_username), message);
			}
			std::string m_get_data(SOCKET socket, uint64_t size) const {
				uint64_t amount_of_packets = std::ceil(size / max_packet_size) + 1;
				uint64_t received = 0;
				std::string data;
				for (uint_t i = 0; i < amount_of_packets; i++) {
					uint16_t packet_size = (!(size % max_packet_size) && size ? max_packet_size : size % max_packet_size);
					data.append(m_get_single_packet(socket, packet_size));
					received += packet_size;
					size -= packet_size;
				}
				return data;
			}
			std::string m_get_single_packet(SOCKET socket, uint16_t size) const {
				std::string data;
				data.resize(size);

				int receive_length = 0;
				uint16_t received = 0;
				uint16_t remaining = size;
				while (received < size) {
					receive_length = recv(socket, &data[received], remaining, 0);
					if (receive_length == SOCKET_ERROR) {
						puts("recv failed");
						return std::string();
					}
					received += receive_length;
					remaining -= receive_length;
				}
				return data;
			}
			void m_get_registeration(SOCKET socket) const {
				File f("codes");
				f.read();
				packet_type send_type = packet_type::send_message;
				m_send_message(socket, std::string((const char*)&send_type, sizeof(send_type)));
				if (f.data.find(m_get_message(socket)) != std::string::npos) {
					File::write("usernames", m_get_message(socket).operator+=(":").operator+=(m_get_message(socket)).operator+=("\n"));
					m_send_message(socket, "account has been registered");
				}
				else {
					m_send_message(socket, "invalid code");
				}
			}
			void m_send_file(SOCKET socket, File file, const std::string& username = std::string()) const {
				file.read();

				auto found = file.name.find("temp\\");
				if (found != std::string::npos) {
					file.name.erase(found, std::string("temp\\").size());
					auto rename_begin = file.name.find_last_of(" (");
					auto rename_end = file.name.find_last_of(")");
					if (rename_begin != std::string::npos && rename_end != std::string::npos) {
						file.name.erase(rename_begin - 1, (rename_end - (rename_begin - 1)) + 1);
					}
				}

				if (!username.empty()) {
					packet_type send_type = packet_type::send_file;
					m_send_message(socket, std::string((const char*)&send_type, sizeof(send_type)));
					m_send_message(socket, username);
				}

				uint64_t sendlen(0);
				uint64_t totalsend(0);
				uint64_t remaining = file.data.size();
				m_send_data(socket, (const char*)&remaining, sizeof(remaining));
				m_send_message(socket, file.name.c_str());

				while (sendlen != file.data.size()) {
					sendlen = send(socket, &file.data[totalsend], remaining, 0);
					if (sendlen == (uint64_t)SOCKET_ERROR) {
						printf("error\n");
					}
					totalsend += sendlen;
					remaining = file.data.size() - totalsend;
				}
				std::cout << "Sent file: " << file.name << '\n';
			}
			template <typename type>
			void m_send_vector(SOCKET socket, const std::vector<type>& vector) const {
				std::string str(vector.begin(), vector.end());
				m_send_message(socket, str);
			}
			template <typename type>
			std::vector<type> m_get_vector(SOCKET socket) const {
				std::string input = m_get_message(socket);
				std::vector<type> data(input.begin(), input.end());
				return data;
			}
			void m_send_data(SOCKET socket, const char* data, uint64_t size) const {
				uint64_t amount_of_packets = std::ceil(size / max_packet_size) + 1;
				uint64_t sent = 0;
				for (uint_t i = 0; i < amount_of_packets; i++) {
					uint16_t packet_size = (!(size % max_packet_size) && size ? max_packet_size : size % max_packet_size);
					std::string packet(data + sent, data + sent + packet_size);
					m_send_single_packet(socket, packet.c_str(), packet_size);
					sent += packet_size;
					size -= packet_size;
				}
			}
			inline void m_send_single_packet(SOCKET socket, const char* data, uint16_t size) const {
				uint16_t sent = 0;
				int send_length = 0;
				uint16_t remaining = size;
				while (sent < size) {
					send_length = send(socket, &data[sent], remaining, 0);
					if (send_length == SOCKET_ERROR) {
		#ifdef FAN_WINDOWS
						printf("send failed: %d\n", WSAGetLastError());
		#else
						fprintf(stderr, "send failed: %s\n", strerror(errno));
		#endif
						send_length = 0;
					}
					sent += send_length;
					remaining -= send_length;
				}
			}
			void m_send_message(SOCKET socket, std::string message) const {
				uint64_t message_size = message.size();
				m_send_data(socket, (const char*)&message_size, sizeof(uint64_t));
				m_send_data(socket, message.c_str(), message_size);
			}
			void m_send_message(SOCKET socket, const std::string& message, const std::string& destination) {
				packet_type type = packet_type::send_message_user;
				m_send_message(socket, std::string((const char*)&type, sizeof(type)));

				uint64_t my_username_size = get_username().size();
				m_send_data(socket, (const char*)&my_username_size, sizeof(uint64_t));
				m_send_data(socket, get_username().c_str(), my_username_size);

				uint64_t destination_username_size = destination.size();
				m_send_data(socket, (const char*)&destination_username_size, sizeof(uint64_t));
				m_send_data(socket, destination.c_str(), destination_username_size);

				uint64_t message_size = message.size();
				m_send_data(socket, (const char*)&message_size, sizeof(uint64_t));
				m_send_data(socket, message.c_str(), message_size);
			}
			void m_send_registeration(SOCKET socket, const User& user_info, const std::string& code) const {
				packet_type send_type = packet_type::send_registeration;
				m_send_message(socket, (const char*)&send_type);
				m_send_message(socket, code.c_str());
				m_send_message(socket, (const char*)&user_info);
			}
		};

		class tcp_server : public data_handler {
		public:
			tcp_server(unsigned short port, bool multithread = false) {
				if (multithread) {
					std::thread listen_thread(
						&tcp_server::initialize,
						this,
						port,
						multithread
					);
					listen_thread.detach();
				}
				else {
					initialize(port);
				}
			}
			~tcp_server() {
				exit_program = true;
		#ifdef FAN_WINDOWS
				for (auto& i : sockets) {
					closesocket(i);
				}
				WSACleanup();
		#else
				for (auto&& i : sockets) {
					close(i);
				}
		#endif
			}

			inline SOCKET get_socket(uint64_t socket = 0) const { return sockets[socket]; }

			inline std::string get_message(SOCKET socket = 0) const { return m_get_message(!socket ? sockets[0] : socket); }
			inline Message_Info get_message_user(SOCKET socket) const { return m_get_message_user(socket); }
			inline void send_message(SOCKET socket, std::string message) const { m_send_message(socket, message); }

			inline std::string get_file(bool overwrite = false, std::string path = std::string(), SOCKET socket = 0) const {
				return m_get_file(!socket ? sockets[0] : socket, overwrite, path);
			}
			inline void send_file(File file, SOCKET socket = 0) const { m_send_file(!socket ? sockets[0] : socket, file); }

			template <typename type>
			inline void send_vector(SOCKET socket, const std::vector<type>& vector) const { return m_send_vector<type>(socket, vector); }
			template <typename type>
			inline std::vector<type> get_vector(SOCKET socket) const { return m_get_vector<type>(socket); }
			inline std::string get_data(SOCKET socket, uint64_t size) const { return m_get_data(socket, size); }
			inline void send_data(SOCKET socket, const char* data, uint64_t size) const { m_send_data(socket, data, size); }

			void redirect_message(SOCKET socket) {
				std::string sender_username = m_get_message(
					socket
				);
				std::string receiver_username = m_get_message(
					socket
				);
				if (users.find(receiver_username.c_str()) == users.end()) {
					printf("%s offline\n", receiver_username.c_str());
					return;
				}
				int receiver = users[receiver_username];

				packet_type type = packet_type::send_message_user;
				m_send_message(sockets[receiver], std::string((const char*)&type, sizeof(type)));

				//Message_Info info = m_get_message_user(socket);

				m_send_message(sockets[receiver], sender_username);
				m_send_message(sockets[receiver], m_get_message(socket));
				printf(
					"message redirected from %s to %s",
					sender_username.c_str(),
					receiver_username.c_str()
				);
			}

			void process_handler(SOCKET socket = 0) {
				socket = !socket ? sockets[socket] : socket;

				while (1) {
					packet_type job = *(packet_type*)m_get_message(socket).data();
					if ((long unsigned int)eti(job) >= sizeof(packet_type_str) / sizeof(*packet_type_str)) {
						continue;
					}
					printf("packet type: %s\n", packet_type_str[eti(job)]);

					switch (job) {
					case packet_type::send_file: {
						packet_type type = packet_type::send_file;
						std::string username = m_get_message(socket);
						auto found = users.find(username);
						SOCKET receiver;
						if (found != users.end()) {
							receiver = sockets[found->second];
						}
						else {
							printf("%s is offline\n", username.c_str());
							break;
						}
						m_send_message(receiver, std::string((const char*)&type, sizeof(type)));
						m_send_file(receiver, m_get_file(socket, false, std::string("temp")));
						break;
					}
					case packet_type::send_message: {
						std::cout << "message received from client: " << get_message(socket) << std::endl;
						break;
					}
					case packet_type::send_registeration: {
						m_get_registeration(socket);
						break;
					}
					case packet_type::send_message_user: {
						redirect_message(socket);
						break;
					}
					case packet_type::exit: {
						std::string user = m_get_message(socket);
						send_message(socket, "exit");
						printf("%s has left\n", user.c_str());
		#ifdef FAN_WINDOWS
						closesocket(socket);
		#else
						close(socket);
		#endif
						return;
					}
					default: {
						break;
					}
					}
				}
			}

			unsigned short convert_endian(unsigned short port) {
				return (unsigned short)port << 8 | (unsigned short)port >> 8;
			}

			constexpr bool quit() const {
				return exit_program;
			}

			inline uint64_t socket_size() const {
				return sockets.size();
			}

			inline bool empty() const {
				return socket_size();
			}

		private:

			void initialize(unsigned short port, bool multithread = false) {
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

				for (uint_t i = 0; !multithread ? i < 1 : 1; i++) {
					SOCKET sock = accept(server_socket, NULL, NULL);
					if (sock == INVALID_SOCKET) {
						LOG("Socket error");
						exit(1);
					}
					sockets.push_back(sock);
					if (multithread) {
						std::string username = m_get_message(sock).c_str();
						printf("%s joined\n", username.c_str());
						users[username] = i;
					}
					else {
						puts("connected");
					}

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

		class tcp_client : public data_handler {
		public:
			tcp_client(const char* ip, unsigned short port, std::string username = std::string()) {
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

			~tcp_client() {
				packet_type type = packet_type::exit;
				send_message(std::string((const char*)&type, sizeof(type)));
				send_message(get_username());

		#ifdef FAN_WINDOWS
				closesocket(connect_socket);
				WSACleanup();
		#else
				close(connect_socket);
		#endif
			}

			void process_handler() {
				while (1) {
					packet_type job = *(packet_type*)m_get_message(connect_socket).data();
					printf("packet type: %s\n", packet_type_str[eti(job)]);

					switch (job) {
					case packet_type::send_file: {
						m_get_file(connect_socket, false);
						break;
					}
					case packet_type::send_message: {
						m_get_message(connect_socket);
						break;
					}
					case packet_type::send_registeration: {
						m_get_registeration(connect_socket);
						break;
					}
					case packet_type::send_message_user: {
						Message_Info info = m_get_message_user(connect_socket);
						printf("received message from %s: %s\n", info.get_username().c_str(), info.message.c_str());
						break;
					}
					default: {
						break;
					}
					}
				}
			}

			inline std::string get_message() const { return m_get_message(connect_socket); }
			inline Message_Info get_message_user() const { return m_get_message_user(connect_socket); }
			inline void send_message(std::string message) const { m_send_message(connect_socket, message); }
			inline void send_message(const std::string& message, const std::string& user) { m_send_message(connect_socket, message, user); }

			template <typename type>
			inline void send_vector(const std::vector<type>& vector) const { m_send_vector<type>(connect_socket, vector); }
			template <typename type>
			inline std::vector<type> get_vector() const { return m_get_vector<type>(connect_socket); }

			inline void get_file(bool overwrite = false) const { m_get_file(connect_socket, overwrite); }
			inline void send_file(File file, const std::string& username) const { m_send_file(connect_socket, file, username); }
			inline void send_file(File file) const { m_send_file(connect_socket, file); }

			inline std::string get_data(uint64_t size) const { return m_get_data(connect_socket, size); }
			inline void send_data(const char* data, uint64_t size) const { m_send_data(connect_socket, data, size); }

		private:
			SOCKET connect_socket;
		};

		class UDP {
		public:
			UDP() : _socket(-1), socket_length(-1), this_sockaddr() {}

			void send_message(const std::string& message) const {
				if (sendto(_socket, message.c_str(), message.size(), 0, (sockaddr*)&this_sockaddr, socket_length) == SOCKET_ERROR) {
		#ifdef FAN_WINDOWS
					printf("failed to send data %d\n", WSAGetLastError());
		#else
					puts("failed to send data %d");
		#endif
					exit(EXIT_FAILURE);
				}
			}

			void send_messagen(const std::string& message) const {
				send_message((const char*)message.size());
				send_message(message);
			}

			void listen(std::function<void()> execute = std::function<void()>([]() {})) {
				std::vector<char> buffer(max_packet_size);
		#ifdef FAN_WINDOWS
				if (recvfrom(_socket, &buffer[0], buffer.size(), 0, (sockaddr*)&this_sockaddr, &socket_length) == SOCKET_ERROR) {
					printf("failed to receive %d\n", WSAGetLastError());
		#else
				if (recvfrom(_socket, &buffer[0], buffer.size(), 0, (sockaddr*)&this_sockaddr, (unsigned int*)&socket_length) == SOCKET_ERROR) {
					puts("failed to send data %d");
		#endif
					exit(EXIT_FAILURE);
				}
				data.append(buffer.cbegin(), buffer.cend());
				execute();
			}

			std::string data;
		protected:
			SOCKET _socket;
			int socket_length;
			sockaddr_in this_sockaddr;
		};

		class udp_client : public UDP {
		public:
			udp_client(const char* ip, unsigned short port) {
				socket_length = sizeof(this_sockaddr);
				if ((_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == INVALID_SOCKET) {
					printf("failed to create socket\n");
				}
				memset((char*)&this_sockaddr, 0, sizeof(this_sockaddr));
				this_sockaddr.sin_family = AF_INET;
		#ifdef _WIN64
				this_sockaddr.sin_addr.S_un.S_addr = inet_addr(ip);
		#else
				this_sockaddr.sin_addr.s_addr = inet_addr(ip);
		#endif
				this_sockaddr.sin_port = htons(port);
			}
			~udp_client() {
		#ifdef FAN_WINDOWS
				closesocket(_socket);
				WSACleanup();
		#else
				close(_socket);
		#endif
			}
		};

		class udp_server : public UDP {
		public:
			udp_server(unsigned short port) {
				socket_length = sizeof(this_sockaddr);

				if ((_socket = socket(AF_INET, SOCK_DGRAM, 0)) == INVALID_SOCKET) {
					printf("failed to create socket\n");
				}

				this_sockaddr.sin_family = AF_INET;
				this_sockaddr.sin_addr.s_addr = INADDR_ANY;
				this_sockaddr.sin_port = htons(port);

				if (bind(_socket, (sockaddr*)&this_sockaddr, sizeof(this_sockaddr)) == SOCKET_ERROR) {
					printf("bind failed\n");
				}
			}

			~udp_server() {
		#ifdef FAN_WINDOWS
				closesocket(_socket);
				WSACleanup();
		#else
				close(_socket);
		#endif
			}
		};
	}
}