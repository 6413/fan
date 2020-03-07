#include <FAN/Network.hpp>

int main() {
	init_winsock();

	tcp_server server(PORT);

	server.receive_file();

	return 0;
}