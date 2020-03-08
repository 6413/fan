#include <FAN/Network.hpp>

constexpr auto PORT = 43254;

int main() {
#ifdef FAN_WINDOWS
	init_winsock();
#endif

	tcp_server server(PORT);

	server.get_file(server.get_socket(0));

	return 0;
}