#include <FAN/Network.hpp>

int main(int argc, char** argv) {
#ifdef _WIN64
	init_winsock();
#endif
	client client(SERVER_IP, PORT);
	printf("connected\n");
	client.send_file(File(argv[1]));
	return 0;
}