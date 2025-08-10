CF = -std=c++20 -Wall -Wextra -Werror -Wno-unused-parameter
LINK = -lev -lpthread -lcrypto -lssl
IO = ecps_server.cpp -o ecps_server.exe
REST = $(CF) $(IO) $(LINK)

debug:
	clang++ -g -Dset_Verbose=1 $(REST)

release:
	clang++ -s -O3 -Dset_Verbose=0 $(REST)
