CFLAGS = -w -std=c++2a -Wno-return-type -Wno-unused-variable -Wno-invalid-offsetof

ifeq ($(OS),Windows_NT)
	CFLAGS += -Dfatherload_compile=0
	INCLUDE = -I C:/libs/fan/src/libwebp -I C:/libs/fan/src/libwebp/src -I C:/libs/fan/include -I C:/libs/fan/include/opus
	LINK = C:/libs/fan/fan_windows_clang.a C:/libs/fan/lib/libwebp/libwebp.a
	RM = del
	OUTPUT ="..\..\fatherload.exe"
else
	CFLAGS += -Dfatherload_compile=1
	INCLUDE = -I /usr/include/opus/
	LINK = -lasound -lpthread -lopus -lGL -lX11 -lXrandr -lwebp -ldl fan.a
	RM = rm
	OUTPUT = ../../fatherload
endif

ReleaseFlags = -s -fdata-sections -ffunction-sections -Wl,--gc-sections -mmmx -msse -msse2 -msse3 -mssse3 -msse4 -msse4.1 -O3

debug:
	clang++ $(CFLAGS) -g main.cpp -o $(OUTPUT) $(INCLUDE) $(LINK)

release:
	clang++ $(CFLAGS) $(ReleaseFlags) main.cpp -o $(OUTPUT) $(INCLUDE) $(LINK)

clean:
	$(RM) $(OUTPUT)
