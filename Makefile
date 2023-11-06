#release
GPP = clang++ -I C:\libs -Dstage_loader_path=. -I include/nvidia -I .
#debug for address sanitizer
#GPP = clang-cl /MT -fsanitize=address  /std:c++latest

#-Wall -Wextra -Wshadow -Wconversion -Wpedantic -Werror

CFLAGS = -ferror-limit=3 -w -I .  -std=c++2a -I include #-O3 -march=native -mtune=native \
  #-fsanitize=address -fno-omit-frame-pointer


MAIN = examples/cpp/print.cpp
FAN_OBJECT_FOLDER = 

LINK_PATH = lib/fan/
FAN_INCLUDE_PATH=

ifeq ($(OS),Windows_NT)
	FAN_INCLUDE_PATH += C:/libs/fan/include
  CFLAGS += -I C:\libs\fan\include\baseclasses -I C:/libs/fan/src/libwebp -I C:/libs/fan/src/libwebp/src C:/libs/fan/lib/libwebp/libwebp.a C:/libs/fan/lib/opus/libopus.a
	CFLAGS += -DWITCH_INCLUDE_PATH=C:/libs/WITCH
	CFLAGS += lib/libuv/uv_a.lib lib/imgui/libimgui.lib 
else
	FAN_INCLUDE_PATH += /usr/include/
  CFLAGS += -lX11 -lXrandr -L /usr/local/lib -lopus -L/usr/lib/x86_64-linux-gnu/libGL.so.1 -lwebp -ldl
	CFLAGS += -DWITCH_INCLUDE_PATH=/usr/include/WITCH
	CFLAGS += $(LINK_PATH)libimgui.a
endif

CFLAGS += -DFAN_INCLUDE_PATH=$(FAN_INCLUDE_PATH) 

PCH_NAME = pch.h
CFLAGS += -Dfan_pch=\"$(FAN_INCLUDE_PATH)fan/$(PCH_NAME)\"

debug:
	$(GPP) $(CFLAGS) -include-pch $(LINK_PATH)$(PCH_NAME).gch  $(MAIN) 

release:
	$(GPP) $(CFLAGS) -include-pch $(LINK_PATH)$(PCH_NAME).gch $(MAIN) -fdata-sections -ffunction-sections -Wl,--gc-sections -mmmx -msse -msse2 -msse3 -mssse3 -msse4 -msse4.1 -O3 lib/imgui/libimgui.lib

clean:
	rm -f a.out
