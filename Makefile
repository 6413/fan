# compiler settings
CXX = clang++ -Wall -Wextra
CXXFLAGS =  -I .-ferror-limit=3 -w -std=c++2a -I include -ftime-trace

# source file and output
MAIN ?=  examples/graphics/2D/circle.cpp
OUTPUT ?= a.exe

WINDOWS_ROOT_PATH = ./
LINUX_ROOT_PATH=./

LINUX_ROOT_LIB_PATH=

WITCH_WINDOWS_ROOT_PATH = ./
WITCH_LINUX_ROOT_PATH = ./

FAN_INCLUDE_PATH =
LINK_PATH = lib/fan/

# precompiled header file
PCH_NAME = pch.h

DEBUG_FLAGS = -g
RELEASE_FLAGS = -s -mmmx -msse -msse2 -msse3 -mssse3 -msse4 -msse4.1 -O3 -fdata-sections -ffunction-sections
CXX = clang++ -Wall -Wextra
CXXFLAGS = -ferror-limit=3 -w -std=c++2a -I include -ftime-trace

# source file and output
MAIN ?=  examples/graphics/circle.cpp
OUTPUT ?= a.exe

# includes & link
WINDOWS_INCLUDES = -I $(WINDOWS_ROOT_PATH)fan/include
LINUX_INCLUDES = -I $(LINUX_ROOT_PATH)

WINDOWS_LINK = $(LINUX_ROOT_LIB_PATH)$(LINK_PATH)pch.o lib/libuv/uv_a.lib lib/fan/libimgui.lib lib/libwebp/libwebp.lib lib/opus/libopus.a lib/GLFW/glfw3_mt.lib "C:\Program Files\Assimp\Lib\x64\assimp-vc143-mt.lib"
LINUX_LINK = $(LINUX_ROOT_LIB_PATH)$(LINK_PATH)pch.o -lX11 -lXrandr -lopus -L /usr/lib/x86_64-linux-gnu/libGL.so.1 -lwebp -ldl $(LINUX_ROOT_LIB_PATH)$(LINK_PATH)libimgui.a -lglfw



INCLUDES = -I .
LINK = 

ifeq ($(OS),Windows_NT)
	FAN_INCLUDE_PATH += $(WINDOWS_ROOT_PATH)include
	INCLUDES += $(WINDOWS_INCLUDES)
	LINK += $(WINDOWS_LINK)
	CXXFLAGS += -I $(FAN_INCLUDE_PATH)
else
	FAN_INCLUDE_PATH += $(LINUX_ROOT_PATH)
	INCLUDES += $(LINUX_INCLUDES)
	LINK += $(LINUX_LINK)
	CXXFLAGS += -fuse-ld=gold
endif

CXXFLAGS += -DFAN_INCLUDE_PATH=$(FAN_INCLUDE_PATH)
CXXFLAGS += -Dfan_pch=\"$(FAN_INCLUDE_PATH)/fan/$(PCH_NAME)\"

debug:
	$(CXX) $(CXXFLAGS) $(DEBUG_FLAGS) -include-pch $(LINK_PATH)$(PCH_NAME).gch $(INCLUDES) $(MAIN) -o $(OUTPUT) $(LINK)

release:
	$(CXX) $(CXXFLAGS) $(RELEASE_FLAGS) -include-pch $(LINK_PATH)$(PCH_NAME).gch $(INCLUDES) $(MAIN) -o $(OUTPUT) $(LINK) 

clean:
	rm -f $(OUTPUT)

.PHONY: debug release clean