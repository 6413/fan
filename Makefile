# compiler settings
CXX = clang++
CXXFLAGS = -ferror-limit=3 -w -std=c++2a -I include -ftime-trace

# source file and output
MAIN = examples/gui/tile_map_editor.cpp
OUTPUT = a.out

WINDOWS_ROOT_PATH=C:/libs/
LINUX_ROOT_PATH=/usr/include/

LINUX_ROOT_LIB_PATH=/usr/include/

FAN_INCLUDE_PATH =
LINK_PATH = lib/fan/

# precompiled header file
PCH_NAME = pch.h

DEBUG_FLAGS = 
RELEASE_FLAGS = -s -mmmx -msse -msse2 -msse3 -mssse3 -msse4 -msse4.1 -O3 -fdata-sections -ffunction-sections -Wl,--gc-sections

# includes & link
WINDOWS_INCLUDES = -I $(WINDOWS_ROOT_PATH)fan/include -I $(WINDOWS_ROOT_PATH)/fan/include/baseclasses -I $(WINDOWS_ROOT_PATH)/fan/src/libwebp -I $(WINDOWS_ROOT_PATH)/fan/src/libwebp/src
LINUX_INCLUDES = -I $(LINUX_ROOT_PATH)

WINDOWS_LINK = lib/libuv/uv_a.lib $(WINDOWS_ROOT_PATH)fan/$(LINK_PATH)libimgui.a lib/libwebp/libwebp.a lib/opus/libopus.a
LINUX_LINK = -lX11 -lXrandr -lopus -L /usr/lib/x86_64-linux-gnu/libGL.so.1 -lwebp -ldl $(LINUX_ROOT_LIB_PATH)/fan/$(LINK_PATH)libimgui.a

INCLUDES =
LINK = 

ifeq ($(OS),Windows_NT)
	FAN_INCLUDE_PATH += $(WINDOWS_ROOT_PATH)/fan/include/
	INCLUDES += $(WINDOWS_INCLUDES)
	LINK += $(WINDOWS_LINK)
	CXXFLAGS += -DWITCH_INCLUDE_PATH=$(WINDOWS_ROOT_PATH)WITCH
	CXXFLAGS += -I $(FAN_INCLUDE_PATH)
else
	FAN_INCLUDE_PATH += $(LINUX_ROOT_PATH)
	INCLUDES += $(LINUX_INCLUDES)
	LINK += $(LINUX_LINK)
	CXXFLAGS += -fuse-ld=gold
	CXXFLAGS += -DWITCH_INCLUDE_PATH=/usr/include/WITCH
	CXXFLAGS += -I $(FAN_INCLUDE_PATH)
endif

CXXFLAGS += -Dfan_pch=\"$(FAN_INCLUDE_PATH)fan/$(PCH_NAME)\"

debug:
	$(CXX) $(CXXFLAGS) $(DEBUG_FLAGS) -include-pch $(LINK_PATH)$(PCH_NAME).gch $(INCLUDES) $(MAIN) -o $(OUTPUT) $(LINK) 

release:
	$(CXX) $(CXXFLAGS) $(RELEASE_FLAGS) -include-pch $(LINK_PATH)$(PCH_NAME).gch $(INCLUDES) $(MAIN) -o $(OUTPUT) $(LINK) 

clean:
	rm -f $(OUTPUT)

.PHONY: debug release clean