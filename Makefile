GPP = clang++

DEBUGFLAGS = -g
RELEASEFLAGS = -s -fdata-sections -ffunction-sections -Wl,--gc-sections -mmmx -msse -msse2 -msse3 -mssse3 -msse4 -msse4.1 -O3

CFLAGS = -w -std=c++2a -I /usr/local/include -I include -g -Wl #-O3 -march=native -mtune=native $(RELEASEFLAGS)

BASE_PATH = 

ifeq ($(OS),Windows_NT)
	AR = llvm-ar
	RM = del 
	LIBNAME = fan_windows_clang.a
  BASE_PATH += lib/fan/
else
	CFLAGS += -DFAN_INCLUDE_PATH=/usr/include -fPIE
	AR = ar
	RM = rm -f
	LIBNAME = fan.a
  FAN_OBJECT_FOLDER = $(BASE_PATH)
endif

ifeq ($(OS),Windows_NT)
  #                        magic - replace / with \ thanks to windows
  FAN_OBJECT_FOLDER = $(subst /,\,$(BASE_PATH))
endif

all: fan_window.o fan_window_input.o run

LIBS = $(FAN_OBJECT_FOLDER)fan_window.o $(FAN_OBJECT_FOLDER)fan_window_input.o

fan_window.o:  src/fan/window/window.cpp
	$(GPP) $(CFLAGS) -c src/fan/window/window.cpp -o $(FAN_OBJECT_FOLDER)fan_window.o
	
fan_window_input.o:	src/fan/window/window_input.cpp
	$(GPP) $(CFLAGS) -c src/fan/window/window_input.cpp -o $(FAN_OBJECT_FOLDER)fan_window_input.o

clean:
	$(RM) $(FAN_OBJECT_FOLDER)fan_*.o

run:	$(LIBS)
	$(AR) rvs $(FAN_OBJECT_FOLDER)$(LIBNAME) $(LIBS)
