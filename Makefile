GPP = clang++

CFLAGS = -std=c++2a -I /usr/local/include -I include #-O3 -march=native -mtune=native

FAN_OBJECT_FOLDER = 

all: fan_window.o fan_window_input.o fan_shared_gui.o fan_shared_graphics.o run

LIBS = fan_window.o fan_window_input.o fan_shared_gui.o fan_shared_graphics.o

LIBNAME = fan.a

fan_window.o:  src/fan/window/window.cpp
	$(GPP) $(CFLAGS) -c src/fan/window/window.cpp -o $(FAN_OBJECT_FOLDER)fan_window.o
	
fan_window_input.o:	src/fan/window/window_input.cpp
	$(GPP) $(CFLAGS) -c src/fan/window/window_input.cpp -o $(FAN_OBJECT_FOLDER)fan_window_input.o

fan_shared_gui.o:	src/fan/graphics/shared_gui.cpp
	$(GPP) $(CFLAGS) -c src/fan/graphics/shared_gui.cpp -o $(FAN_OBJECT_FOLDER)fan_shared_gui.o

fan_shared_graphics.o:	src/fan/graphics/shared_graphics.cpp
	$(GPP) $(CFLAGS) -c src/fan/graphics/shared_graphics.cpp -o fan_shared_graphics.o	
clean:
	rm -f fan_*.o

run:	$(LIBS)
	ar rvs $(LIBNAME) $(LIBS)
	#rm -f fan_*.o