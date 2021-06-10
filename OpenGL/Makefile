GPP = clang++

CFLAGS = -std=c++2a -I /usr/local/include -I include #-O3 -march=native -mtune=native
MAINFLAGS = -L /usr/local/lib -lGL -lGLEW -lX11 -lXrandr -lassimp -lavcodec -lavutil -lavformat -lswscale box2d.o

MAIN = Source.cpp

TARGET = a.out

default : $(TARGET)

$(TARGET): fan_main.o fan_window.o fan_window_input.o fan_camera.o fan_graphics.o fan_gui.o
	$(GPP) $(CFLAGS) fan_main.o fan_window.o fan_window_input.o fan_camera.o fan_graphics.o fan_gui.o $(MAINFLAGS) -o $(TARGET)
	
fan_main.o:	$(MAIN)
	$(GPP) $(CFLAGS) -c $(MAIN) -o fan_main.o

fan_window.o:	src/fan/graphics/window/window.cpp
	$(GPP) $(CFLAGS) -c src/fan/graphics/window/window.cpp -o fan_window.o
	
fan_window_input.o:	src/fan/graphics/window/window_input.cpp
	$(GPP) $(CFLAGS) -c src/fan/graphics/window/window_input.cpp -o fan_window_input.o
	
fan_camera.o:	src/fan/graphics/camera.cpp
	$(GPP) $(CFLAGS) -c src/fan/graphics/camera.cpp -o fan_camera.o
	
fan_graphics.o:	src/fan/graphics/graphics.cpp
	$(GPP) $(CFLAGS) -c src/fan/graphics/graphics.cpp -o fan_graphics.o
	
fan_gui.o:	src/fan/graphics/gui.cpp
	$(GPP) $(CFLAGS) -c src/fan/graphics/gui.cpp -o fan_gui.o
	
clean:
	rm -f fan_*.o
