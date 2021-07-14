GPP = g++

CFLAGS = -g -std=c++2a -I /usr/local/include -I include #-O3 -march=native -mtune=native
MAINFLAGS = -g -L /usr/local/lib -lGL -lGLEW -lX11 -lXrandr -lassimp -lavcodec -lavutil -lavformat -lswscale -lbox2d -lvulkan

MAIN = Source.cpp

TARGET = a.out

default : $(TARGET)

LIBS = fan_main.o fan_window.o fan_window_input.o fan_camera.o fan_gl_graphics.o fan_gl_gui.o fan_vk_graphics.o fan_vk_gui.o

$(TARGET): $(LIBS)
	$(GPP) $(CFLAGS) $(LIBS) $(MAINFLAGS) -o $(TARGET)
	
fan_main.o:	$(MAIN)
	$(GPP) $(CFLAGS) -c $(MAIN) -o fan_main.o

fan_window.o:	src/fan/window/window.cpp
	$(GPP) $(CFLAGS) -c src/fan/window/window.cpp -o fan_window.o
	
fan_window_input.o:	src/fan/window/window_input.cpp
	$(GPP) $(CFLAGS) -c src/fan/window/window_input.cpp -o fan_window_input.o
	
fan_camera.o:	src/fan/graphics/camera.cpp
	$(GPP) $(CFLAGS) -c src/fan/graphics/camera.cpp -o fan_camera.o
	
fan_gl_graphics.o:	src/fan/graphics/opengl/gl_graphics.cpp
	$(GPP) $(CFLAGS) -c src/fan/graphics/opengl/gl_graphics.cpp -o fan_gl_graphics.o
	
fan_gl_gui.o:	src/fan/graphics/opengl/gl_gui.cpp
	$(GPP) $(CFLAGS) -c src/fan/graphics/opengl/gl_gui.cpp -o fan_gl_gui.o
	
fan_vk_graphics.o:	src/fan/graphics/vulkan/vk_graphics.cpp
	$(GPP) $(CFLAGS) -c src/fan/graphics/vulkan/vk_graphics.cpp -o fan_vk_graphics.o

fan_vk_gui.o:	src/fan/graphics/vulkan/vk_gui.cpp
	$(GPP) $(CFLAGS) -c src/fan/graphics/vulkan/vk_gui.cpp -o fan_vk_gui.o
	
clean:
	rm -f fan_*.o
