GPP = clang++

CFLAGS = -std=c++2a -I /usr/local/include -I include -O3 -march=native -mtune=native

FAN_OBJECT_FOLDER = graphics/opengl/

all: fan_window.o fan_window_input.o fan_camera.o fan_gl_graphics.o fan_gl_gui.o fan_vk_gui.o fan_shared_gui.o fan_shared_graphics.o

fan_window.o:  window/window.cpp
	$(GPP) $(CFLAGS) -c window/window.cpp -o $(FAN_OBJECT_FOLDER)fan_window.o

fan_window_input.o:	window/window_input.cpp
	$(GPP) $(CFLAGS) -c window/window_input.cpp -o $(FAN_OBJECT_FOLDER)fan_window_input.o

fan_camera.o:	graphics/camera.cpp
	$(GPP) $(CFLAGS) -c graphics/camera.cpp -o $(FAN_OBJECT_FOLDER)fan_camera.o

fan_gl_graphics.o:	graphics/opengl/gl_graphics.cpp
	$(GPP) $(CFLAGS) -c graphics/opengl/gl_graphics.cpp -o $(FAN_OBJECT_FOLDER)fan_gl_graphics.o

fan_gl_gui.o:	graphics/opengl/gl_gui.cpp
	$(GPP) $(CFLAGS) -c graphics/opengl/gl_gui.cpp -o $(FAN_OBJECT_FOLDER)fan_gl_gui.o

fan_vk_gui.o:	graphics/vulkan/vk_gui.cpp
	$(GPP) $(CFLAGS) -c graphics/vulkan/vk_gui.cpp -o $(FAN_OBJECT_FOLDER)fan_vk_gui.o

fan_shared_gui.o:	graphics/shared_gui.cpp
	$(GPP) $(CFLAGS) -c graphics/shared_gui.cpp -o $(FAN_OBJECT_FOLDER)fan_shared_gui.o

fan_shared_graphics.o:	graphics/shared_graphics.cpp
	$(GPP) $(CFLAGS) -c graphics/shared_graphics.cpp -o $(FAN_OBJECT_FOLDER)fan_shared_graphics.o

clean:
	rm graphics/opengl/*.o
