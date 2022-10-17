#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#define fan_debug 3
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define loco_vulkan

#include _FAN_PATH(graphics/graphics.h)

//#include _FAN_PATH(graphics/vulkan/vk_core.h)

int main() {
	fan::window_t window;
	window.open();
	fan::vulkan::context_t context;
	context.open();
	context.bind_to_window(&window);

	while (1) {
		context.drawFrame(&window);
		window.handle_events();
	}
}