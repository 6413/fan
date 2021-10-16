#pragma once

namespace fan {

	inline std::vector<std::function<void()>> g_next_loop_queue;

	// used for instance in erase which needs to be queued for next frame in vulkan
	inline void push_for_next_loop(std::function<void()> function) {
		g_next_loop_queue.emplace_back(function);
	}
}

namespace fan_2d {

    namespace graphics {


        enum class shape {
			line,
			line_strip,
			triangle,
			triangle_strip,
			triangle_fan,
			last
		};

    }

}