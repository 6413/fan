module;

#if defined(FAN_2D)


#if defined(fan_platform_windows)
  #define VK_USE_PLATFORM_WIN32_KHR
#elif defined(fan_platform_unix)
  #define VK_USE_PLATFORM_XLIB_KHR
#endif
#include <vulkan/vulkan.h>
#include <fan/utility.h>

export module fan.graphics.vulkan.core:camera_subsystem;
import std;

import fan.types;
import fan.types.vector;
import fan.graphics.common_context;

export namespace fan::vulkan {
  struct context_t;

  struct camera_subsystem_t {
    context_t* ctx = nullptr;

    void init(context_t& context) { ctx = &context; }

    fan::graphics::camera_nr_t camera_create();
    fan::graphics::context_camera_t& camera_get(fan::graphics::camera_nr_t nr);
    void camera_erase(fan::graphics::camera_nr_t nr);
    void camera_set_ortho(fan::graphics::camera_nr_t nr, fan::vec2 x, fan::vec2 y);
    void camera_update_projection(fan::graphics::camera_nr_t nr);
    void camera_update_view(fan::graphics::camera_nr_t nr);
    fan::graphics::camera_nr_t camera_create(const fan::vec2& x, const fan::vec2& y);
    fan::vec3 camera_get_position(fan::graphics::camera_nr_t nr);
    void camera_set_position(fan::graphics::camera_nr_t nr, const fan::vec3& cp);
    fan::vec3 camera_get_center(fan::graphics::camera_nr_t nr);
    void camera_set_center(fan::graphics::camera_nr_t nr, const fan::vec3& cp);
    fan::vec2 camera_get_size(fan::graphics::camera_nr_t nr);
    f32_t camera_get_zoom(fan::graphics::camera_nr_t nr);
    void camera_set_zoom(fan::graphics::camera_nr_t nr, f32_t new_zoom);
    void camera_set_perspective(fan::graphics::camera_nr_t nr, f32_t fov, const fan::vec2& window_size);
    void camera_rotate(fan::graphics::camera_nr_t nr, const fan::vec2& offset);

    void viewport_set(const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size);
    fan::graphics::context_viewport_t& viewport_get(fan::graphics::viewport_nr_t nr);
    void viewport_set(fan::graphics::viewport_nr_t nr, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size);
    fan::graphics::viewport_nr_t viewport_create();
    void viewport_erase(fan::graphics::viewport_nr_t nr);
    fan::vec2 viewport_get_position(fan::graphics::viewport_nr_t nr);
    fan::vec2 viewport_get_size(fan::graphics::viewport_nr_t nr);
    void viewport_zero(fan::graphics::viewport_nr_t nr);
    bool viewport_inside(fan::graphics::viewport_nr_t nr, const fan::vec2& position);
    bool viewport_inside_wir(fan::graphics::viewport_nr_t nr, const fan::vec2& position);

    VkViewport pending_viewport{};
    VkRect2D pending_scissor{};
    bool viewport_dirty = false;
  };
}

#endif