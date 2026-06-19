module;

#if defined(FAN_VULKAN)
#if defined(fan_platform_windows)
#define VK_USE_PLATFORM_WIN32_KHR
#elif defined(fan_platform_unix)
#define VK_USE_PLATFORM_XLIB_KHR
#endif
#if defined(FAN_GUI)
#include <fan/imgui/imgui_impl_vulkan.h>
#endif
#define loco_window
#include <vulkan/vulkan.h>
#include <shaderc/shaderc.hpp>
#if defined(fan_platform_windows)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WGL
#define GLFW_NATIVE_INCLUDE_NONE
#endif
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#endif

module fan.graphics.vulkan.core;

import std;

#if defined(FAN_VULKAN)

import fan.types.fstring;
import fan.types.color;

#if defined(loco_window)
import fan.window;
#endif

import fan.utility;
import fan.print;
import fan.print.error;
import fan.graphics.image_load;
import fan.graphics.common_context;

import fan.math;
import fan.math.intersection;

#define __fan_internal_camera_list (*fan::graphics::ctx().camera_list)
#define __fan_internal_shader_list (*fan::graphics::ctx().shader_list)
#define __fan_internal_image_list (*fan::graphics::ctx().image_list)
#define __fan_internal_viewport_list (*fan::graphics::ctx().viewport_list)

#if defined(fan_compiler_msvc)
#pragma comment(lib, "vulkan-1.lib")
#pragma comment(lib, "shaderc_combined_mt.lib")
#endif

#define ENABLE_RAYTRACING_DEPENDENCIES

#define VK_CTX ((fan::vulkan::context_t*)context)

fan::graphics::camera_nr_t fan::vulkan::context_t::camera_create() {
  return __fan_internal_camera_list.NewNode();
}
fan::graphics::context_camera_t& fan::vulkan::context_t::camera_get(fan::graphics::camera_nr_t nr) {
  return __fan_internal_camera_list[nr];
}
void fan::vulkan::context_t::camera_erase(fan::graphics::camera_nr_t nr) {
  __fan_internal_camera_list.Recycle(nr);
}
void fan::vulkan::context_t::camera_set_ortho(fan::graphics::camera_nr_t nr, fan::vec2 x, fan::vec2 y) {
  camera_get(nr).coordinates.v = fan::vec4(x, y);
  camera_update_projection(nr);
  camera_update_view(nr);
}
void fan::vulkan::context_t::camera_update_projection(fan::graphics::camera_nr_t nr) {
  auto& camera = camera_get(nr);

  camera.projection = fan::math::ortho<fan::mat4>(
    camera.coordinates.left / camera.zoom,
    camera.coordinates.right / camera.zoom,
    camera.coordinates.top / camera.zoom,
    camera.coordinates.bottom / camera.zoom,
    -fan::graphics::znearfar / 2,
    fan::graphics::znearfar / 2
  );
}
void fan::vulkan::context_t::camera_update_view(fan::graphics::camera_nr_t nr) {
  auto& camera = camera_get(nr);
  camera.view[3][0] = 0;
  camera.view[3][1] = 0;
  camera.view[3][2] = 0;
  camera.view = camera.view.translate(camera.position);
  fan::vec3 position = camera.view.get_translation();
  constexpr fan::vec3 front(0, 0, 1);
  camera.view = fan::math::look_at_left<fan::mat4, fan::vec3>(position, position + front, fan::camera::world_up);
}
fan::graphics::camera_nr_t fan::vulkan::context_t::camera_create(const fan::vec2& x, const fan::vec2& y) {
  fan::graphics::camera_nr_t nr = camera_create();
  camera_set_ortho(nr, fan::vec2(x.x, x.y), fan::vec2(y.x, y.y));
  return nr;
}
fan::vec3 fan::vulkan::context_t::camera_get_position(fan::graphics::camera_nr_t nr) {
  return camera_get(nr).position;
}
void fan::vulkan::context_t::camera_set_position(fan::graphics::camera_nr_t nr, const fan::vec3& cp) {
  auto& camera = camera_get(nr);
  camera.position = cp;
  camera_update_view(nr);
}
fan::vec3 fan::vulkan::context_t::camera_get_center(fan::graphics::camera_nr_t nr) {
  auto& c = camera_get(nr);
  fan::vec2 center_offset = fan::vec2(
    c.coordinates.left + c.coordinates.right,
    c.coordinates.top + c.coordinates.bottom
  ) / (2.f * c.zoom);
  return fan::vec2(c.position.x, c.position.y) + center_offset;
}
void fan::vulkan::context_t::camera_set_center(fan::graphics::camera_nr_t nr, const fan::vec3& cp) {
  auto& c = camera_get(nr);
  fan::vec2 center_offset = fan::vec2(
    c.coordinates.left + c.coordinates.right,
    c.coordinates.top + c.coordinates.bottom
  ) / (2.f * c.zoom);

  camera_set_position(nr, fan::vec3(cp.xy() - center_offset, cp.z));
}
fan::vec2 fan::vulkan::context_t::camera_get_size(fan::graphics::camera_nr_t nr) {
  fan::graphics::context_camera_t& camera = camera_get(nr);
  return fan::vec2(std::abs(camera.coordinates.right - camera.coordinates.left), std::abs(camera.coordinates.bottom - camera.coordinates.top));
}
f32_t fan::vulkan::context_t::camera_get_zoom(fan::graphics::camera_nr_t nr) {
  return camera_get(nr).zoom;
}
void fan::vulkan::context_t::camera_set_zoom(fan::graphics::camera_nr_t nr, f32_t new_zoom) {
  camera_get(nr).zoom = new_zoom;
  camera_update_projection(nr);
  camera_update_view(nr);
}
void fan::vulkan::context_t::camera_set_perspective(fan::graphics::camera_nr_t nr, f32_t fov, const fan::vec2& window_size) {
  fan::graphics::context_camera_t& camera = camera_get(nr);

  camera.fov = fov;
  camera.projection = fan::math::perspective<fan::mat4>(fan::math::radians(camera.fov), (f32_t)window_size.x / (f32_t)window_size.y, camera.znear, camera.zfar);

  camera.update_view();

  camera.view = camera.get_view_matrix();

  //auto it = gloco()->m_viewport_resize_callback.GetNodeFirst();

  //while (it != gloco()->m_viewport_resize_callback.dst) {

  //  gloco()->m_viewport_resize_callback.StartSafeNext(it);

  //  resize_cb_data_t cbd;
  //  cbd.camera = this;
  //  cbd.position = get_position();
  //  cbd.size = get_camera_size();
  //  gloco()->m_viewport_resize_callback[it].data(cbd);

  //  it = gloco()->m_viewport_resize_callback.EndSafeNext();
  //}
}
void fan::vulkan::context_t::camera_rotate(fan::graphics::camera_nr_t nr, const fan::vec2& offset) {
  fan::graphics::context_camera_t& camera = camera_get(nr);
  camera.rotate_camera(offset);
  camera.view = camera.get_view_matrix();
}
//-----------------------------camera-----------------------------

      //-----------------------------viewport-----------------------------

void fan::vulkan::context_t::viewport_set(const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
  VkViewport viewport {};
  viewport.x = viewport_position_.x;
  viewport.y = viewport_position_.y;
  viewport.width = viewport_size_.x;
  viewport.height = viewport_size_.y;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  VkCommandBufferBeginInfo beginInfo {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

  if (!command_buffer_in_use) {
    VkResult result = vkGetFenceStatus(device, in_flight_fences[current_frame]);
    if (result == VK_NOT_READY) {
      vkDeviceWaitIdle(device);
    }

    if (vkBeginCommandBuffer(command_buffers[current_frame], &beginInfo) != VK_SUCCESS) {
      fan::throw_error("failed to begin recording command buffer!");
    }
  }
  vkCmdSetViewport(command_buffers[current_frame], 0, 1, &viewport);

  if (!command_buffer_in_use) {
    if (vkEndCommandBuffer(command_buffers[current_frame]) != VK_SUCCESS) {
      fan::throw_error("failed to record command buffer!");
    }
    command_buffer_in_use = false;
  }
}
fan::graphics::context_viewport_t& fan::vulkan::context_t::viewport_get(fan::graphics::viewport_nr_t nr) {
  return __fan_internal_viewport_list[nr];
}
void fan::vulkan::context_t::viewport_set(fan::graphics::viewport_nr_t nr, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
  fan::graphics::context_viewport_t& viewport = viewport_get(nr);
  viewport.position = viewport_position_;
  viewport.size = viewport_size_;

  viewport_set(viewport_position_, viewport_size_, window_size);
}
fan::graphics::viewport_nr_t fan::vulkan::context_t::viewport_create() {
  auto nr = __fan_internal_viewport_list.NewNode();

  viewport_set(nr, 0, 1, 0);
  return nr;
}
void fan::vulkan::context_t::viewport_erase(fan::graphics::viewport_nr_t nr) {
  __fan_internal_viewport_list.Recycle(nr);
}
fan::vec2 fan::vulkan::context_t::viewport_get_position(fan::graphics::viewport_nr_t nr) {
  return viewport_get(nr).position;
}
fan::vec2 fan::vulkan::context_t::viewport_get_size(fan::graphics::viewport_nr_t nr) {
  return viewport_get(nr).size;
}
void fan::vulkan::context_t::viewport_zero(fan::graphics::viewport_nr_t nr) {
  auto& viewport = viewport_get(nr);
  viewport.position = 0;
  viewport.size = 0;
  viewport_set(0, 0, 0); // window_size not used
}
bool fan::vulkan::context_t::viewport_inside(fan::graphics::viewport_nr_t nr, const fan::vec2& position) {
  fan::graphics::context_viewport_t& viewport = viewport_get(nr);
  return fan::math::d2::aabb_point_inside(position, viewport.position + viewport.size / 2, viewport.size / 2);
}
bool fan::vulkan::context_t::viewport_inside_wir(fan::graphics::viewport_nr_t nr, const fan::vec2& position) {
  fan::graphics::context_viewport_t& viewport = viewport_get(nr);
  return fan::math::d2::aabb_point_inside(position, viewport.size / 2, viewport.size / 2);
}
#endif