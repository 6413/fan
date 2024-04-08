#include "gl_viewport.h"

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

//#define loco_imgui

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif

#ifndef FAN_INCLUDE_PATH
#define _FAN_PATH(p0) <fan/p0>
#else
#define FAN_INCLUDE_PATH_END fan/
#define _FAN_PATH(p0) <FAN_INCLUDE_PATH/fan/p0>
#define _FAN_PATH_QUOTE(p0) STRINGIFY_DEFINE(FAN_INCLUDE_PATH) "/fan/" STRINGIFY(p0)
#endif

#if defined(loco_imgui)
#define IMGUI_IMPL_OPENGL_LOADER_CUSTOM
#define IMGUI_DEFINE_MATH_OPERATORS
#include _FAN_PATH(imgui/imgui.h)
#include _FAN_PATH(imgui/imgui_impl_opengl3.h)
#include _FAN_PATH(imgui/imgui_impl_glfw.h)
#include _FAN_PATH(imgui/imgui_neo_sequencer.h)
#endif

#ifndef fan_verbose_print_level
#define fan_verbose_print_level 1
#endif
#ifndef fan_debug
#define fan_debug 0
#endif
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH,fan/types/types.h)

#include <fan/graphics/loco_settings.h>
#include <fan/graphics/loco.h>

void fan::opengl::viewport_t::open() {
  viewport_reference = gloco->get_context().viewport_list.NewNode();
  gloco->get_context().viewport_list[viewport_reference].viewport_id = this;
}

void fan::opengl::viewport_t::close() {
  gloco->get_context().viewport_list.Recycle(viewport_reference);
}

void fan::opengl::viewport_t::set_viewport(const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
  gloco->get_context().opengl.call(
    gloco->get_context().opengl.glViewport,
    viewport_position_.x,
    window_size.y - viewport_size_.y - viewport_position_.y,
    viewport_size_.x, viewport_size_.y
  );
}

void fan::opengl::viewport_t::zero() {
  viewport_position = 0;
  viewport_size = 0;
  gloco->get_context().opengl.call(
    gloco->get_context().opengl.glViewport,
    0, 0, 0, 0
  );
}

void fan::opengl::viewport_t::set(const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
  viewport_position = viewport_position_;
  viewport_size = viewport_size_;

  gloco->get_context().opengl.call(
    gloco->get_context().opengl.glViewport,
    viewport_position.x, window_size.y - viewport_size.y - viewport_position.y,
    viewport_size.x, viewport_size.y
  );
}