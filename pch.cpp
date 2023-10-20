#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include "pch.h"

#if defined(loco_window)
void fan::opengl::viewport_t::set(const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
  viewport_position = viewport_position_;
  viewport_size = viewport_size_;

  gloco->get_context()->opengl.call(
    gloco->get_context()->opengl.glViewport,
    viewport_position.x, window_size.y - viewport_size.y - viewport_position.y,
    viewport_size.x, viewport_size.y
  );
}

#if defined(loco_window)
loco_t::image_list_NodeReference_t::image_list_NodeReference_t(loco_t::image_t* image) {
  NRI = image->texture_reference.NRI;
}

loco_t::camera_list_NodeReference_t::camera_list_NodeReference_t(loco_t::camera_t* camera) {
  NRI = camera->camera_reference.NRI;
}

namespace fan::opengl {
  // Primary template for the constructor
  theme_list_NodeReference_t::theme_list_NodeReference_t(void* theme) {
    //static_assert(std::is_same_v<decltype(theme), loco_t::theme_t*>, "invalid parameter passed to theme");
    NRI = ((loco_t::theme_t*)theme)->theme_reference.NRI;
  }
}

fan::opengl::viewport_list_NodeReference_t::viewport_list_NodeReference_t(fan::opengl::viewport_t* viewport) {
  NRI = viewport->viewport_reference.NRI;
}

#endif
#endif