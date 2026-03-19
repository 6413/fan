#pragma once

// renderer dispatch
#if defined(FAN_OPENGL)
  #define IF_GL(...) __VA_ARGS__
#else
  #define IF_GL(...)
#endif
#if defined(FAN_VULKAN)
  #define IF_VK(...) __VA_ARGS__
#else
  #define IF_VK(...)
#endif

#if defined(FAN_GUI)
  #define IF_GUI(...) __VA_ARGS__
#else
  #define IF_GUI(...)
#endif

#define renderer_call(gl_expr, vk_expr) \
  do { \
    IF_GL(if (fan::graphics::get_window().renderer == fan::window_t::renderer_t::opengl) { gl_expr; }) \
    IF_VK(if (fan::graphics::get_window().renderer == fan::window_t::renderer_t::vulkan) { vk_expr; }) \
  } while(0)

#define renderer_call_ret(gl_expr, vk_expr) \
  IF_GL(if (fan::graphics::get_window().renderer == fan::window_t::renderer_t::opengl) { return gl_expr; }) \
  IF_VK(if (fan::graphics::get_window().renderer == fan::window_t::renderer_t::vulkan) { return vk_expr; })

// shape push_back for trivial shapes
#define DEFINE_PUSH_BACK(name) \
  shapes::shape_t shapes::name##_t::push_back(const properties_t& p) { \
    return make_shape_ret(g_shapes->add_shape(g_shapes->name##_list, p).NRI); \
  }

// json field emit/load macros
#define SHAPE_JSON_EMIT(name) \
  if (shape.get_##name() != defaults.name) out[#name] = shape.get_##name()

#define SHAPE_JSON_EMIT_CUSTOM(key, getter, default_field) \
  if (shape.getter() != defaults.default_field) out[key] = shape.getter()

#define SHAPE_JSON_GET(field) in.get_if(#field, p.field)

#define SHAPE_PROP_SIMPLE(ret, name) \
  ret shapes::shape_t::get_##name() const { \
    return g_shapes->shape_functions[get_shape_type()].get_##name(this); \
  } \
  void shapes::shape_t::set_##name(const ret& v) { \
    g_shapes->shape_functions[get_shape_type()].set_##name(this, v); \
  }

#define SHAPE_PROP_CULLING(ret, name) \
  ret shapes::shape_t::get_##name() const { \
    return g_shapes->shape_functions[get_shape_type()].get_##name(this); \
  } \
  void shapes::shape_t::set_##name(const ret& v) { \
    g_shapes->shape_functions[get_shape_type()].set_##name(this, v); \
    update_culling(); \
  }