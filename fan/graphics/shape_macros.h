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

// internal dispatch — do not use directly
#define _rcc_gl_ctx  fan::graphics::get_gl_context()
#define _rcc_vk_ctx  fan::graphics::get_vk_context()
#define _rcc_renderer fan::graphics::get_window().renderer

#define _rcc_dispatch_void(ctx, func, ...)                                         \
  if constexpr (requires { ctx.func(__VA_ARGS__); }) {                             \
    ctx.func(__VA_ARGS__); return;                                                 \
  } else { fan::throw_error_impl("backend TODO: " #func); }

#define _rcc_dispatch_ret(type, ctx, member, func, ...)                            \
  if constexpr (requires { ctx.func(__VA_ARGS__); }) {                             \
    type obj{}; obj.member = ctx.func(__VA_ARGS__); return obj;                    \
  } else { fan::throw_error_impl("backend TODO: " #func); }

// set — void return
#define renderer_set(func, ...)                                                    \
do {                                                                               \
  IF_GL(if (_rcc_renderer == fan::window_t::renderer_t::opengl)                   \
    { _rcc_dispatch_void(_rcc_gl_ctx, func, __VA_ARGS__) })                        \
  IF_VK(if (_rcc_renderer == fan::window_t::renderer_t::vulkan)                   \
    { _rcc_dispatch_void(_rcc_vk_ctx, func, __VA_ARGS__) })                        \
  fan::throw_error_impl("renderer not supported: " #func);                         \
} while(0)

// get — non-void return, caller specifies return type and union member
#define renderer_get(type, gl_member, vk_member, func, ...)                        \
([&]() -> type {                                                                   \
  IF_GL(if (_rcc_renderer == fan::window_t::renderer_t::opengl)                   \
    { _rcc_dispatch_ret(type, _rcc_gl_ctx, gl_member, func, __VA_ARGS__) })        \
  IF_VK(if (_rcc_renderer == fan::window_t::renderer_t::vulkan)                   \
    { _rcc_dispatch_ret(type, _rcc_vk_ctx, vk_member, func, __VA_ARGS__) })        \
  fan::throw_error_impl("renderer not supported: " #func);                         \
  __unreachable();                                                                 \
}())

#define render_context_call(type, func, ...)                                       \
  render_context_call_ret(type, gl, func, __VA_ARGS__)

#define render_context_call_raw(gl_expr, vk_expr) \
  do { \
    IF_GL(if (fan::graphics::get_window().renderer == fan::window_t::renderer_t::opengl) { gl_expr; }) \
    IF_VK(if (fan::graphics::get_window().renderer == fan::window_t::renderer_t::vulkan) { vk_expr; }) \
  } while(0)

#define renderer_call(func) { \
  auto& w = fan::graphics::get_window(); \
  IF_GL(if (w.renderer == fan::window_t::renderer_t::opengl) { \
    gl->func(); \
  }) \
  IF_VK(if (w.renderer == fan::window_t::renderer_t::vulkan) { \
    vk.func(); \
  }) \
}

// shape push_back for trivial shapes
#define DEFINE_PUSH_BACK(name) \
  shapes::shape_t shapes::name##_t::push_back(const properties_t& p) { \
    return make_shape_ret(g_shapes->add_shape(fan::graphics::shape_type_t::name, p).NRI); \
  }

// json field emit/load macros
#define SHAPE_JSON_EMIT(name) \
  if (shape.get_##name() != defaults.name) out[#name] = shape.get_##name()

#define SHAPE_JSON_EMIT_CUSTOM(key, getter, default_field) \
  if (shape.getter() != defaults.default_field) out[key] = shape.getter()

#define SHAPE_JSON_GET(field) in.get_if(#field, p.field)

#define SHAPE_PROP_SIMPLE(ret, name) \
  ret shapes::shape_t::get_##name() const { \
    return fan::graphics::shapes::get_shape_functions()[get_shape_type()].get_##name(this); \
  } \
  void shapes::shape_t::set_##name(const ret& v) { \
    if (shapes::shape_t::get_##name() == v) return; \
    fan::graphics::shapes::get_shape_functions()[get_shape_type()].set_##name(this, v); \
  }

#define SHAPE_PROP_CULLING(ret, name) \
  ret shapes::shape_t::get_##name() const { \
    return fan::graphics::shapes::get_shape_functions()[get_shape_type()].get_##name(this); \
  } \
  void shapes::shape_t::set_##name(const ret& v) { \
    if (shapes::shape_t::get_##name() == v) return; \
    fan::graphics::shapes::get_shape_functions()[get_shape_type()].set_##name(this, v); \
    update_culling(); \
  }
