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

#define RENDER_CALL_CTX(type, ctx, member, err, func, ...)                         \
  if constexpr (requires { static_cast<void>(ctx.func(__VA_ARGS__)); }) {          \
    if constexpr (std::is_void_v<type>) {                                          \
      ctx.func(__VA_ARGS__);                                                       \
      return;                                                                      \
    }                                                                              \
    else {                                                                         \
      obj.member = (std::remove_reference_t<decltype(obj.member)>)                 \
        ctx.func(__VA_ARGS__);                                                     \
      return obj;                                                                  \
    }                                                                              \
  }                                                                                \
  else { fan::throw_error(err); }                                                  \

#define render_context_call(type, func, ...)                                       \
([&]() {                                                                           \
  auto& w = fan::graphics::get_window();                                           \
  if constexpr (!std::is_void_v<type>) type obj;                                   \
  IF_GL(                                                                           \
    if (w.renderer == fan::window_t::renderer_t::opengl) {                         \
      RENDER_CALL_CTX(                                                             \
        type,                                                                      \
        fan::graphics::get_gl_context(),                                           \
        gl,                                                                        \
        std::string("opengl backend TODO: ") +                                     \
          STRINGIFY(fan::graphics::get_gl_context().func(__VA_ARGS__)),            \
        func,                                                                      \
        __VA_ARGS__                                                                \
      )                                                                            \
    }                                                                              \
  )                                                                                \
  IF_VK(                                                                           \
    if (w.renderer == fan::window_t::renderer_t::vulkan) {                         \
      RENDER_CALL_CTX(                                                             \
        type,                                                                      \
        fan::graphics::get_vk_context(),                                           \
        vk,                                                                        \
        std::string("vulkan backend TODO: ") +                                     \
          STRINGIFY(fan::graphics::get_vk_context().func(__VA_ARGS__)),            \
        func,                                                                      \
        __VA_ARGS__                                                                \
      )                                                                            \
    }                                                                              \
  )                                                                                \
  fan::throw_error(std::string("renderer not supported: ") + STRINGIFY(func));     \
  if constexpr (!std::is_void_v<type>) return obj;                                 \
}())



#define render_context_call_raw(gl_expr, vk_expr) \
  do { \
    IF_GL(if (fan::graphics::get_window().renderer == fan::window_t::renderer_t::opengl) { gl_expr; }) \
    IF_VK(if (fan::graphics::get_window().renderer == fan::window_t::renderer_t::vulkan) { vk_expr; }) \
  } while(0)

#define renderer_call(func) { \
  auto& w = fan::graphics::get_window(); \
  IF_GL(if (w.renderer == fan::window_t::renderer_t::opengl) { \
    gl.func(); \
  }) \
  IF_VK(if (w.renderer == fan::window_t::renderer_t::opengl) { \
    vk.func(); \
  }) \
}

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