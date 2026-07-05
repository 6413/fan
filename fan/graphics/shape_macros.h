#pragma once

#if defined(FAN_GUI)
  #define IF_GUI(...) __VA_ARGS__
#else
  #define IF_GUI(...)
#endif

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
