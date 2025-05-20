module;

#include <fan/types/types.h>
#if defined(fan_compiler_msvc)
  import <fan/types/json_impl.h>;
#else
  #include <fan/types/json_impl.h>
#endif

export module fan:types.json;

import :types.vector;
import :types.color;

export {
  namespace fan {
    using json = fan::json;
  }

  template <typename T>
  struct fan::adl_serializer<fan::vec2_wrap_t<T>> {
    static void to_json(fan::json& j, const fan::vec2_wrap_t<T>& v) {
      j = fan::json{ v.x, v.y };
    }
    static void from_json(const fan::json& j, fan::vec2_wrap_t<T>& v) {
      v.x = j[0].get<T>();
      v.y = j[1].get<T>();
    }
  };

  template <typename T>
  struct fan::adl_serializer<fan::vec3_wrap_t<T>> {
    static void to_json(fan::json& j, const fan::vec3_wrap_t<T>& v) {
      j = fan::json{ v.x, v.y, v.z };
    }
    static void from_json(const fan::json& j, fan::vec3_wrap_t<T>& v) {
      v.x = j[0].get<T>();
      v.y = j[1].get<T>();
      v.z = j[2].get<T>();
    }
  };

  template <typename T>
  struct fan::adl_serializer<fan::vec4_wrap_t<T>> {
    static void to_json(fan::json& j, const fan::vec4_wrap_t<T>& v) {
      j = fan::json{ v.x, v.y, v.z, v.w };
    }
    static void from_json(const fan::json& j, fan::vec4_wrap_t<T>& v) {
      v.x = j[0].get<T>();
      v.y = j[1].get<T>();
      v.z = j[2].get<T>();
      v.w = j[3].get<T>();
    }
  };

  template <> struct fan::adl_serializer<fan::color> {
    static void to_json(json& j, const fan::color& c) {
      j = json{ c.r, c.g, c.b, c.a };
    }
    static void from_json(const json& j, fan::color& c) {
      c.r = j[0];
      c.g = j[1];
      c.b = j[2];
      c.a = j[3];
    }
  };
}