#pragma once

#include <iostream>
#include <algorithm>
#include <numeric>
#include <string>
#include <compare>
#include <tuple>

#include _FAN_PATH(math/math.h)

#define fan_coordinate_letters0
#define fan_coordinate_letters1 x
#define fan_coordinate_letters2 x, y
#define fan_coordinate_letters3 x, y, z
#define fan_coordinate_letters4 x, y, z, w

#define fan_coordinate(x) CONCAT(fan_coordinate_letters, x)

namespace fan {

  using access_type_t = uint16_t;

  template <typename value_type_t>
  struct vec0_wrap_t {
    #define vec_t vec0_wrap_t
    #define vec_n 0
    #include "vector_impl.h"
  };

  template <typename value_type_t>
  struct vec1_wrap_t {
    #define vec_t vec1_wrap_t
    #define vec_n 1
    #include "vector_impl.h"
  };

  template <typename value_type_t>
  struct vec3_wrap_t;

	// wrappers for type specific functions
	template <typename value_type_t>
  struct vec2_wrap_t {
    #define vec_t vec2_wrap_t
    #define vec_n 2
    #include "vector_impl.h"

    template <typename T> constexpr vec2_wrap_t(const vec3_wrap_t<T>& test0) 
    : vec2_wrap_t(test0.x, test0.y) { } 
    constexpr auto copysign(const auto& test0) const { return vec2_wrap_t(fan::math::copysign(x, test0.x), fan::math::copysign(y, test0.y)); }
    template <typename T>
    vec2_wrap_t reflect(const T& normal) {
      auto k = fan::math::cross(vec3_wrap_t<typename T::value_type>{ normal.x, normal.y, 0 }, vec3_wrap_t<typename T::value_type>{ 0, 0, -1 });
      f32_t multiplier = k.dot(vec3_wrap_t<typename T::value_type>{ x, y, 0 });
      return vec2_wrap_t( k.x * multiplier, k.y * multiplier);
    }
    #if defined(loco_imgui)
    constexpr operator ImVec2() const { return ImVec2(x, y); }
    constexpr vec2_wrap_t(const ImVec2& v) { x = v.x; y = v.y;}
    #endif
    // coordinate system angle. TODO need rename to something meaningful
		constexpr auto csangle() const { return atan2(x, -y);}

    template <typename T>
    vec2_wrap_t<T> lerp(const vec2_wrap_t<T>& dst, T t) {
      return { x + t * (dst.x - x), y + t * (dst.y - y) };
    }
  };

  template <typename value_type_t>
  struct vec3_wrap_t {
    #define vec_t vec3_wrap_t
    #define vec_n 3
    #include "vector_impl.h"

    template <typename T>
		constexpr vec3_wrap_t(const vec2_wrap_t<T>& test0) 
      : vec3_wrap_t(test0.x, test0.y, 0) { } 
    template <typename T>
		constexpr vec3_wrap_t(const vec2_wrap_t<T>& test0, auto value) 
      : vec3_wrap_t(test0.x, test0.y, value) { } 


  #if defined(loco_assimp)
  vec3_wrap_t(const aiVector3D& v) {
    x = v.x;
    y = v.y;
    z = v.z;
  }
 /*  operator aiVector3D() {
    aiVector3D v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
  }*/
  #endif

   template <typename T>
    vec3_wrap_t& operator=(const vec2_wrap_t<T>& test0) {
      x = test0.x;
      y = test0.y;
      return *this;
    }
		template <typename T>
		constexpr auto cross(const fan::vec3_wrap_t<T>& vector) const {
			return fan::math::cross<vec3_wrap_t<T>>(*this, vector);
		}
    template <typename T>
    vec3_wrap_t<T> reflect(const vec3_wrap_t<T>& normal) {
      double dot_product = dot(normal);
      vec3_wrap_t<T> scaled_normal = normal * (2 * dot_product);
      vec3_wrap_t<T> reflection = *this - scaled_normal;
      return reflection;
    }
    template <typename T>
    vec3_wrap_t<T> lerp(const vec3_wrap_t<T>& dst, T t) {
      return { x + t * (dst.x - x), y + t * (dst.y - y), z + t * (dst.z - z) };
    }
  };

  struct color;

  template <typename value_type_t>
  struct vec4_wrap_t {
    #define vec_t vec4_wrap_t
    #define vec_n 4
    #include "vector_impl.h"
    //incomplete type
    //constexpr vec4_wrap_t(const fan::color& test0)
    //  : vec4_wrap_t() { 
    //  *(fan::color*)this = test0;
    //}
    template <typename T>
    constexpr vec4_wrap_t(const vec2_wrap_t<T>& test0, const vec2_wrap_t<T>& test1)
      : vec4_wrap_t(test0.x, test0.y, test1.x, test1.y) { }
    template <typename T> constexpr vec4_wrap_t(const vec3_wrap_t<T>& test0)
      : vec4_wrap_t(test0.x, test0.y, test0.z, 0) { }
    template <typename T> 
    constexpr vec4_wrap_t(const vec3_wrap_t<T>& test0, auto value)
      : vec4_wrap_t(test0.x, test0.y, test0.z, value) { }
    #if defined(loco_imgui)
    constexpr operator ImVec4() const { return ImVec4(x, y, z, w); }
    constexpr vec4_wrap_t(const ImVec4& v) { x = v.x; y = v.y; z = v.z; w = v.w; }
    #endif
  };

	 using vec1b = vec1_wrap_t<bool>;
   using vec2b = vec2_wrap_t<bool>;
	 using vec3b = vec3_wrap_t<bool>;
	 using vec4b = vec4_wrap_t<bool>;

   using vec1i = vec1_wrap_t<int>;
	 using vec2i = vec2_wrap_t<int>;
	 using vec3i = vec3_wrap_t<int>;
	 using vec4i = vec4_wrap_t<int>;

   using vec1si = vec1i;
	 using vec2si = vec2i;
	 using vec3si = vec3i;
	 using vec4si = vec4i;

   using vec1ui = vec1_wrap_t<uint32_t>;
	 using vec2ui = vec2_wrap_t<uint32_t>;
	 using vec3ui = vec3_wrap_t<uint32_t>;
	 using vec4ui = vec4_wrap_t<uint32_t>;

   using vec1f = vec1_wrap_t<f32_t>;
	 using vec2f = vec2_wrap_t<f32_t>;
	 using vec3f = vec3_wrap_t<f32_t>;
	 using vec4f = vec4_wrap_t<f32_t>;

   using vec1d = vec1_wrap_t<f64_t>;
	 using vec2d = vec2_wrap_t<f64_t>;
	 using vec3d = vec3_wrap_t<f64_t>;
	 using vec4d = vec4_wrap_t<f64_t>;

   using vec1 = vec1_wrap_t<cf_t>;
	 using vec2 = vec2_wrap_t<cf_t>;
	 using vec3 = vec3_wrap_t<cf_t>;
	 using vec4 = vec4_wrap_t<cf_t>;

	template <typename casted_t, template<typename> typename vec_t, typename old_t>
	constexpr vec_t<casted_t> cast(const vec_t<old_t>& v) { return vec_t<casted_t>(v); }

  template <int n, typename T>
  using vec_wrap_t = std::tuple_element_t<n, std::tuple<fan::vec0_wrap_t<T>, fan::vec1_wrap_t<T>, fan::vec2_wrap_t<T>, fan::vec3_wrap_t<T>, fan::vec4_wrap_t<T>>>;
}

namespace fmt {
   template<typename T>
   struct fmt::formatter<fan::vec2_wrap_t<T>> {
     auto parse(fmt::format_parse_context& ctx) { return ctx.end(); }
     auto format(const fan::vec2_wrap_t<T>& obj, fmt::format_context& ctx) { return fmt::format_to(ctx.out(), "{}", obj.to_string()); }
   };
   template<typename T>
   struct fmt::formatter<fan::vec3_wrap_t<T>> {
     auto parse(fmt::format_parse_context& ctx) { return ctx.end(); }
     auto format(const fan::vec3_wrap_t<T>& obj, fmt::format_context& ctx) { return fmt::format_to(ctx.out(), "{}", obj.to_string());}
   };
   template<typename T>
   struct fmt::formatter<fan::vec4_wrap_t<T>> {
     auto parse(fmt::format_parse_context& ctx) { return ctx.end(); }
     auto format(const fan::vec4_wrap_t<T>& obj, fmt::format_context& ctx) {return fmt::format_to(ctx.out(), "{}", obj.to_string());}
   };
}

#undef fan_coordinate_letters0
#undef fan_coordinate_letters1
#undef fan_coordinate_letters2
#undef fan_coordinate_letters3
#undef fan_coordinate_letters4
#undef fan_coordinate