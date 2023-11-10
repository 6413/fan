#pragma once

#include <iostream>
#include <algorithm>
#include <numeric>
#include <string>
#include <compare>

#include _FAN_PATH(math/math.h)

#define fan_coordinate_letters0
#define fan_coordinate_letters1 x
#define fan_coordinate_letters2 x, y
#define fan_coordinate_letters3 x, y, z
#define fan_coordinate_letters4 x, y, z, w

#define fan_coordinate(x) CONCAT(fan_coordinate_letters, x)

namespace fan {
	using access_type_t = int;

  template <typename T, int n_>
  struct vec;

  template <typename T>
  struct vec<T, 0> {

    auto operator<=>(const vec<T, 0>&) const = default;

    static constexpr access_type_t size() {
      return 0;
    }
  };
}

#define vec_n 1
#include "vector_impl.h"

#define vec_n 2
#include "vector_impl.h"

#define vec_n 3
#include "vector_impl.h"

#define vec_n 4
#include "vector_impl.h"

namespace fan {

  template <typename value_type_t>
  struct vec3_wrap_t;

	// wrappers for type specific functions
	template <typename value_type_t>
  struct vec2_wrap_t : vec<value_type_t, 2> {
		using inherit_t = vec<value_type_t, 2>;
		using inherit_t::vec;

    vec2_wrap_t() = default;
		constexpr vec2_wrap_t(const vec2_wrap_t& test0) 
      : inherit_t((vec<value_type_t, 2>)test0) { } 
    template <typename T> constexpr vec2_wrap_t(const vec<T, inherit_t::size()>& test0)
      : inherit_t(test0) { } 
    template <typename T> constexpr vec2_wrap_t(const vec<T, 3>& test0) 
    : inherit_t(test0.x, test0.y) { } 

    constexpr auto copysign(const auto& test0) const { return vec2_wrap_t(fan::math::copysign(inherit_t::x, test0.x), fan::math::copysign(inherit_t::y, test0.y)); }
    vec2_wrap_t reflect(const auto& normal) {
      auto k = fan::math::cross(vec<value_type_t, 3>{ normal.x, normal.y, 0 }, vec<value_type_t, 3>{ 0, 0, -1 });
      f32_t multiplier = k.dot(vec<value_type_t, 3>{ inherit_t::x, inherit_t::y, 0 });
      return vec2_wrap_t( k.x * multiplier, k.y * multiplier);
    }
    #if defined(loco_imgui)
    operator ImVec2() const { return ImVec2(inherit_t::x, inherit_t::y); }
    constexpr vec2_wrap_t(const ImVec2& v) { inherit_t::x = v.x; inherit_t::y = v.y;}
    #endif
  };

	template <typename value_type_t>
  struct vec3_wrap_t : vec<value_type_t, 3> {
		using inherit_t = vec<value_type_t, 3>;
		using inherit_t::vec;
		
    vec3_wrap_t() = default;
		// constexpr vec3_wrap_t(const vec3_wrap_t& test0) 
    //   : inherit_t((vec<value_type_t, 3>)test0) { } 
    template <typename T> constexpr vec3_wrap_t(const vec<T, inherit_t::size()>& test0)
      : inherit_t(test0) { } 
    template <typename T> constexpr vec3_wrap_t(const fan::vec<T, 2>& test0) 
      : inherit_t(test0.x, test0.y, inherit_t::z) { }
    template <typename T> constexpr vec3_wrap_t(const fan::vec<T, 2>& test0, auto value) 
      : inherit_t(test0.x, test0.y, value) { } 

		template <typename T>
		constexpr auto cross(const fan::vec3_wrap_t<T>& vector) const {
			return fan::math::cross<vec3_wrap_t<T>>(*this, vector);
		}
  };

	template <typename value_type_t>
  struct vec4_wrap_t : vec<value_type_t, 4> {
		using inherit_t = vec<value_type_t, 4>;
		using inherit_t::vec;

    vec4_wrap_t() = default;
		constexpr vec4_wrap_t(const vec4_wrap_t& test0) 
      : inherit_t((vec<value_type_t, 4>)test0) { } 
    template <typename T, access_type_t n>
    constexpr vec4_wrap_t(const vec<T, n>& test0) 
      : inherit_t(test0) { } 
  };

	using vec1 = vec<f32_t, 1>;

  using vec2b = vec2_wrap_t<bool>;
	using vec3b = vec3_wrap_t<bool>;
	using vec4b = vec4_wrap_t<bool>;

	using vec2i = vec2_wrap_t<int>;
	using vec3i = vec3_wrap_t<int>;
	using vec4i = vec4_wrap_t<int>;

	using vec2si = vec2i;
	using vec3si = vec3i;
	using vec4si = vec4i;

	using vec2ui = vec2_wrap_t<uint32_t>;
	using vec3ui = vec3_wrap_t<uint32_t>;
	using vec4ui = vec4_wrap_t<uint32_t>;

	using vec2f = vec2_wrap_t<f32_t>;
	using vec3f = vec3_wrap_t<f32_t>;
	using vec4f = vec4_wrap_t<f32_t>;

	using vec2d = vec2_wrap_t<f64_t>;
	using vec3d = vec3_wrap_t<f64_t>;
	using vec4d = vec4_wrap_t<f64_t>;

	using vec2 = vec2_wrap_t<cf_t>;
	using vec3 = vec3_wrap_t<cf_t>;
	using vec4 = vec4_wrap_t<cf_t>;

	template <typename casted_t, access_type_t n, typename old_t>
	constexpr fan::vec<casted_t, n> cast(const fan::vec<old_t, n>& v)
	{
		return fan::vec<casted_t, n>(v);
	}
}

namespace fmt {
  template<typename T>
  struct fmt::formatter<fan::vec2_wrap_t<T>> {
    auto parse(fmt::format_parse_context& ctx) {
      return ctx.end();
    }
    auto format(const fan::vec2_wrap_t<T>& obj, fmt::format_context& ctx) {

      return fmt::format_to(ctx.out(), "{}", obj.to_string());
    }
  };
  template<typename T>
  struct fmt::formatter<fan::vec3_wrap_t<T>> {
    auto parse(fmt::format_parse_context& ctx) {
      return ctx.end();
    }
    auto format(const fan::vec3_wrap_t<T>& obj, fmt::format_context& ctx) {

      return fmt::format_to(ctx.out(), "{}", obj.to_string());
    }
  };
  template<typename T>
  struct fmt::formatter<fan::vec4_wrap_t<T>> {
    auto parse(fmt::format_parse_context& ctx) {
      return ctx.end();
    }
    auto format(const fan::vec4_wrap_t<T>& obj, fmt::format_context& ctx) {

      return fmt::format_to(ctx.out(), "{}", obj.to_string());
    }
  };
	namespace math {
    fan::vec2 reflect(const auto& direction, const auto& normal) {
      auto k = fan::math::cross<fan::vec3>(fan::vec3(normal.x, normal.y, 0), fan::vec3(0, 0, -1));
      f32_t multiplier = k.dot(fan::vec3{ direction.x, direction.y, 0 });
      return fan::vec2(k.x * multiplier, k.y * multiplier);
    }
  }
}

#undef fan_coordinate_letters0
#undef fan_coordinate_letters1
#undef fan_coordinate_letters2
#undef fan_coordinate_letters3
#undef fan_coordinate_letters4
#undef fan_coordinate