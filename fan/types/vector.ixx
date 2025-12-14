module;

#include <fan/utility.h>

#if defined(fan_vulkan)
  #include <vulkan/vulkan.h>
#endif

#if defined(fan_gui)
  #include <fan/imgui/imgui.h>
  #include <fan/imgui/implot.h>
#endif

#if defined(fan_3D)
  #include <assimp/vector3.h>
#endif

#if defined(fan_physics)
  #include <box2d/math_functions.h>
#endif

#include <string>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <cmath>

export module fan.types.vector;

export import fan.math;

//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------

#define fan_compile_component_functions

#define fan_coordinate_letters0
#define fan_coordinate_letters1 x
#define fan_coordinate_letters2 x, y
#define fan_coordinate_letters3 x, y, z
#define fan_coordinate_letters4 x, y, z, w

#define fan_coordinate(x) CONCAT(fan_coordinate_letters, x)

#if defined(fan_compile_component_functions)

#define vec_letter_0 x
#define vec_letter_1 y
#define vec_letter_2 z
#define vec_letter_3 w

// 2) Variadic gating: emit __VA_ARGS__ iff I < N
#define vec_if_has(N,I,...) vec_if_has_##I(N, __VA_ARGS__)
#define vec_if_has_0(N, ...) __VA_ARGS__

// I == 1 → available for N >= 2
#define vec_if_has_1(N, ...) vec_if_has_1_##N(__VA_ARGS__)
#define vec_if_has_1_2(...) __VA_ARGS__
#define vec_if_has_1_3(...) __VA_ARGS__
#define vec_if_has_1_4(...) __VA_ARGS__

// I == 2 → available for N >= 3
#define vec_if_has_2(N, ...) vec_if_has_2_##N(__VA_ARGS__)
#define vec_if_has_2_2(...)
#define vec_if_has_2_3(...) __VA_ARGS__
#define vec_if_has_2_4(...) __VA_ARGS__

// I == 3 → available for N >= 4
#define vec_if_has_3(N, ...) vec_if_has_3_##N(__VA_ARGS__)
#define vec_if_has_3_2(...)
#define vec_if_has_3_3(...)
#define vec_if_has_3_4(...) __VA_ARGS__

#define vec_value(I) EXPAND(CONCAT(vec_comp_, I))

//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------

#define gen_vec2_swizzle(N,A,B) \
  vec_if_has(N,A, \
    vec_if_has(N,B, \
      constexpr vec2_wrap_t<value_type_t> \
      CONCAT2(vec_letter_##A, vec_letter_##B)() const { \
        return {vec_letter_##A, vec_letter_##B}; \
      } \
      constexpr vec_ref2<vec2_wrap_t<value_type_t>, value_type_t> \
      CONCAT2(vec_letter_##A, vec_letter_##B)() { \
        return {vec_letter_##A, vec_letter_##B}; \
      } \
    ) \
  )

#define gen_vec3_swizzle(N,A,B,C) \
  vec_if_has(N,A, \
    vec_if_has(N,B, \
      vec_if_has(N,C, \
        constexpr vec3_wrap_t<value_type_t> \
        CONCAT3(vec_letter_##A, vec_letter_##B, vec_letter_##C)() const { \
          return {vec_letter_##A, vec_letter_##B, vec_letter_##C}; \
        } \
        constexpr vec_ref3<vec3_wrap_t<value_type_t>, value_type_t> \
        CONCAT3(vec_letter_##A, vec_letter_##B, vec_letter_##C)() { \
          return {vec_letter_##A, vec_letter_##B, vec_letter_##C}; \
        } \
      ) \
    ) \
  )

#define gen_vec4_swizzle(N,A,B,C,D) \
  vec_if_has(N,A, \
    vec_if_has(N,B, \
      vec_if_has(N,C, \
        vec_if_has(N,D, \
          constexpr vec4_wrap_t<value_type_t> \
          CONCAT4(vec_letter_##A, vec_letter_##B, vec_letter_##C, vec_letter_##D)() const { \
            return {vec_letter_##A, vec_letter_##B, vec_letter_##C, vec_letter_##D}; \
          } \
          constexpr vec_ref4<vec4_wrap_t<value_type_t>, value_type_t> \
          CONCAT4(vec_letter_##A, vec_letter_##B, vec_letter_##C, vec_letter_##D)() { \
            return {vec_letter_##A, vec_letter_##B, vec_letter_##C, vec_letter_##D}; \
          } \
        ) \
      ) \
    ) \
  )

#define generate_vec_component_functions(N) \
  gen_vec2_swizzle(N,0,0) gen_vec2_swizzle(N,0,1) gen_vec2_swizzle(N,0,2) gen_vec2_swizzle(N,0,3) \
  gen_vec2_swizzle(N,1,0) gen_vec2_swizzle(N,1,1) gen_vec2_swizzle(N,1,2) gen_vec2_swizzle(N,1,3) \
  gen_vec2_swizzle(N,2,0) gen_vec2_swizzle(N,2,1) gen_vec2_swizzle(N,2,2) gen_vec2_swizzle(N,2,3) \
  gen_vec2_swizzle(N,3,0) gen_vec2_swizzle(N,3,1) gen_vec2_swizzle(N,3,2) gen_vec2_swizzle(N,3,3)

#define fan_gen_vec_ref_arith(N, op) \
  constexpr vec_t operator op (const vec_ref##N& v) { \
    return vec_t(*this) op vec_t(v); \
  } \
  constexpr vec_t operator op (value_type_t v) { \
    return vec_t(*this) op v; \
  }

//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------

export namespace fan {
  template <typename vec_t, typename value_type_t>
  struct vec_ref2 {
    value_type_t& a;
    value_type_t& b;
    constexpr operator vec_t() const { return {a, b}; }
    constexpr vec_ref2& operator=(const vec_t& v) { a = v.x; b = v.y; return *this; }
    fan_gen_vec_ref_arith(2, +);
    fan_gen_vec_ref_arith(2, -);
    fan_gen_vec_ref_arith(2, *);
    fan_gen_vec_ref_arith(2, /);
  };

  template <typename vec_t, typename value_type_t>
  struct vec_ref3 {
    value_type_t& a;
    value_type_t& b;
    value_type_t& c;
    constexpr operator vec_t() const { return {a, b, c}; }
    constexpr vec_ref3& operator=(const vec_t& v) { a = v.x; b = v.y; c = v.z; return *this; }

    fan_gen_vec_ref_arith(3, +);
    fan_gen_vec_ref_arith(3, -);
    fan_gen_vec_ref_arith(3, *);
    fan_gen_vec_ref_arith(3, /);
  };

  template <typename vec_t, typename value_type_t>
  struct vec_ref4 {
    value_type_t& a;
    value_type_t& b;
    value_type_t& c;
    value_type_t& d;
    constexpr operator vec_t() const { return {a, b, c, d}; }
    constexpr vec_ref4& operator=(const vec_t& v) { a = v.x; b = v.y; c = v.z; d = v.w; return *this; }

    fan_gen_vec_ref_arith(4, +);
    fan_gen_vec_ref_arith(4, -);
    fan_gen_vec_ref_arith(4, *);
    fan_gen_vec_ref_arith(4, /);
  };
}

#endif

//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------

export namespace fan {

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

  template <typename value_type_t>
  struct vec4_wrap_t;


	// wrappers for type specific functions
  template <typename value_type_t>
  struct vec2_wrap_t {
  #define vec_t vec2_wrap_t
  #define vec_n 2
  #include "vector_impl.h"

    template <typename T> constexpr vec2_wrap_t(const vec3_wrap_t<T>& test0)
      : vec2_wrap_t(test0.x, test0.y) { }

    constexpr auto copysign(const auto& test0) const { return vec2_wrap_t(fan::math::copysign(x, test0.x), fan::math::copysign(y, test0.y)); }

  #if defined(fan_gui)
    constexpr operator ImPlotPoint() const { return ImPlotPoint((f32_t)x, (f32_t)y); }
    constexpr vec2_wrap_t(const ImPlotPoint& v) { x = v.x; y = v.y; }
    constexpr operator ImVec2() const { return ImVec2{(f32_t)x, (f32_t)y}; }
    constexpr vec2_wrap_t(const ImVec2& v) { x = v.x; y = v.y; }
  #endif

  #if defined(fan_physics)
    constexpr operator b2Vec2() const { return b2Vec2{(f32_t)x, (f32_t)y}; }
    constexpr vec2_wrap_t(const b2Vec2& v) { x = v.x; y = v.y; }
  #endif

  #if defined(fan_vulkan)
    constexpr operator VkExtent2D() const { return VkExtent2D{(uint32_t)x, (uint32_t)y}; }
    constexpr vec2_wrap_t(const VkExtent2D& v) { x = v.width; y = v.height; }
  #endif

    // coordinate system angle. TODO need rename to something meaningful
    constexpr auto csangle() const { return std::atan2(x, -y); }
    constexpr auto angle() const { return std::atan2(y, x); }

    template <typename T>
    bool is_collinear(const vec2_wrap_t<T>& a) { return a.x == x || a.y == y; }

    template <typename T>
    vec2_wrap_t<T> get_corner(const vec2_wrap_t<T>& a) { return {a.x, y}; }

    template <typename T>
    vec2_wrap_t<T> lerp(const vec2_wrap_t<T>& dst, T t) {
      return { x + t * (dst.x - x), y + t * (dst.y - y) };
    }

    constexpr vec2_wrap_t<value_type_t> perpendicular() const {
      return { -(*this)[1], (*this)[0] };
    }

    constexpr value_type_t cross(const vec2_wrap_t<value_type_t>& b) {
      return x * b.y - y * b.x;
    }

  #if defined(fan_compile_component_functions)
    generate_vec_component_functions(2);
  #endif
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

    template <typename T>
    constexpr vec3_wrap_t(const vec4_wrap_t<T>& test0)
      : vec3_wrap_t(test0.x, test0.y, test0.z) { }

  #if defined(fan_3D)
    vec3_wrap_t(const aiVector3D& v) { x = v.x; y = v.y; z = v.z; }
    operator aiVector3D() { return {x, y, z}; }
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
    vec3_wrap_t<T> lerp(const vec3_wrap_t<T>& dst, T t) {
      return { x + t * (dst.x - x), y + t * (dst.y - y), z + t * (dst.z - z) };
    }

  #if defined(fan_compile_component_functions)
    generate_vec_component_functions(3);

    constexpr vec4_wrap_t<value_type_t> xyxy() const { return {x, y, x, y}; };
    constexpr vec4_wrap_t<value_type_t> xxyy() const { return {x, x, y, y}; };
    constexpr vec4_wrap_t<value_type_t> yxyx() const { return {y, x, y, x}; };
  #endif
  };

  template <typename value_type_t>
  struct vec4_wrap_t {
  #define vec_t vec4_wrap_t
  #define vec_n 4
  #include "vector_impl.h"

    template <typename T>
    constexpr vec4_wrap_t(const vec2_wrap_t<T>& test0, auto third, auto fourth)
      : vec4_wrap_t(test0.x, test0.y, third, fourth) { }

    template <typename T>
    constexpr vec4_wrap_t(const vec2_wrap_t<T>& test0, const vec2_wrap_t<T>& test1)
      : vec4_wrap_t(test0.x, test0.y, test1.x, test1.y) { }

    template <typename T>
    constexpr vec4_wrap_t(const vec3_wrap_t<T>& test0)
      : vec4_wrap_t(test0.x, test0.y, test0.z, 0) { }

    template <typename T>
    constexpr vec4_wrap_t(const vec3_wrap_t<T>& test0, auto value)
      : vec4_wrap_t(test0.x, test0.y, test0.z, value) { }

  #if defined(fan_gui)
    constexpr operator ImVec4() const { return ImVec4(x, y, z, w); }
    constexpr vec4_wrap_t(const ImVec4& v) { x = v.x; y = v.y; z = v.z; w = v.w; }
  #endif

    template <typename T>
    constexpr operator vec2_wrap_t<T>() const { return {x, y}; }

  #if defined(fan_compile_component_functions)
    generate_vec_component_functions(4);

    constexpr vec4_wrap_t xyxy() const { return {x, y, x, y}; }
    constexpr vec4_wrap_t ywyw() const { return {y, w, y, w}; }
  #endif
  };

#define fan_vector_types(X) \
  X(b,   bool) \
  X(i,   int) \
  X(ui,  uint32_t) \
  X(ull, uint64_t) \
  X(f,   f32_t) \
  X(d,   f64_t)

#define fan_define_vec_aliases(T) \
  using vec1##T = vec1_wrap_t<T>; \
  using vec2##T = vec2_wrap_t<T>; \
  using vec3##T = vec3_wrap_t<T>; \
  using vec4##T = vec4_wrap_t<T>;

#define fan_gen_vec_aliases(suffix, type) \
  using vec1##suffix = vec1_wrap_t<type>; \
  using vec2##suffix = vec2_wrap_t<type>; \
  using vec3##suffix = vec3_wrap_t<type>; \
  using vec4##suffix = vec4_wrap_t<type>;

  fan_vector_types(fan_gen_vec_aliases);

  using vec1 = vec1f;
  using vec2 = vec2f;
  using vec3 = vec3f;
  using vec4 = vec4f;

  using vec2si = vec2_wrap_t<signed int>;

	template <typename casted_t, template<typename> typename vec_t, typename old_t>
	constexpr vec_t<casted_t> cast(const vec_t<old_t>& v) { return vec_t<casted_t>(v); }

  #define fan_vector_array
  #undef fan_coordinate
  #define fan_coordinate(x) arr[x]
  template <int vector_n, typename value_type_t>
  struct vec_wrap_t {
    #define vec_t vec_wrap_t
    #define vec_n vector_n
    #include "vector_impl.h"
    
    template <typename T>
    requires(vector_n >= 2)
		constexpr vec_wrap_t(const vec2_wrap_t<T>& test0) 
      : vec_wrap_t(test0.x, test0.y) { } 

    template <typename T>
    requires(vector_n >= 2)
		constexpr vec_wrap_t(const vec3_wrap_t<T>& test0) 
      : vec_wrap_t(test0.x, test0.y) { } 

    template <typename T>
    requires(vector_n >= 3)
		constexpr vec_wrap_t(const vec3_wrap_t<T>& test0) 
      : vec_wrap_t(test0.x, test0.y, test0.z) { } 

    template <typename T>
    requires(vector_n >= 2)
    operator vec2_wrap_t<T>() const {
      return vec2_wrap_t<T>(operator[](0), operator[](1));
    }
    template <typename T>
    requires(vector_n >= 3)
    operator vec3_wrap_t<T>() const {
      return vec3_wrap_t<T>(operator[](0), operator[](1), operator[](2));
    }
  };

  struct ray3_t {
    fan::vec3 origin;
    // normalized
    fan::vec3 direction;

    constexpr ray3_t() = default;
    constexpr ray3_t(const fan::vec3& origin_, fan::vec3& direction_) : origin(origin_), direction(direction_){}
  };

#undef fan_coordinate_letters0
#undef fan_coordinate_letters1
#undef fan_coordinate_letters2
#undef fan_coordinate_letters3
#undef fan_coordinate_letters4
#undef fan_coordinate

}

export namespace fan {
  namespace math {
    template <typename T>
    constexpr fan::vec2_wrap_t<T> angle_to_vector(const T& angle_radians) {
      return fan::vec2_wrap_t<T>(std::cos(angle_radians), std::sin(angle_radians));
    }
  }
  template <typename>
  inline constexpr bool is_vector_type_v = false;
  template <template <typename> typename V, typename T>
  inline constexpr bool is_vector_type_v<V<T>> =
    std::is_same_v<V<T>, vec0_wrap_t<T>> ||
    std::is_same_v<V<T>, vec1_wrap_t<T>> ||
    std::is_same_v<V<T>, vec2_wrap_t<T>> ||
    std::is_same_v<V<T>, vec3_wrap_t<T>> ||
    std::is_same_v<V<T>, vec4_wrap_t<T>>;
  template <int N, typename T>
  inline constexpr bool is_vector_type_v<vec_wrap_t<N, T>> = true;
  template <typename T>
  concept is_vector = is_vector_type_v<std::remove_cvref_t<T>>;
}

#define fan_hash_vec1(type) \
template <> struct std::hash<fan::vec1_wrap_t<type>> { \
  size_t operator()(const fan::vec1_wrap_t<type>& v) const noexcept { \
    return std::hash<type>{}(v.x); \
  } \
};

#define fan_hash_vec2(type) \
template <> struct std::hash<fan::vec2_wrap_t<type>> { \
  size_t operator()(const fan::vec2_wrap_t<type>& v) const noexcept { \
    size_t s = 0; \
    s = hash_combine(s, std::hash<type>{}(v.x)); \
    s = hash_combine(s, std::hash<type>{}(v.y)); \
    return s; \
  } \
};

#define fan_hash_vec3(type) \
template <> struct std::hash<fan::vec3_wrap_t<type>> { \
  size_t operator()(const fan::vec3_wrap_t<type>& v) const noexcept { \
    size_t s = 0; \
    s = hash_combine(s, std::hash<type>{}(v.x)); \
    s = hash_combine(s, std::hash<type>{}(v.y)); \
    s = hash_combine(s, std::hash<type>{}(v.z)); \
    return s; \
  } \
};

#define fan_hash_vec4(type) \
template <> struct std::hash<fan::vec4_wrap_t<type>> { \
  size_t operator()(const fan::vec4_wrap_t<type>& v) const noexcept { \
    size_t s = 0; \
    s = hash_combine(s, std::hash<type>{}(v.x)); \
    s = hash_combine(s, std::hash<type>{}(v.y)); \
    s = hash_combine(s, std::hash<type>{}(v.z)); \
    s = hash_combine(s, std::hash<type>{}(v.w)); \
    return s; \
  } \
};


namespace std {
  constexpr size_t hash_combine(size_t seed, size_t h) {
    return seed ^ (h + 0x9e3779b9 + (seed << 6) + (seed >> 2));
  }

#define fan_gen_all_hashes(suffix, type) \
  fan_hash_vec1(type) \
  fan_hash_vec2(type) \
  fan_hash_vec3(type) \
  fan_hash_vec4(type)

  fan_vector_types(fan_gen_all_hashes)
}