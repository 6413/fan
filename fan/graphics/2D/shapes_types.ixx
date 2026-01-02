module;
#if defined(FAN_2D)

#include <fan/utility.h>
#include <cstring>
#include <type_traits>
#include <algorithm>
#include <sstream>

#include <fan/graphics/opengl/init.h>

#endif

export module fan.graphics.shapes.types;

#if defined(FAN_2D)
  import fan.window;
  import fan.graphics.common_context;
  import fan.graphics.opengl.core;

  #if defined(FAN_VULKAN)
  import fan.graphics.vulkan.core;
  #endif

#else
  import fan.types.vector;
  import fan.types.color;
#endif

#if defined(FAN_2D)

export namespace fan::graphics {

  #define TO_ENUM(x) x,
  #define TO_STRING(x) std::string(#x),

  // shapes defined here
  #include "shapes.h"

  #define GEN_SHAPES_SKIP_ENUM(x) x
  #define GEN_SHAPES_SKIP_STRING(x) x

    struct shape_type_t {
      enum {
        invalid = -1,
        GEN_SHAPES(TO_ENUM, GEN_SHAPES_SKIP_ENUM)
        last
      };
    };
    std::string shape_names[] = {
      GEN_SHAPES(TO_STRING, GEN_SHAPES_SKIP_STRING)
    };

  #undef TO_ENUM
  #undef TO_STRING
  #undef GEN_SHAPES_SKIP_ENUM
  #undef GEN_SHAPES_SKIP_STRING

  struct shape_gl_init_t {
    std::pair<int, const char*> index;
    uint32_t size;
    uint32_t type; // for example GL_FLOAT
    uint32_t stride;
    uint32_t offset;
  };
  struct shape_gl_init_list_t {
    fan::graphics::shape_gl_init_t* ptr = nullptr;
    int count = 0;
  };

  std::uint8_t* A_resize(void* ptr, std::uintptr_t size);
}

export namespace fan::graphics::shaper {
#define shaper_set_fan 1
#define shaper_set_MaxMaxElementPerBlock 0x10000
  inline constexpr uint32_t MaxElementPerBlock = shaper_set_MaxMaxElementPerBlock;

  // sizeof(image_t) == 2
  static_assert(sizeof(fan::graphics::image_t) == 2, "update shaper_set_MaxKeySize");
#define shaper_set_MaxKeySize 2 * 30

#ifndef bcontainer_set_alloc_open
#define bcontainer_set_alloc_open(n) std::malloc(n)
#endif
#ifndef bcontainer_set_alloc_resize
#define bcontainer_set_alloc_resize(ptr, n) std::realloc(ptr, n)
#endif
#ifndef bcontainer_set_alloc_close
#define bcontainer_set_alloc_close(ptr) std::free(ptr)
#endif
#include <fan/graphics/2D/shaper.h>
  // will die if renderer has different sizes of structs
#define shaper_set_ShapeTypeChange \
			__builtin_memcpy(new_renderdata, old_renderdata, element_count * g_shapes->shaper.GetRenderDataSize(sti)); \
			__builtin_memcpy(new_data, old_data, element_count * g_shapes->shaper.GetDataSize(sti));
}

namespace fan {
  template <bool cond>
  struct type_or_uint8_t {
    template <typename T>
    using d = std::conditional_t<cond, T, uint8_t>;
  };
}

#endif

export namespace fan::graphics {
#if defined(FAN_2D)

  using fan::graphics::shaper::MaxElementPerBlock;

  using shaper_t = shaper::shaper_t;

#pragma pack(push, 1)

  using blending_t = uint8_t;
  using depth_t = uint16_t;
  using visible_t = uint8_t;

#define st(name, viewport_inside) \
  struct CONCAT(_, name); \
	template <bool cond> \
	struct CONCAT(name, _cond) { \
		template <typename T> \
		using d = typename fan::type_or_uint8_t<cond>::template d<T>; \
		viewport_inside \
    using type = CONCAT(_, name); \
	}; \
	using name = CONCAT(name, _cond)<1>; \
	struct CONCAT(_, name) : CONCAT(name, _cond<0>) {};

  using multitexture_image_t = std::array<fan::graphics::image_t, 30>;

  struct kps_t {
    st(light_t,
      d<visible_t> visible;
      d<uint8_t> genre;
      d<fan::graphics::viewport_t> viewport;
      d<fan::graphics::camera_t> camera;
      d<shaper_t::ShapeTypeIndex_t> ShapeType;
      d<uint8_t> draw_mode;
      d<uint32_t> vertex_count;
    );
    st(common_t,
      d<visible_t> visible;
      d<depth_t> depth;
      d<blending_t> blending;
      d<fan::graphics::viewport_t> viewport;
      d<fan::graphics::camera_t> camera;
      d<shaper_t::ShapeTypeIndex_t> ShapeType;
      d<uint8_t> draw_mode;
      d<uint32_t> vertex_count;
    );
    st(vfi_t,
      d<uint8_t> filler = 0;
    );
    st(texture_t,
      d<visible_t> visible;
      d<depth_t> depth;
      d<blending_t> blending;
      d<fan::graphics::image_t> image;
      d<fan::graphics::viewport_t> viewport;
      d<fan::graphics::camera_t> camera;
      d<shaper_t::ShapeTypeIndex_t> ShapeType;
      d<uint8_t> draw_mode;
      d<uint32_t> vertex_count;
    );
  };

  struct Key_e {
    enum : shaper_t::KeyTypeIndex_t {
      light,
      light_end,
      visible, // mainly for culling
      blending,
      depth,
      image,
      viewport,
      camera,
      ShapeType,
      filler,
      draw_mode,
      vertex_count,
      shadow
    };
  };
#endif

  // weird place for these
  struct vertex_t {
    fan::vec3 position;
    fan::color color;
  };

  struct polygon_vertex_t {
    fan::vec3 position;
    fan::color color;
    fan::vec3 offset;
    fan::vec3 angle;
    fan::vec2 rotation_point;
  };

#undef st
#pragma pack(pop)
}