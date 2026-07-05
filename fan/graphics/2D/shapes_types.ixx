module;

#if defined (FAN_WINDOW)

#include <fan/utility.h>

#if defined(FAN_OPENGL)
  #include <fan/graphics/gl_api.h>
#endif

#endif

export module fan.graphics.shapes.types;

#if defined (FAN_WINDOW)

import std;

import fan.print.error;
import fan.types.color;
import fan.types.vector;

#if defined(FAN_2D)
  import fan.window;
  import fan.graphics.common_context;
  
  #if defined(FAN_JSON)
    import fan.types.json;
  #endif

  import fan.graphics.vulkan.core;
#endif

#if defined(FAN_2D)

export namespace fan::graphics {

  #define TO_ENUM(x) x,
  #define TO_STRING(x) STRINGIFY(x),

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
    std::string_view shape_names[] = {
      GEN_SHAPES(TO_STRING, GEN_SHAPES_SKIP_STRING)
    };

  #undef TO_ENUM
  #undef TO_STRING
  #undef GEN_SHAPES_SKIP_ENUM
  #undef GEN_SHAPES_SKIP_STRING

  struct shape_gl_init_t {
    std::pair<int, const char*> index;
    std::uint32_t size;
    std::uint32_t type; // for example GL_FLOAT
    std::uint32_t stride;
    std::uint32_t offset;
  };
  struct shape_gl_init_list_t {
    fan::graphics::shape_gl_init_t* ptr = nullptr;
    int count = 0;
  };

  std::uint8_t* A_resize(void* ptr, std::uintptr_t size);
}

export namespace fan::graphics::shaper {
  #define shaper_set_fan 1
  #define shaper_set_MaxMaxElementPerBlock 0x1000
    inline constexpr std::uint32_t MaxElementPerBlock = shaper_set_MaxMaxElementPerBlock;

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

  #define shaper_set_ExpandInside_BlockProperties \
  BlockProperties_t(){\
  }\
  ~BlockProperties_t(){\
    if (0) {}\
    std::destroy_at(&renderer.vk);\
  }\
  \
  struct vk_t {\
    vk_t() = default;\
  \
    fan::vulkan::context_t::pipeline_t pipeline;\
    fan::vulkan::context_t::ssbo_t shape_data;\
    std::uint32_t vertex_count = 6;\
  };\
  \
  struct renderer_t {\
    renderer_t(){}\
    ~renderer_t(){}\
  \
    union {\
      vk_t vk;\
    };\
  };\
  \
  renderer_t renderer;

  #define shaper_set_ExpandInside_ShapeType \
  ShapeType_t(){\
  }\
  \
  struct vk_t {\
    vk_t() = default;\
  \
    fan::vulkan::context_t::pipeline_t pipeline;\
    fan::vulkan::context_t::ssbo_t shape_data;\
    std::uint32_t vertex_count = 6;\
  };\
  struct renderer_t {\
    renderer_t(){}\
    renderer_t(const renderer_t& other){\
      new (&vk) vk_t(other.vk);\
    }\
    renderer_t& operator=(const renderer_t& other){\
      if (this == &other) return *this;\
      destroy();\
      new (&vk) vk_t(other.vk);\
      return *this;\
    }\
    ~renderer_t(){\
      destroy();\
    }\
    void destroy(){\
      vk.~vk_t();\
    }\
  \
    union {\
      vk_t vk; \
    };\
  \
  }; \
  renderer_t renderer;

  #define shaper_set_ShapeTypeChange \
  __builtin_memcpy(new_renderdata, old_renderdata, element_count * GetRenderDataSize(sti));\
  __builtin_memcpy(new_data, old_data, element_count * GetDataSize(sti));

  #define shaper_set_ExpandInside_SetShapeType \
    ShapeType_t::vk_t d;\
    std::construct_at(&st.renderer.vk);\
    auto& bpr = bp.renderer.vk;\
    d.pipeline = bpr.pipeline;\
    d.shape_data = std::move(bpr.shape_data);\
    d.vertex_count = bpr.vertex_count;\
    st.renderer.vk = std::move(d);

  #define shaper_set_ExpandInside_ProcessBlockEditQueue

  #define shaper_set_ExpandInside_ProcessBlockEditQueue_Traverse \
    auto& vk = st.renderer.vk;\
    auto wrote = bu.MaxEdit - bu.MinEdit;\
    std::uint64_t dst_offset = GetRenderDataOffset(be.sti, be.blid) + bu.MinEdit;\
    std::uint64_t wanted = dst_offset + wrote;\
    auto& context = *static_cast<fan::vulkan::context_t*>(static_cast<void*>(fan::graphics::ctx()));\
    if (wanted > vk.shape_data.vram_capacity) {\
      std::uint64_t new_size = std::max(wanted, (std::uint64_t)(vk.shape_data.vram_capacity ? vk.shape_data.vram_capacity * 2 : wanted));\
      vk.shape_data.allocate(context, new_size);\
      _RenderDataReset(be.sti);\
    }\
    auto frame = context.current_frame;\
    memcpy(\
      vk.shape_data.data[frame] + dst_offset,\
      GetRenderData(be.sti, be.blid, 0) + bu.MinEdit,\
      wrote\
    );

  #define shaper_set_ExpandInside__RenderDataReset \
  BlockList_t::nrtra_t traverse;\
  BlockList_t::nr_t node_id;\
  traverse.Open(&st.BlockList, &node_id);\
  auto& vk = st.renderer.vk;\
  auto& context = *static_cast<fan::vulkan::context_t*>(static_cast<void*>(fan::graphics::ctx()));\
  auto frame = context.current_frame;\
  while (traverse.Loop(&st.BlockList, &node_id)) {\
    memcpy(vk.shape_data.data[frame] + GetRenderDataOffset(sti, node_id), GetRenderData(sti, node_id, 0), st.RenderDataSize * st.MaxElementPerBlock());\
  }\
  traverse.Close(&st.BlockList);

  #define shaper_set_ExpandInside__BlockListCapacityChange \
    auto& vk = st.renderer.vk;\
    std::uint64_t wanted = (std::uint64_t)new_capacity * st.RenderDataSize * st.MaxElementPerBlock();\
    if (wanted && wanted > vk.shape_data.vram_capacity) {\
      auto& context = *static_cast<fan::vulkan::context_t*>(static_cast<void*>(fan::graphics::ctx()));\
      std::uint64_t new_size = std::max(wanted, (std::uint64_t)(vk.shape_data.vram_capacity ? vk.shape_data.vram_capacity * 2 : wanted));\
      vk.shape_data.allocate(context, new_size);\
    }

  #define shaper_set_ExpandInside \
    fan::graphics::shader_nr_t& GetShader(ShapeTypeIndex_t sti) {\
      auto& d = ShapeTypes[sti];\
      return d.renderer.vk.pipeline.shader_nr;\
    }\
    ShapeTypes_NodeData_t& GetShapeTypes(ShapeTypeIndex_t sti) {\
      return ShapeTypes[sti];\
    }\
    \
    static std::function<void(ShapeTypes_NodeData_t&, const BlockProperties_t&)>& gl_add_shape_type() {\
    static std::function<void(ShapeTypes_NodeData_t&, const BlockProperties_t&)> f;\
    return f;\
  }
  #define shaper_set_ExpandInside_ShapeID \
    bool operator==(const ShapeID_t& other) const noexcept { return gint() == other.gint(); }

  #include <fan/graphics/2D/shaper.h>
}

namespace fan {
  template <bool cond>
  struct type_or_uint8_t {
    template <typename T>
    using d = std::conditional_t<cond, T, std::uint8_t>;
  };
}

#endif

export namespace fan::graphics {
#if defined(FAN_2D)

  struct sprite_sheet_t {
    struct image_t {
      fan::graphics::image_t image{fan::graphics::ctx().default_texture};
      int hframes = 1, vframes = 1;
    #if defined(FAN_JSON)
      operator fan::json() const;
      sprite_sheet_t::image_t& assign(const fan::json& j, const std::source_location& callers_path = std::source_location::current());
    #endif
    };
    sprite_sheet_t() = default;
    sprite_sheet_t(const std::string& name, int fps, const std::vector<fan::graphics::image_t>& frame_images);
    std::vector<int> selected_frames;
    std::vector<sprite_sheet_t::image_t> images;
    std::string name;
    int fps = 15;
    bool loop = true;
  };


  struct sprite_sheet_id_t {
    sprite_sheet_id_t();
    sprite_sheet_id_t(std::uint32_t id);
    operator std::uint32_t() const;
    explicit operator bool() const;
    sprite_sheet_id_t operator++(int);
    bool operator==(const sprite_sheet_id_t& other) const;
    bool operator!=(const sprite_sheet_id_t& other) const;
    std::uint32_t id = -1;
  };

  using fan::graphics::shaper::MaxElementPerBlock;

  using shaper_t = shaper::shaper_t;

#pragma pack(push, 1)

  using blending_t = std::uint8_t;
  using depth_t = std::uint16_t;
  using visible_t = std::uint8_t;
  using shader_raw_t = decltype(fan::graphics::shader_t::NRI);

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
      d<std::uint8_t> genre;
      d<fan::graphics::viewport_t> viewport;
      d<fan::graphics::camera_t> camera;
      d<shaper_t::ShapeTypeIndex_t> ShapeType;
      d<std::uint8_t> draw_mode;
      d<std::uint32_t> vertex_count;
    );
    st(common_t,
      d<visible_t> visible;
      d<depth_t> depth;
      d<blending_t> blending;
      d<fan::graphics::viewport_t> viewport;
      d<fan::graphics::camera_t> camera;
      d<shaper_t::ShapeTypeIndex_t> ShapeType;
      d<std::uint8_t> draw_mode;
      d<std::uint32_t> vertex_count;
    );
    st(vfi_t,
      d<std::uint8_t> filler = 0;
    );
    st(texture_t,
      d<visible_t> visible;
      d<depth_t> depth;
      d<shader_raw_t> shader_raw;
      d<blending_t> blending;
      d<fan::graphics::image_t> image;
      d<fan::graphics::viewport_t> viewport;
      d<fan::graphics::camera_t> camera;
      d<shaper_t::ShapeTypeIndex_t> ShapeType;
      d<std::uint8_t> draw_mode;
      d<std::uint32_t> vertex_count;
    );
  };

  struct Key_e {
    enum : shaper_t::KeyTypeIndex_t {
      light,
      light_end,
      visible, // mainly for culling
      blending,
      depth,
      shader,
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


#if defined(FAN_2D)
export namespace std {
  template<>
  struct hash<fan::graphics::shaper::shaper_t::ShapeID_t> {
    std::size_t operator()(const fan::graphics::shaper::shaper_t::ShapeID_t& s) const noexcept {
      return std::hash<std::uint32_t>()(s.NRI);
    }
  };
}
#endif

#endif

