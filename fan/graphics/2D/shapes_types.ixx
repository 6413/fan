module;

#if defined (FAN_WINDOW)

#include <fan/utility.h>
#include <cstdint>
#include <vk_mem_alloc.h>

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



  std::uint8_t* A_resize(void* ptr, std::uintptr_t size);
}

export namespace fan::graphics::shaper {
  #define shaper_set_fan 1
  #define shaper_set_MaxMaxElementPerBlock 0x4000
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
    struct pending_update_t {\
      std::uint32_t src_frame;\
      std::uint64_t dst_offset;\
      std::uint64_t wrote;\
    };\
    std::vector<pending_update_t> pending_updates[fan::vulkan::max_frames_in_flight];\
    bool is_queued = false;\
  \
    static void memory_cb(fan::vulkan::context_t& context, void* user_data) {\
      auto* vk_ptr = static_cast<vk_t*>(user_data);\
      vk_ptr->is_queued = false;\
      auto _f = context.current_frame;\
      for (const auto& update : vk_ptr->pending_updates[_f]) {\
        if (vk_ptr->shape_data.data[_f] && vk_ptr->shape_data.data[update.src_frame]) {\
          std::memcpy(vk_ptr->shape_data.data[_f] + update.dst_offset, vk_ptr->shape_data.data[update.src_frame] + update.dst_offset, update.wrote);\
          fan::vulkan::validate(vmaFlushAllocation(context.allocator, vk_ptr->shape_data.common.memory[_f].device_memory, update.dst_offset, update.wrote));\
        }\
      }\
      vk_ptr->pending_updates[_f].clear();\
      bool has_more = false;\
      for (std::uint32_t i = 0; i < fan::vulkan::max_frames_in_flight; ++i) {\
        if (!vk_ptr->pending_updates[i].empty()) {\
          has_more = true;\
          break;\
        }\
      }\
      if (has_more) {\
        vk_ptr->queue(context);\
      }\
    }\
    void queue(fan::vulkan::context_t& context) {\
      if (!is_queued) {\
        is_queued = true;\
        context.memory_queue.push_back(memory_cb, this);\
      }\
    }\
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
    struct pending_update_t {\
      std::uint32_t src_frame;\
      std::uint64_t dst_offset;\
      std::uint64_t wrote;\
    };\
    std::vector<pending_update_t> pending_updates[fan::vulkan::max_frames_in_flight];\
    bool is_queued = false;\
  \
    static void memory_cb(fan::vulkan::context_t& context, void* user_data) {\
      auto* vk_ptr = static_cast<vk_t*>(user_data);\
      vk_ptr->is_queued = false;\
      auto _f = context.current_frame;\
      for (const auto& update : vk_ptr->pending_updates[_f]) {\
        if (vk_ptr->shape_data.data[_f] && vk_ptr->shape_data.data[update.src_frame]) {\
          std::memcpy(vk_ptr->shape_data.data[_f] + update.dst_offset, vk_ptr->shape_data.data[update.src_frame] + update.dst_offset, update.wrote);\
          fan::vulkan::validate(vmaFlushAllocation(context.allocator, vk_ptr->shape_data.common.memory[_f].device_memory, update.dst_offset, update.wrote));\
        }\
      }\
      vk_ptr->pending_updates[_f].clear();\
      bool has_more = false;\
      for (std::uint32_t i = 0; i < fan::vulkan::max_frames_in_flight; ++i) {\
        if (!vk_ptr->pending_updates[i].empty()) {\
          has_more = true;\
          break;\
        }\
      }\
      if (has_more) {\
        vk_ptr->queue(context);\
      }\
    }\
    void queue(fan::vulkan::context_t& context) {\
      if (!is_queued) {\
        is_queued = true;\
        context.memory_queue.push_back(memory_cb, this);\
      }\
    }\
  };\
  struct renderer_t {\
    renderer_t(){}\
    renderer_t(const renderer_t& other){\
      new (&vk) vk_t(other.vk);\
    }\
    renderer_t(renderer_t&& other) noexcept {\
      new (&vk) vk_t(std::move(other.vk));\
    }\
    renderer_t& operator=(const renderer_t& other){\
      if (this == &other) return *this;\
      destroy();\
      new (&vk) vk_t(other.vk);\
      return *this;\
    }\
    renderer_t& operator=(renderer_t&& other) noexcept {\
      if (this == &other) return *this;\
      destroy();\
      new (&vk) vk_t(std::move(other.vk));\
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
      std::uint64_t capacity_wanted = (std::uint64_t)(st.BlockList.NodeList.e.c + st.BlockList.NodeList.e.p + 1) * st.RenderDataSize * st.MaxElementPerBlock();\
      std::uint64_t new_size = std::max(wanted, std::max(capacity_wanted, (std::uint64_t)(vk.shape_data.vram_capacity ? vk.shape_data.vram_capacity * 2 : wanted)));\
      vk.shape_data.allocate(context, new_size);\
      _RenderDataReset(be.sti);\
    }\
    auto src = GetRenderData(be.sti, be.blid, 0) + bu.MinEdit;\
    if (vk.shape_data.data[context.current_frame]) {\
      std::memcpy(vk.shape_data.data[context.current_frame] + dst_offset, src, wrote);\
      fan::vulkan::validate(vmaFlushAllocation(context.allocator, vk.shape_data.common.memory[context.current_frame].device_memory, dst_offset, wrote));\
    }\
    for (std::uint32_t _f = 0; _f < fan::vulkan::max_frames_in_flight; ++_f) {\
      if (_f != context.current_frame) {\
        vk.pending_updates[_f].push_back({context.current_frame, dst_offset, wrote});\
      }\
    }\
    vk.queue(context);

  #define shaper_set_ExpandInside__RenderDataReset \
  auto& vk = st.renderer.vk;\
  auto& context = *static_cast<fan::vulkan::context_t*>(static_cast<void*>(fan::graphics::ctx()));\
  for (std::uint32_t _frame = 0; _frame < fan::vulkan::max_frames_in_flight; ++_frame) {\
    vk.pending_updates[_frame].clear();\
    if (vk.shape_data.data[_frame] == nullptr) continue;\
    BlockList_t::nrtra_t traverse;\
    BlockList_t::nr_t node_id;\
    traverse.Open(&st.BlockList, &node_id);\
    while (traverse.Loop(&st.BlockList, &node_id)) {\
      auto _rdo = GetRenderDataOffset(sti, node_id);\
      auto _rsize = st.RenderDataSize * st.MaxElementPerBlock();\
      std::memcpy(vk.shape_data.data[_frame] + _rdo, GetRenderData(sti, node_id, 0), _rsize);\
      fan::vulkan::validate(vmaFlushAllocation(context.allocator, vk.shape_data.common.memory[_frame].device_memory, _rdo, _rsize));\
    }\
    traverse.Close(&st.BlockList);\
  }

  #define shaper_set_ExpandInside__BlockListCapacityChange \
    auto& vk = st.renderer.vk;\
    std::uint64_t wanted = (std::uint64_t)(new_capacity + 1) * st.RenderDataSize * st.MaxElementPerBlock();\
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
#undef st
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
      d<std::uint8_t> genre;
      d<visible_t> visible;
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
      d<fan::graphics::viewport_t> viewport;
      d<fan::graphics::camera_t> camera;
      d<shaper_t::ShapeTypeIndex_t> ShapeType;
      d<std::uint8_t> draw_mode;
      d<std::uint32_t> vertex_count;
    );
    st(particles_t,
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
      visible,
      depth,
      shader,
      blending,
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

