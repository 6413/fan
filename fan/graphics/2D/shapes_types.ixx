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

#ifdef FAN_VULKAN
  #define vulkan_expand(...) __VA_ARGS__
#else
  #define vulkan_expand(...)
#endif

#ifdef FAN_OPENGL
  #define opengl_expand(...) __VA_ARGS__
#else
  #define opengl_expand(...)
#endif

#define shaper_set_ExpandInside_BlockProperties \
BlockProperties_t(){\
  opengl_expand(std::construct_at(&renderer.gl, gl_t{});)\
}\
~BlockProperties_t(){\
  if (0) {}\
  opengl_expand(\
    else if (fan::graphics::ctx().get_renderer() == fan::window_t::renderer_t::opengl) {\
      std::destroy_at(&renderer.gl);\
    }\
  )\
  vulkan_expand(\
    else if (fan::graphics::ctx().get_renderer() == fan::window_t::renderer_t::vulkan) {\
      std::destroy_at(&renderer.vk);\
    }\
  )\
}\
\
opengl_expand(\
struct gl_t {\
  gl_t() = default;\
\
  shape_gl_init_list_t locations;\
  fan::graphics::shader_nr_t shader;\
  bool instanced = true;\
  GLuint draw_mode = GL_TRIANGLES;\
  GLsizei vertex_count = 6;\
};\
)\
\
vulkan_expand(\
struct vk_t {\
  vk_t() = default;\
\
  fan::vulkan::context_t::pipeline_t pipeline;\
  fan::vulkan::context_t::ssbo_t shape_data;\
  uint32_t vertex_count = 6;\
};\
)\
\
struct renderer_t {\
  renderer_t(){}\
  ~renderer_t(){}\
\
  union {\
    opengl_expand(gl_t gl;)\
    vulkan_expand(vk_t vk;)\
  };\
};\
\
renderer_t renderer;

#define shaper_set_ExpandInside_ShapeType \
ShapeType_t(){\
  opengl_expand(std::construct_at(&renderer.gl, gl_t{});)\
}\
\
opengl_expand(\
struct gl_t {\
  gl_t() = default;\
\
  fan::opengl::core::vao_t m_vao;\
  fan::opengl::core::vbo_t m_vbo;\
  shape_gl_init_list_t locations;\
  fan::graphics::shader_nr_t shader;\
  bool instanced = true;\
  int vertex_count = 6;\
};\
)\
\
vulkan_expand(\
struct vk_t {\
  vk_t() = default;\
\
  fan::vulkan::context_t::pipeline_t pipeline;\
  fan::vulkan::context_t::ssbo_t shape_data;\
  uint32_t vertex_count = 6;\
};\
)\
\
struct renderer_t {\
  enum class type_t {\
    none,\
    opengl_expand(gl,)\
    vulkan_expand(vk)\
  };\
\
  renderer_t(){}\
  renderer_t(const renderer_t& other){\
    type = other.type;\
    switch (type) {\
    opengl_expand(\
    case type_t::gl:\
      new (&gl) gl_t(other.gl);\
      break;\
    )\
    vulkan_expand(\
    case type_t::vk:\
      new (&vk) vk_t(other.vk);\
      break;\
    )\
    case type_t::none:\
      break;\
    }\
  }\
  renderer_t& operator=(const renderer_t& other){\
    if (this == &other) return *this;\
    destroy();\
    type = other.type;\
    switch (type) {\
    opengl_expand(\
    case type_t::gl:\
      new (&gl) gl_t(other.gl);\
      break;\
    )\
    vulkan_expand(\
    case type_t::vk:\
      new (&vk) vk_t(other.vk);\
      break;\
    )\
    case type_t::none:\
      break;\
    }\
    return *this;\
  }\
  ~renderer_t(){\
    destroy();\
  }\
  void destroy(){\
    switch (type) {\
    opengl_expand(\
    case type_t::gl:\
      gl.~gl_t();\
      break;\
    )\
    vulkan_expand(\
    case type_t::vk:\
      vk.~vk_t();\
      break;\
    )\
    case type_t::none:\
      break;\
    }\
    type = type_t::none;\
  }\
\
  union {\
    opengl_expand(gl_t gl;)\
    vulkan_expand(vk_t vk;)\
  };\
\
  type_t type = opengl_expand(type_t::gl) vulkan_expand(type_t::vk);\
};\
\
renderer_t renderer;

#define shaper_set_ShapeTypeChange \
__builtin_memcpy(new_renderdata, old_renderdata, element_count * GetRenderDataSize(sti));\
__builtin_memcpy(new_data, old_data, element_count * GetDataSize(sti));

#define shaper_set_ExpandInside_SetShapeType \
opengl_expand(\
if (fan::graphics::ctx().get_renderer() == fan::window_t::renderer_t::opengl) {\
  ShapeType_t::gl_t d;\
  st.renderer.gl = d;\
  shaper_t::gl_add_shape_type()(st, bp);\
}\
)\
vulkan_expand(\
else if (fan::graphics::ctx().get_renderer() == fan::window_t::renderer_t::vulkan) {\
  ShapeType_t::vk_t d;\
  std::construct_at(&st.renderer.vk);\
  auto& bpr = bp.renderer.vk;\
  d.pipeline = bpr.pipeline;\
  d.shape_data = bpr.shape_data;\
  d.vertex_count = bpr.vertex_count;\
  st.renderer.vk = d;\
}\
)

#define shaper_set_ExpandInside_ProcessBlockEditQueue \
opengl_expand(\
fan::opengl::context_t &context = *static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::ctx()));\
)

#define shaper_set_ExpandInside_ProcessBlockEditQueue_Traverse \
opengl_expand(\
if (fan::graphics::ctx().get_renderer() == fan::window_t::renderer_t::opengl) {\
  auto& gl = st.renderer.gl;\
  gl.m_vao.bind(context);\
  fan::opengl::core::edit_glbuffer(\
    context,\
    gl.m_vbo.m_buffer,\
    GetRenderData(be.sti, be.blid, 0) + bu.MinEdit,\
    GetRenderDataOffset(be.sti, be.blid) + bu.MinEdit,\
    bu.MaxEdit - bu.MinEdit,\
    GL_ARRAY_BUFFER\
  );\
}\
)\
vulkan_expand(\
else if (fan::graphics::ctx().get_renderer() == fan::window_t::renderer_t::vulkan) {\
  auto& vk = st.renderer.vk;\
  auto wrote = bu.MaxEdit - bu.MinEdit;\
  for (uint32_t frame = 0; frame < fan::vulkan::max_frames_in_flight; frame++) {\
    memcpy(\
      vk.shape_data.data[frame] + (GetRenderDataOffset(be.sti, be.blid) + bu.MinEdit),\
      GetRenderData(be.sti, be.blid, 0) + bu.MinEdit,\
      wrote\
    );\
  }\
}\
)

#define shaper_set_ExpandInside__RenderDataReset \
BlockList_t::nrtra_t traverse;\
BlockList_t::nr_t node_id;\
traverse.Open(&st.BlockList, &node_id);\
\
opengl_expand(\
if (fan::graphics::ctx().get_renderer() == fan::window_t::renderer_t::opengl) {\
  auto& gl = st.renderer.gl;\
  fan::opengl::context_t &context = *static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::ctx()));\
  gl.m_vao.bind(context);\
  while(traverse.Loop(&st.BlockList, &node_id)){\
    fan::opengl::core::edit_glbuffer(\
      context,\
      gl.m_vbo.m_buffer,\
      GetRenderData(sti, node_id, 0),\
      GetRenderDataOffset(sti, node_id),\
      st.RenderDataSize * st.MaxElementPerBlock(),\
      GL_ARRAY_BUFFER\
    );\
  }\
}\
)\
vulkan_expand(\
else if (fan::graphics::ctx().get_renderer() == fan::window_t::renderer_t::vulkan){\
  auto& vk = st.renderer.vk;\
  while (traverse.Loop(&st.BlockList, &node_id)) {\
    for (uint32_t frame = 0; frame < fan::vulkan::max_frames_in_flight; frame++) {\
      memcpy(vk.shape_data.data[frame], GetRenderData(sti, node_id, 0), st.RenderDataSize * st.MaxElementPerBlock());\
    }\
  }\
}\
)\
traverse.Close(&st.BlockList);

#define shaper_set_ExpandInside__BlockListCapacityChange \
opengl_expand(\
if (fan::graphics::ctx().get_renderer() == fan::window_t::renderer_t::opengl) {\
  auto& gl = st.renderer.gl;\
  gl.m_vbo.bind(*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::ctx())));\
  fan::opengl::core::write_glbuffer(\
    *static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::ctx())),\
    gl.m_vbo.m_buffer,\
    0,\
    new_capacity * st.RenderDataSize * st.MaxElementPerBlock(),\
    GL_DYNAMIC_DRAW,\
    GL_ARRAY_BUFFER\
  );\
  _RenderDataReset(sti);\
}\
)\
vulkan_expand(\
else if (fan::graphics::ctx().get_renderer() == fan::window_t::renderer_t::vulkan){\
  _RenderDataReset(sti);\
}\
)

#define shaper_set_ExpandInside \
opengl_expand(\
fan::graphics::shader_nr_t& GetShader(ShapeTypeIndex_t sti) {\
  auto& d = ShapeTypes[sti];\
  if (fan::graphics::ctx().get_renderer() == fan::window_t::renderer_t::opengl) {\
    return d.renderer.gl.shader;\
  }\
  vulkan_expand(\
  else if (fan::graphics::ctx().get_renderer() == fan::window_t::renderer_t::vulkan) {\
    return d.renderer.vk.pipeline.shader_nr;\
  }\
  )\
  fan::throw_error("");\
  static fan::graphics::shader_nr_t doesnt_happen;\
  return doesnt_happen;\
}\
fan::opengl::core::vao_t GetVAO(ShapeTypeIndex_t sti) {\
  auto& st = ShapeTypes[sti];\
  if (fan::graphics::ctx().get_renderer() == fan::window_t::renderer_t::opengl) {\
    return st.renderer.gl.m_vao;\
  }\
  fan::throw_error("Unsupported renderer type");\
  fan::opengl::core::vao_t doesnt_happen;\
  return doesnt_happen;\
}\
fan::opengl::core::vbo_t GetVBO(ShapeTypeIndex_t sti) {\
  auto& st = ShapeTypes[sti];\
  if (fan::graphics::ctx().get_renderer() == fan::window_t::renderer_t::opengl) {\
    return st.renderer.gl.m_vbo;\
  }\
  fan::throw_error("Unsupported renderer type");\
  fan::opengl::core::vbo_t doesnt_happen;\
  return doesnt_happen;\
}\
fan::graphics::shape_gl_init_list_t& GetLocations(ShapeTypeIndex_t sti) {\
  auto& st = ShapeTypes[sti];\
  if (fan::graphics::ctx().get_renderer() == fan::window_t::renderer_t::opengl) {\
    return st.renderer.gl.locations;\
  }\
  fan::throw_error("Unsupported renderer type");\
  __unreachable();\
  static fan::graphics::shape_gl_init_list_t doesnt_happen;\
  return doesnt_happen;\
}\
ShapeTypes_NodeData_t& GetShapeTypes(ShapeTypeIndex_t sti) {\
  return ShapeTypes[sti];\
}\
)\
\
static auto& gl_add_shape_type() {\
  static std::function<void(ShapeTypes_NodeData_t&, const BlockProperties_t&)> f;\
  return f;\
}

#include <fan/graphics/2D/shaper.h>
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