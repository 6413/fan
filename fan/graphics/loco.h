#ifndef loco_vulkan
#define loco_opengl
#endif

#if !defined(loco_gl_major)
  #define loco_gl_major 3
#endif
#if !defined(loco_gl_minor)
  #define loco_gl_minor 2
#endif

struct loco_t;

// to set new loco use gloco = new_loco;
inline struct global_loco_t {

  loco_t* loco = nullptr;

  operator loco_t* () {
    return loco;
  }
  global_loco_t& operator=(loco_t* l) {
    loco = l;
    return *this;
  }
  loco_t* operator->() {
    return loco;
  }
}thread_local gloco;

struct loco_t;

#ifndef loco_legacy
  //#define loco_framebuffer
#endif

#include _FAN_PATH(graphics/graphics.h)
#include _FAN_PATH(time/timer.h)
#include _FAN_PATH(font.h)
#include _FAN_PATH(physics/collision/circle.h)
#include _FAN_PATH(io/directory.h)
#include _FAN_PATH(event/event.h)

#include _FAN_PATH(trees/quad_tree.h)
#include _FAN_PATH(graphics/divider.h)


#if defined(loco_imgui) && defined(fan_platform_linux)
static void imgui_xorg_init();
static void imgui_xorg_new_frame();
#endif

// automatically gets necessary macros for shapes

#if defined(loco_cuda)
#include "cuda_runtime.h"
#include <cuda.h>
#include <nvcuvid.h>

namespace fan {
  namespace cuda {
    void check_error(auto result) {
      if (result != CUDA_SUCCESS) {
        if constexpr (std::is_same_v<decltype(result), CUresult>) {
          const char* err_str = nullptr;
          cuGetErrorString(result, &err_str);
          fan::throw_error("function failed with:" + std::to_string(result) + ", " + err_str);
        }
        else {
          fan::throw_error("function failed with:" + std::to_string(result) + ", ");
        }
      }
    }
  }
}

extern "C" {
  extern __host__ cudaError_t CUDARTAPI cudaGraphicsGLRegisterImage(struct cudaGraphicsResource** resource, fan::opengl::GLuint image, fan::opengl::GLenum target, unsigned int flags);
}

#endif

#if defined(loco_sprite_sheet)
  #define loco_sprite
#endif
#if defined(loco_sprite)
#if defined(loco_opengl)
#define loco_texture_pack
#define loco_unlit_sprite
#endif
#endif

#if defined(loco_button)
#define loco_letter
#define loco_text
#define loco_vfi
#endif

#if defined(loco_text)
#define loco_letter
#define loco_responsive_text
#endif



#ifdef loco_vulkan
#ifdef loco_line 
#ifndef loco_vulkan_descriptor_ssbo
#define loco_vulkan_descriptor_ssbo
#endif
#ifndef loco_vulkan_descriptor_uniform_block
#define loco_vulkan_descriptor_uniform_block
#endif
#endif
#ifdef loco_rectangle 
#ifndef loco_vulkan_descriptor_ssbo
#define loco_vulkan_descriptor_ssbo
#endif
#ifndef loco_vulkan_descriptor_uniform_block
#define loco_vulkan_descriptor_uniform_block
#endif
#endif
#ifdef loco_sprite
#ifndef loco_vulkan_descriptor_ssbo
#define loco_vulkan_descriptor_ssbo
#endif
#ifndef loco_vulkan_descriptor_uniform_block
#define loco_vulkan_descriptor_uniform_block
#endif
#ifndef loco_vulkan_descriptor_image_sampler
#define loco_vulkan_descriptor_image_sampler
#endif
#endif
#ifdef loco_yuv420p
#ifndef loco_vulkan_descriptor_ssbo
#define loco_vulkan_descriptor_ssbo
#endif
#ifndef loco_vulkan_descriptor_uniform_block
#define loco_vulkan_descriptor_uniform_block
#endif
#ifndef loco_vulkan_descriptor_image_sampler
#define loco_vulkan_descriptor_image_sampler
#endif
#endif
#ifdef loco_letter
#ifndef loco_vulkan_descriptor_ssbo
#define loco_vulkan_descriptor_ssbo
#endif
#ifndef loco_vulkan_descriptor_uniform_block
#define loco_vulkan_descriptor_uniform_block
#endif
#ifndef loco_vulkan_descriptor_image_sampler
#define loco_vulkan_descriptor_image_sampler
#endif
#endif
#if defined loco_compute_shader
#define loco_vulkan_descriptor_ssbo
#endif
#endif


// increase this to increase shape limit
using bdbt_key_type_t = uint16_t;

#define BDBT_set_prefix loco_bdbt
#define BDBT_set_type_node bdbt_key_type_t
#define BDBT_set_BitPerNode 2
#define BDBT_set_declare_rest 1
#define BDBT_set_declare_Key 0
#define BDBT_set_BaseLibrary 1
#define BDBT_set_CPP_ConstructDestruct
#include _FAN_PATH(BDBT/BDBT.h)

#define BDBT_set_prefix loco_bdbt
#define BDBT_set_type_node bdbt_key_type_t
#define BDBT_set_KeySize 0
#define BDBT_set_BitPerNode 2
#define BDBT_set_declare_rest 0 
#define BDBT_set_declare_Key 1
#define BDBT_set_base_prefix loco_bdbt
#define BDBT_set_BaseLibrary 1
#define BDBT_set_CPP_ConstructDestruct
#include _FAN_PATH(BDBT/BDBT.h)

#if defined(loco_pixel_format_renderer)

namespace fan {
  struct pixel_format {
    enum {
      undefined,
      yuv420p,
      nv12,
    };

    constexpr static uint8_t get_texture_amount(uint8_t format) {
      switch (format) {
        case undefined: {
          return 0;
        }
        case yuv420p: {
          return 3;
        }
        case nv12: {
          return 2;
        }
        default: {
          fan::throw_error("invalid format");
          return undefined;
        }
      }
    }
    constexpr static std::array<fan::vec2ui, 4> get_image_sizes(uint8_t format, const fan::vec2ui& image_size) {
      switch (format) {
        case yuv420p: {
          return std::array<fan::vec2ui, 4>{image_size, image_size / 2, image_size / 2};
        }
        case nv12: {
          return std::array<fan::vec2ui, 4>{image_size, fan::vec2ui{ image_size.x, image_size.y }};
        }
        default: {
          fan::throw_error("invalid format");
          return std::array<fan::vec2ui, 4>{};
        }
      }
    }
    template <typename T>
    constexpr static std::array<T, 4> get_image_properties(uint8_t format);
  };
}
#endif

namespace fan {
  namespace graphics {
  #if defined(loco_physics)
    static void open_bcol();
  #endif
    using direction_e = fan::graphics::viewport_divider_t::direction_e;
    inline fan::vec2 default_camera_ortho_x{-1, 1};
    inline fan::vec2 default_camera_ortho_y{-1, 1};
  }
}

struct loco_t {
  struct position2_t : public  fan::vec2 {
    //using fan::vec2::vec2;
  };

  struct position3_t : public fan::vec3 {
    using fan::vec3::vec3;
    using fan::vec3::operator=;
    position3_t& operator=(const position3_t& p) {
      fan::vec3::operator=(p);
      return *this;
    }
  };

  void use() {
    gloco = this;
    get_context().set_current(get_window());
  }

  std::vector<fan::function_t<void()>> m_draw_queue_light;
  std::vector<fan::function_t<void()>> m_post_draw;

  using cid_t = fan::graphics::cid_t;

  #define get_key_value(type) \
    *p.key.get_value<decltype(p.key)::get_index_with_type<type>()>()

  enum class shape_type_t : uint16_t {
    invalid = (uint16_t)-1,
    button = 0,
    sprite,
    text,
    hitbox,
    line,
    mark,
    rectangle,
    light,
    unlit_sprite,
    letter,
    circle,
    pixel_format_renderer,
    responsive_text,
    sprite_sheet,
    line_grid,
    grass_2d,
    shader,
    shader_light,
    rectangle_3d
  };

  // can be incorrect
  static constexpr const char* shape_names[] = {
    "Button",
    "Sprite",
    "Text",
    "Hitbox",
    "Line",
    "Mark",
    "Rectangle",
    "Light",
    "Unlit Sprite",
    "Letter",
    "Text Box",
    "Circle",
    "Pixel Format Renderer",
    "Reponsive Text"
  };


#pragma pack(push, 1)
  template <typename T, bool order_matters = false>
  struct make_push_key_t {
    T data;
    using key_t = loco_bdbt_Key_t<sizeof(decltype(data)) * 8, order_matters>;
    key_t k;
  };
  #pragma pack(pop)

#pragma pack(push, 1)
  template <typename T, bool order_matters = false>
  struct make_erase_key_t {
    T data;
    using key_t = loco_bdbt_Key_t<sizeof(decltype(data)) * 8, order_matters>;
    typename key_t::KeySize_t key_size;
    loco_bdbt_NodeReference_t key_nr;
  };
#pragma pack(pop)

  struct redraw_key_t {
    uint8_t blending;
  };

  struct lighting_t {
    static constexpr const char* ambient_name = "lighting_ambient";
    fan::vec3 ambient = fan::vec3(1, 1, 1);

    //void set_ambient() {

    //  //m_current_shader->set_vec3(loco->get_context(), loco_t::lighting_t::ambient_name, loco->lighting.ambient);
    //}
  }lighting;


protected:
  struct gloco_priority_init_t {
    gloco_priority_init_t(loco_t* l) {
      gloco = l;
    }
  }gloco_dummy;
public:
  #ifdef loco_window
  fan::window_t window;
  #endif

protected:
  #ifdef loco_context
  fan::graphics::context_t context;
  #endif

  #if defined(loco_context)
public:
    using viewport_t = fan::graphics::viewport_t;
protected:
#endif

  #if defined(loco_opengl) && defined(loco_context)

  uint32_t fb_vao, post_fb_vao;
  uint32_t fb_vbo, post_fb_vbo;

  void initialize_fb_vaos(uint32_t& vao, uint32_t& vbo) {
    static constexpr f32_t quad_vertices[] = {
      -1.0f, 1.0f, 0, 0.0f, 1.0f,
      -1.0f, -1.0f, 0, 0.0f, 0.0f,
      1.0f, 1.0f, 0, 1.0f, 1.0f,
      1.0f, -1.0f, 0, 1.0f, 0.0f,
    };
    auto& context = get_context();
    context.opengl.glGenVertexArrays(1, &vao);
    context.opengl.glGenBuffers(1, &vbo);
    context.opengl.glBindVertexArray(vao);
    context.opengl.glBindBuffer(fan::opengl::GL_ARRAY_BUFFER, vbo);
    context.opengl.glBufferData(fan::opengl::GL_ARRAY_BUFFER, sizeof(quad_vertices), &quad_vertices, fan::opengl::GL_STATIC_DRAW);
    context.opengl.glEnableVertexAttribArray(0);
    context.opengl.glVertexAttribPointer(0, 3, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, 5 * sizeof(float), (void*)0);
    context.opengl.glEnableVertexAttribArray(1);
    context.opengl.glVertexAttribPointer(1, 2, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
  }

  void render_final_fb() {
    auto& context = get_context();
    context.opengl.glBindVertexArray(fb_vao);
    context.opengl.glDrawArrays(fan::opengl::GL_TRIANGLE_STRIP, 0, 4);
    context.opengl.glBindVertexArray(0);
  }
  void render_post_fb() {
    auto& context = get_context();
    context.opengl.glBindVertexArray(post_fb_vao);
    context.opengl.glDrawArrays(fan::opengl::GL_TRIANGLE_STRIP, 0, 4);
    context.opengl.glBindVertexArray(0);
  }

  #endif

public:
  struct image_t;
  struct shader_t;

  #if defined(loco_window)

  #if defined(loco_opengl)

  #include _FAN_PATH(graphics/opengl/shader_list_builder_settings.h)
  #elif defined(loco_vulkan)
  #include _FAN_PATH(graphics/vulkan/shader_list_builder_settings.h)
  #endif
  #include _FAN_PATH(BLL/BLL.h)

  // TODO REMOVE
  //#if defined(loco_opengl)
  shader_list_t shader_list;
  //#endif

  #if defined(loco_opengl)
  #include _FAN_PATH(graphics/opengl/gl_shader.h)
  #elif defined(loco_vulkan)
  #include _FAN_PATH(graphics/vulkan/vk_shader.h)
  #endif
  #if defined(loco_opengl)
  #include _FAN_PATH(graphics/opengl/image_list_builder_settings.h)
  #elif defined(loco_vulkan)
  #include _FAN_PATH(graphics/vulkan/image_list_builder_settings.h)
  #endif
  #include _FAN_PATH(BLL/BLL.h)
  template <uint8_t n_>
  struct textureid_t : image_list_NodeReference_t {
    static constexpr std::array<const char*, 32> texture_names = {
      "_t00", "_t01", "_t02", "_t03",
      "_t04", "_t05", "_t06", "_t07",
      "_t08", "_t09", "_t10", "_t11",
      "_t12", "_t13", "_t14", "_t15",
      "_t16", "_t17", "_t18", "_t19",
      "_t20", "_t21", "_t22", "_t23",
      "_t24", "_t25", "_t26", "_t27",
      "_t28", "_t29", "_t30", "_t31"
    };
    static constexpr uint8_t n = n_;
    static constexpr auto name = texture_names[n];

    textureid_t() = default;
    textureid_t(image_t* image) : image_list_NodeReference_t::image_list_NodeReference_t(image) {
    }
  };
  #endif
  // TODO REMOVE
  //#if defined(loco_opengl)
  image_list_t image_list;
  //#endif

  #if defined(loco_opengl)
  #include _FAN_PATH(graphics/opengl/gl_image.h)
  #elif defined(loco_vulkan)
  #include _FAN_PATH(graphics/vulkan/vk_image.h)
  #endif

  // TODO camera_t;
  struct camera_t;

  #define BLL_set_declare_NodeReference 1
  #define BLL_set_declare_rest 0
  #if defined(loco_opengl)
  #include _FAN_PATH(graphics/opengl/camera_list_builder_settings.h)
  #elif defined(loco_vulkan)
  #include _FAN_PATH(graphics/vulkan/camera_list_builder_settings.h)
  #endif
  //TODO REMOVE
  #include _FAN_PATH(BLL/BLL.h)

  struct viewport_resize_cb_data_t {
    camera_t* camera;
    fan::vec3 position;
    fan::vec2 size;
  };
  using viewport_resize_cb_t = fan::function_t<void(const viewport_resize_cb_data_t&)>;

  #define BLL_set_Mark 2
  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_prefix viewport_resize_callback
  #define BLL_set_NodeData viewport_resize_cb_t data;
  #define BLL_set_CPP_nrsic 1
  #include _FAN_PATH(window/cb_list_builder_settings.h)
  #include _FAN_PATH(BLL/BLL.h)

  viewport_resize_callback_t m_viewport_resize_callback;

  struct camera_t : fan::camera {

    using resize_cb_data_t = loco_t::viewport_resize_cb_data_t;
    using resize_cb_t = loco_t::viewport_resize_cb_t;
    struct resize_callback_id_t : loco_t::viewport_resize_callback_NodeReference_t {
      using inherit_t = viewport_resize_callback_NodeReference_t;
      resize_callback_id_t() : loco_t::viewport_resize_callback_NodeReference_t() {}
      resize_callback_id_t(const inherit_t& i) : inherit_t(i) {}
      resize_callback_id_t(resize_callback_id_t&& i) : inherit_t(i) {
        i.sic();
      }

      resize_callback_id_t& operator=(const resize_callback_id_t& i) = delete;

      resize_callback_id_t& operator=(resize_callback_id_t&& i) {
        if (this != &i) {
          *(inherit_t*)this = *(inherit_t*)&i;
          i.sic();
        }
        return *this;
      }

      operator loco_t::viewport_resize_callback_NodeReference_t() {
        return *(loco_t::viewport_resize_callback_NodeReference_t*)this;
      }
      ~resize_callback_id_t() {
        if (iic()) {
          return;
        }
        gloco->m_viewport_resize_callback.unlrec(*this);
        sic();
      }
    };

    resize_callback_id_t add_resize_callback(resize_cb_t function) {
      auto nr = gloco->m_viewport_resize_callback.NewNodeLast();
      gloco->m_viewport_resize_callback[nr].data = function;
      return resize_callback_id_t(nr);
    }

    camera_t() {
      camera_reference.sic();
    }

    static constexpr f32_t znearfar = 0xffff;

    void open() {
      auto& context = gloco->get_context();
      m_view = fan::mat4(1);
      position = 0;
      camera_reference = gloco->camera_list.NewNode();
      gloco->camera_list[camera_reference].camera_id = this;
    }
    void close() {
      gloco->camera_list.Recycle(camera_reference);
    }

    void open_camera(loco_t::camera_t* camera, const fan::vec2& x, const fan::vec2& y) {
      camera->open();
      camera->set_ortho(fan::vec2(x.x, x.y), fan::vec2(y.x, y.y));
    }

    fan::vec3 get_position() const {
      return position;
    }
    void set_position(const fan::vec3& cp) {
      position = cp;

      m_view[3][0] = 0;
      m_view[3][1] = 0;
      m_view[3][2] = 0;
      m_view = m_view.translate(position);
      fan::vec3 position = m_view.get_translation();
      constexpr fan::vec3 front(0, 0, 1);

      m_view = fan::math::look_at_left<fan::mat4, fan::vec3>(position, position + front, fan::camera::world_up);
    }

    fan::vec2 get_camera_size() const {
      return fan::vec2(std::abs(coordinates.right - coordinates.left), std::abs(coordinates.down - coordinates.up));
    }

    void set_ortho(fan::vec2 x, fan::vec2 y) {

      coordinates.left = x.x;
      coordinates.right = x.y;
      coordinates.down = y.y;
      coordinates.up = y.x;

      m_projection = fan::math::ortho<fan::mat4>(
        coordinates.left,
        coordinates.right,
        #if defined (loco_opengl)
        coordinates.down,
        coordinates.up,
        0.1,
        znearfar / 2
        #elif defined(loco_vulkan)
        coordinates.up,
        coordinates.down,
        -znearfar / 2,
        znearfar / 2
        #endif
      );

      m_view[3][0] = 0;
      m_view[3][1] = 0;
      m_view[3][2] = 0;
      m_view = m_view.translate(position);
      fan::vec3 position = m_view.get_translation();
      constexpr fan::vec3 front(0, 0, 1);

      m_view = fan::math::look_at_left<fan::mat4, fan::vec3>(position, position + front, fan::camera::world_up);
      
      auto it = gloco->m_viewport_resize_callback.GetNodeFirst();

      while (it != gloco->m_viewport_resize_callback.dst) {

        gloco->m_viewport_resize_callback.StartSafeNext(it);

        resize_cb_data_t cbd;
        cbd.camera = this;
        cbd.position = get_position();
        cbd.size = get_camera_size();
        gloco->m_viewport_resize_callback[it].data(cbd);

        it = gloco->m_viewport_resize_callback.EndSafeNext();
      }
    }

    void set_perspective(f32_t fov) {

      m_projection = fan::math::perspective<fan::mat4>(fan::math::radians(fov), (f32_t)gloco->window.get_size().x / (f32_t)gloco->window.get_size().y, 0.1f, 1000.0f);

      update_view();

      m_view = fan::math::look_at_left<fan::mat4, fan::vec3>(fan::vec3(position), position + m_front, m_up);

      auto it = gloco->m_viewport_resize_callback.GetNodeFirst();

      while (it != gloco->m_viewport_resize_callback.dst) {

        gloco->m_viewport_resize_callback.StartSafeNext(it);

        resize_cb_data_t cbd;
        cbd.camera = this;
        cbd.position = get_position();
        cbd.size = get_camera_size();
        gloco->m_viewport_resize_callback[it].data(cbd);

        it = gloco->m_viewport_resize_callback.EndSafeNext();
      }
    }

    void rotate_camera(const fan::vec2& offset) {
      fan::camera::rotate_camera(offset);
      m_view = get_view_matrix();
    }

    void move(f32_t movement_speed, f32_t friction = 12) {
      fan::camera::move(movement_speed, friction);
      m_view = get_view_matrix();
    }


    fan::mat4 m_projection;
    // temporary
    fan::mat4 m_view;

    //fan::vec3 camera_position;


    union {
      struct {
        f32_t left;
        f32_t right;
        f32_t up;
        f32_t down;
      };
      fan::vec4 v;
    }coordinates;

    camera_list_NodeReference_t camera_reference;
  };

  void open_camera(camera_t* camera, const fan::vec2& x, const fan::vec2& y) {
    camera->open();
    camera->set_ortho(x, y);
  }

  void open_camera(camera_t* camera, f32_t fov) {
    camera->open();
    camera->set_perspective(fov);
  }

  void open_viewport(fan::graphics::viewport_t* viewport, const fan::vec2& viewport_position, const fan::vec2& viewport_size) {
    viewport->open();
    viewport->set(viewport_position, viewport_size, window.get_size());
  }

  void set_viewport(fan::graphics::viewport_t* viewport, const fan::vec2& viewport_position, const fan::vec2& viewport_size) {
    viewport->set(viewport_position, viewport_size, window.get_size());
  }

  struct camera_impl_t {

    camera_impl_t() = default;
    camera_impl_t(fan::graphics::direction_e split_direction) {
      fan::graphics::viewport_divider_t::iterator_t it = gloco->viewport_divider.insert(split_direction);
      fan::vec2 p = it.parent->position;
      fan::vec2 s = it.parent->size;

      fan::vec2 window_size = gloco->window.get_size();
      gloco->open_viewport(&viewport, (p - s / 2) * window_size, (s)*window_size);
      gloco->open_camera(&camera, fan::graphics::default_camera_ortho_x, fan::graphics::default_camera_ortho_y);
    }
    loco_t::camera_t camera;
    //TODO REMOVE
    //#if defined(loco_opengl)
    fan::graphics::viewport_t viewport;
    //#endif
  };

  #define BLL_set_declare_NodeReference 0
  #define BLL_set_declare_rest 1

  // TODO should be global header
  #if defined(loco_opengl)
  #include _FAN_PATH(graphics/opengl/camera_list_builder_settings.h)
  #elif defined(loco_vulkan)
  #include _FAN_PATH(graphics/vulkan/camera_list_builder_settings.h)
  #endif
  #include _FAN_PATH(BLL/BLL.h)

  camera_list_t camera_list;

  image_t unloaded_image;
  fan::color clear_color = {0.10, 0.10, 0.131, 1};

  #if defined(loco_texture_pack)
  #include _FAN_PATH(graphics/opengl/texture_pack.h)
  #endif

  #if defined(loco_tp)
  #if defined(loco_opengl)
  #include _FAN_PATH(tp/tp0.h)
  #endif
  #endif

protected:
  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_AreWeInsideStruct 1
  #include _FAN_PATH(fan_bll_preset.h)
  #define BLL_set_prefix cid_list
  #define BLL_set_type_node uint32_t
  #define BLL_set_NodeData fan::graphics::cid_t cid;
  #define BLL_set_Link 1
  #include _FAN_PATH(BLL/BLL.h)
public:

  struct cid_nt_t : cid_list_NodeReference_t {
    loco_t::cid_t* operator->() const {
      return &gloco->cid_list[*(cid_list_NodeReference_t*)(this)].cid;
    }
    using base_t = cid_list_NodeReference_t;
    void init() {
      *(base_t*)this = gloco->cid_list.NewNodeLast();
    }

    bool is_invalid() const {
      return cid_list_inric(*this);
    }

    void invalidate_soft() {
      *(base_t*)this = gloco->cid_list.gnric();
    }

    void invalidate() {
      if (is_invalid()) {
        return;
      }
      gloco->cid_list.unlrec(*this);
      *(base_t*)this = gloco->cid_list.gnric();
    }

    uint32_t* gdp4() {
      return (uint32_t*)&(*this)->bm_id;
    }
  };

  struct cid_nr_t : cid_nt_t {

    cid_nr_t() { *(cid_list_NodeReference_t*)this = cid_list_gnric(); }
    cid_nr_t(const cid_nt_t& c) : cid_nt_t(c) {

    }

    cid_nr_t(const cid_nr_t& nr) : cid_nr_t() {
      if (nr.is_invalid()) {
        return;
      }
      init();
      gloco->cid_list[*this].cid.shape_type = gloco->cid_list[nr].cid.shape_type;
    }

    cid_nr_t(cid_nr_t&& nr) {
      NRI = nr.NRI;
      nr.invalidate_soft();
    }

    loco_t::cid_nr_t& operator=(const cid_nr_t& id) {
      if (!is_invalid()) {
        invalidate();
      }
      if (id.is_invalid()) {
        return *this;
      }

      if (this != &id) {
        init();
        gloco->cid_list[*this].cid.shape_type = gloco->cid_list[id].cid.shape_type;
      }
      return *this;
    }

    loco_t::cid_nr_t& operator=(cid_nr_t&& id) {
      if (!is_invalid()) {
        invalidate();
      }
      if (id.is_invalid()) {
        return *this;
      }

      if (this != &id) {
        if (!is_invalid()) {
          invalidate();
        }
        NRI = id.NRI;

        id.invalidate_soft();
      }
      return *this;
    }
  };

  #if defined(loco_vfi)

  #define vfi_var_name vfi
  #include _FAN_PATH(graphics/gui/vfi.h)

  #define create_loco_wrap(type) struct type : vfi_t::type { \
    type(const vfi_t::type& mm) : vfi_t::type(mm) {} \
    loco_t::cid_nt_t id; \
  } 

  create_loco_wrap(mouse_move_data_t);
  create_loco_wrap(mouse_button_data_t);
  create_loco_wrap(keyboard_data_t);
  create_loco_wrap(text_data_t);

  #undef create_loco_wrap

  using mouse_move_cb_t = fan::function_t<int(const mouse_move_data_t&)>;
  using mouse_button_cb_t = fan::function_t<int(const mouse_button_data_t&)>;
  using keyboard_cb_t = fan::function_t<int(const keyboard_data_t&)>;
  using text_cb_t = fan::function_t<int(const text_data_t&)>;

  #include _FAN_PATH(graphics/gui/themes.h)

  #endif

  #ifdef loco_window
  using mouse_buttons_cb_data_t = fan::window_t::mouse_buttons_cb_data_t;
  using keyboard_keys_cb_data_t = fan::window_t::keyboard_keys_cb_data_t;
  using keyboard_key_cb_data_t = fan::window_t::keyboard_key_cb_data_t;
  using text_cb_data_t = fan::window_t::text_cb_data_t;
  using mouse_move_cb_data_t = fan::window_t::mouse_move_cb_data_t;
  using close_cb_data_t = fan::window_t::close_cb_data_t;
  using resize_cb_data_t = fan::window_t::resize_cb_data_t;
  using move_cb_data_t = fan::window_t::move_cb_data_t;
  #endif

  struct properties_t {
    bool vsync = true;
    fan::vec2 window_size = fan::sys::get_screen_resolution() / fan::vec2(1.2, 1.2);
    uint64_t window_flags = 0;
  };

  static constexpr uint32_t max_depths = 2;

  #ifdef loco_window
  fan::window_t* get_window() {
    return &window;
  }
  #endif

  #ifdef loco_context
  // required function if program using multiple contexts, so it gets current
  fan::graphics::context_t& get_context() {
    return context;
  }
  #endif

  f64_t& delta_time = window.m_delta_time;

  #if defined(loco_window)
  f32_t get_delta_time() {
    return delta_time;
  }
  #endif

  struct push_constants_t {
    uint32_t texture_id;
    uint32_t camera_id;
  };

  #if defined(loco_window)
  void process_block_properties_element(auto* shape, loco_t::camera_list_NodeReference_t camera_id) {
    #if defined(loco_opengl)
    auto* camera = camera_list[camera_id].camera_id;
    shape->m_current_shader->use();
    shape->m_current_shader->set_camera(camera, &m_write_queue);
    shape->m_current_shader->set_vec2("matrix_size",
      fan::vec2(camera->coordinates.right - camera->coordinates.left, camera->coordinates.down - camera->coordinates.up).abs()
    );
    shape->m_current_shader->set_vec2("camera_position", camera->get_position());
    #elif defined(loco_vulkan)
    uint32_t idx;
    auto* camera = &camera_list[camera_id];
    // TODO fix this mess with array
    #if defined(loco_line)
    if constexpr (std::is_same<typename std::remove_pointer<decltype(shape)>::type, line_t>::value) {
      idx = camera->camera_index.line;
    }
    #endif
    #if defined(loco_rectangle)
    if constexpr (std::is_same<typename std::remove_pointer<decltype(shape)>::type, shapes_t::rectangle_t>::value) {
      idx = camera->camera_index.rectangle;
    }
    #endif
    #if defined(loco_sprite)
    if constexpr (std::is_same<typename std::remove_pointer<decltype(shape)>::type, shapes_t::sprite_t>::value) {
      idx = camera->camera_index.sprite;
    }
    #endif
    #if defined(loco_letter)
    if constexpr (std::is_same<typename std::remove_pointer<decltype(shape)>::type, letter_t>::value) {
      idx = camera->camera_index.letter;
    }
    #endif
    #if defined(loco_button)
    if constexpr (std::is_same<typename std::remove_pointer<decltype(shape)>::type, button_t>::value) {
      idx = camera->camera_index.button;
    }
    #endif

    #if defined(loco_text_box)
    if constexpr (std::is_same<typename std::remove_pointer<decltype(shape)>::type, text_box_t>::value) {
      idx = camera->camera_index.text_box;
    }
    #endif


    #if defined(loco_yuv420p)
    if constexpr (std::is_same<typename std::remove_pointer<decltype(shape)>::type, yuv420p_t>::value) {
      idx = camera->camera_index.yuv420p;
    }
    #endif

    auto& context = get_context();
    vkCmdPushConstants(
      context.commandBuffers[context.currentFrame],
      shape->m_pipeline.m_layout,
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
      offsetof(push_constants_t, camera_id),
      sizeof(uint32_t),
      &idx
    );
    #endif

  }
  void process_block_properties_element(auto* shape, fan::graphics::viewport_list_NodeReference_t viewport_id) {
    fan::graphics::viewport_t* viewport = get_context().viewport_list[viewport_id].viewport_id;
    viewport->set(
      viewport->get_position(),
      viewport->get_size(),
      window.get_size()
    );
    #if defined(loco_opengl)
    shape->m_current_shader->set_vec4("viewport", fan::vec4(viewport->get_position(), viewport->get_size()));
    #endif
  }

  template <uint8_t n>
  void process_block_properties_element(auto* shape, textureid_t<n> tid) {
    #if defined(loco_opengl)
    if (tid.NRI == (decltype(tid.NRI))-1) {
      return;
    }
    shape->m_current_shader->use();
    shape->m_current_shader->set_int(tid.name, n);
    get_context().opengl.call(get_context().opengl.glActiveTexture, fan::opengl::GL_TEXTURE0 + n);
    get_context().opengl.call(get_context().opengl.glBindTexture, fan::opengl::GL_TEXTURE_2D, image_list[tid].texture_id);
    #endif
  }

  void process_block_properties_element(auto* shape, uint16_t depth) {

  }

  void process_block_properties_element(auto* shape, loco_t::shader_list_NodeReference_t shader_id) {
    #if defined(loco_opengl)
    loco_t::shader_t* shader = gloco->shader_list[shader_id].shader;
    shape->m_current_shader = shader;
    shape->m_current_shader->use();
    shape->m_current_shader->get_shader().on_activate(shape->m_current_shader);
    shape->m_current_shader->set_vec3(loco_t::lighting_t::ambient_name, gloco->lighting.ambient);
    shape->m_current_shader->set_int("_t00", 0);
    shape->m_current_shader->set_int("_t01", 1);
    #elif defined(loco_vulkan)
    assert(0);
    #endif
  }

  #endif

  loco_bdbt_t bdbt;
  loco_bdbt_NodeReference_t root;

  fan::ev_timer_t ev_timer;

  #if defined(loco_context)
  fan::graphics::core::memory_write_queue_t m_write_queue;
  #endif

  cid_list_t cid_list;

  #define fan_build_func_definition(rt, name, ...) rt name(__VA_ARGS__)

  #define fan_build_set_plain(rt, name) \
        fan_build_func_definition(void, set_##name, rt data){ gloco->shape_##set_##name(*this, data); }
  #define fan_build_get_set_plain(rt, name) \
    fan_build_func_definition(rt, get_##name) { return gloco->shape_##get_##name(*this);} \
    fan_build_set_plain(rt, name)

  #define fan_build_get_set_cref(rt, name) \
    fan_build_func_definition(rt, get_##name){ return gloco->shape_##get_##name(*this);} \
    fan_build_set_plain(const rt&, name)

  struct shape_t : cid_nr_t {
    using inherit_t = cid_nr_t;

    shape_t() { sic(); };
    template <typename T>
    requires requires(T t) { typename T::type_t; }
    shape_t(const T& properties) {
      inherit_t::init();
      gloco->push_shape(*this, properties);
    }

    inline shape_t(const shape_t& id) : 
      inherit_t(id)
    {
      if (id.is_invalid()) {
        return;
      }
      gloco->shape_get_properties(*(shape_t*)&id, [&]<typename T>(const T& properties) {
        gloco->push_shape(*this, properties);
      });
    }
    inline shape_t(shape_t&& id) : inherit_t(std::move(id)) {}
    
    loco_t::shape_t& operator=(const shape_t& id) {
      if (!is_invalid()) {
        erase();
      }
      if (id.is_invalid()) {
        return *this;
      }
      if (this != &id) {
        gloco->shape_get_properties(*(shape_t*)&id, [&]<typename T>(const T& properties) {
          init();
          gloco->push_shape(*this, properties);
        });
      }
      return *this;
    }
    
    loco_t::shape_t& operator=(shape_t&& id) {
      if (!is_invalid()) {
        erase();
      }
      if (id.is_invalid()) {
        return *this;
      }
      if (this != &id) {
        if (!is_invalid()) {
          erase();
        }
        *(inherit_t*)this = std::move(id);
        id.invalidate();
      }
      return *this;
    }

    ~shape_t() {
      erase();
    }

    void erase() {
      if (is_invalid()) {
        return;
      }
      gloco->shape_erase(*this);
      inherit_t::invalidate();
    }
    #if defined(loco_responsive_text)
    bool append_letter(wchar_t wc, bool force = false) {
      return gloco->shapes.responsive_text.append_letter(*this, wc, force);
    }
    #endif

    fan_build_get_set_cref(fan::vec3, position);

    void set_position(const fan::vec2& data) {
      gloco->shape_set_position(*this, fan::vec3(data, get_position().z));
    }

    fan_build_get_set_cref(fan::vec3, size);
    fan_build_get_set_cref(fan::color, color);
    fan_build_get_set_cref(fan::vec3, angle);
    fan_build_get_set_cref(fan::string, text);
    fan_build_get_set_cref(fan::vec2, rotation_point);
    fan_build_get_set_cref(f32_t, font_size);

    fan_build_get_set_cref(fan::color, outline_color);
    fan_build_get_set_cref(f32_t, outline_size);

    fan_build_get_set_cref(f32_t, depth);

    fan_build_get_set_cref(fan::vec2, tc_position);
    fan_build_get_set_cref(fan::vec2, tc_size);

    fan_build_get_set_plain(loco_t::viewport_t*, viewport);
    fan_build_get_set_plain(loco_t::camera_t*, camera);
    fan_build_get_set_plain(loco_t::image_t*, image);

    void set_line(const fan::vec3& src, const fan::vec2& dst) {
      gloco->shape_set_line(*this, src, dst);
    }

    #if defined(loco_sprite) && defined(loco_tp)
    // pack_id is not set here
    loco_t::texturepack_t::ti_t get_tp() {
      loco_t::texturepack_t::ti_t ti;
      ti.image = get_image();
      ti.position = get_tc_position() * ti.image->size;
      ti.size = get_tc_size() * ti.image->size;
      return ti;
    }
    bool set_tp(loco_t::texturepack_t::ti_t* ti) {
      return gloco->shapes.sprite.load_tp(*this, ti);
    }
    #endif

    operator fan::graphics::cid_t* () {
      return &gloco->cid_list[*this].cid;
    }

    loco_t* get_loco() {
      return gloco;
    }
  };

  #if defined(loco_imgui)
  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_SafeNext 1
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_prefix imgui_draw_cb
  #include _FAN_PATH(fan_bll_preset.h)
  #define BLL_set_Link 1
  #define BLL_set_type_node uint16_t
  #define BLL_set_NodeDataType fan::function_t<void()>
  #include _FAN_PATH(BLL/BLL.h)
  #endif

  #if defined(loco_imgui)
  using imgui_draw_cb_nr_t = imgui_draw_cb_NodeReference_t;
  imgui_draw_cb_t m_imgui_draw_cb;
  #endif

  #if defined(loco_imgui)
  struct imgui_element_nr_t : loco_t::imgui_draw_cb_nr_t {
    using base_t = loco_t::imgui_draw_cb_nr_t;

    imgui_element_nr_t() { /**(nr_t*)this = cid_list_gnric(); should be default*/ }

    imgui_element_nr_t(const imgui_element_nr_t& nr) : imgui_element_nr_t() {
      if (nr.is_invalid()) {
        return;
      }
      init();
    }

    imgui_element_nr_t(imgui_element_nr_t&& nr) {
      NRI = nr.NRI;
      nr.invalidate_soft();
    }
    ~imgui_element_nr_t() {
      invalidate();
    }


    imgui_element_nr_t& operator=(const imgui_element_nr_t& id) {
      if (!is_invalid()) {
        invalidate();
      }
      if (id.is_invalid()) {
        return *this;
      }

      if (this != &id) {
        init();
      }
      return *this;
    }

    imgui_element_nr_t& operator=(imgui_element_nr_t&& id) {
      if (!is_invalid()) {
        invalidate();
      }
      if (id.is_invalid()) {
        return *this;
      }

      if (this != &id) {
        if (!is_invalid()) {
          invalidate();
        }
        NRI = id.NRI;

        id.invalidate_soft();
      }
      return *this;
    }

    void init() {
      *(base_t*)this = gloco->m_imgui_draw_cb.NewNodeLast();
    }

    bool is_invalid() const {
      return loco_t::imgui_draw_cb_inric(*this);
    }

    void invalidate_soft() {
      *(base_t*)this = gloco->m_imgui_draw_cb.gnric();
    }

    void invalidate() {
      if (is_invalid()) {
        return;
      }
      gloco->m_imgui_draw_cb.unlrec(*this);
      *(base_t*)this = gloco->m_imgui_draw_cb.gnric();
    }

    void set(const auto& lambda) {
      gloco->m_imgui_draw_cb[*this] = lambda;
    }
  };

  struct imgui_element_t : imgui_element_nr_t {
    imgui_element_t() = default;
    imgui_element_t(const auto& lambda) {
      imgui_element_nr_t::init();
      imgui_element_nr_t::set(lambda);
    }
  };

  struct imgui_shape_element_t : imgui_element_t, loco_t::shape_t {
    imgui_shape_element_t(const auto& properties, const auto& lambda)
      : imgui_element_t(lambda), loco_t::shape_t(properties) {
    }
  };
  #endif

  #if defined(loco_imgui)
  imgui_element_t gui_debug_element;
  #endif

  #ifdef loco_vulkan
  struct descriptor_pool_t {

    descriptor_pool_t() {
      uint32_t total = 0;
      VkDescriptorPoolSize pool_sizes[] = {
        #ifdef loco_vulkan_descriptor_ssbo
        {
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          fan::vulkan::MAX_FRAMES_IN_FLIGHT
        },
        #endif
        #ifdef loco_vulkan_descriptor_uniform_block
        {
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          fan::vulkan::MAX_FRAMES_IN_FLIGHT
        },
        #endif
        #ifdef loco_vulkan_descriptor_image_sampler
        {
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          fan::vulkan::MAX_FRAMES_IN_FLIGHT
        },
        #endif
      };

      VkDescriptorPoolCreateInfo pool_info{};
      pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
      pool_info.poolSizeCount = std::size(pool_sizes);
      pool_info.pPoolSizes = pool_sizes;
      pool_info.maxSets = fan::vulkan::MAX_FRAMES_IN_FLIGHT * 10;

      fan::vulkan::validate(vkCreateDescriptorPool(gloco->get_context().device, &pool_info, nullptr, &m_descriptor_pool));
    }
    ~descriptor_pool_t() {
      vkDestroyDescriptorPool(gloco->get_context().device, m_descriptor_pool, nullptr);
    }

    VkDescriptorPool m_descriptor_pool;
  }descriptor_pool;
  #endif

  // requirements - create shape_type to shape.h, init in constructor, add type_t to properties
// make get_properties for custom type,
// (if no custom storaging outside vi or ri, its generated automatically)

  struct shapes_t {

    #if defined(loco_vfi)
    using vfi_t = loco_t::vfi_t;
    vfi_t vfi_var_name;
    #undef vfi_var_name
    #endif

    #if defined(loco_line)
    #define sb_depth_var src
    #define sb_shape_var_name line
    #include _FAN_PATH(graphics/opengl/2D/objects/line.h)
    line_t sb_shape_var_name;
    #undef sb_shape_var_name
    #endif
    #if defined(loco_rectangle)
    #define sb_shape_var_name rectangle
    #include _FAN_PATH(graphics/opengl/2D/objects/rectangle.h)
    rectangle_t sb_shape_var_name;
    #undef sb_shape_var_name
    #endif
    #if defined(loco_circle)
    #define sb_shape_var_name circle
    #include _FAN_PATH(graphics/opengl/2D/objects/circle.h)
    circle_t sb_shape_var_name;
    #undef sb_shape_var_name
    #endif
    #if defined(loco_pixel_format_renderer)
    #define sb_pfr_var_name pixel_format_renderer
    #define sb_pfr_name pixel_format_renderer_t
    #include _FAN_PATH(graphics/opengl/2D/objects/pixel_format_renderer.h)
    pixel_format_renderer_t sb_pfr_var_name;
    // #undef sb_pfr_name
    // #undef sb_pfr_var_name
    #endif
    #if defined(loco_sprite)
    #define sb_shape_var_name sprite
    #define sb_sprite_name sprite_t
    #include _FAN_PATH(graphics/opengl/2D/objects/sprite.h)
    sprite_t sb_shape_var_name;
    #undef sb_shape_var_name

    #if defined(loco_opengl)
    #include _FAN_PATH(graphics/opengl/2D/effects/particles.h)
    particles_t particles;
    #elif defined(loco_vulkan)
    // TODO
    #endif

    #endif
    #if defined(loco_unlit_sprite)
    #define sb_shape_var_name unlit_sprite
    #define sb_sprite_name unlit_sprite_t
    #define sb_custom_shape_type loco_t::shape_type_t::unlit_sprite
    #if defined(loco_opengl)
    #define sb_shader_fragment_path _FAN_PATH_QUOTE(graphics/glsl/opengl/2D/objects/unlit_sprite.fs)
    #include _FAN_PATH(graphics/opengl/2D/objects/sprite.h)
    unlit_sprite_t unlit_sprite;
    #elif defined(loco_vulkan)
    // TODO for now
    #define sb_shader_fragment_path graphics/glsl/vulkan/2D/objects/sprite.frag
    #endif
    #undef sb_shape_var_name
    #undef sb_custom_shape_type
    #endif
    #if defined(loco_blended_sprite)
    #define sb_shape_var_name blended_sprite
    #define sb_sprite_name blended_sprite_t
    #define sb_custom_shape_type loco_t::shape_type_t::blended_sprite
    #define sb_shader_fragment_path _FAN_PATH_QUOTE(graphics/glsl/opengl/2D/objects/blended_sprite.fs)
    #include _FAN_PATH(graphics/opengl/2D/objects/sprite.h)
    unlit_sprite_t blended_sprite;
    #undef sb_shape_var_name
    #undef sb_custom_shape_type
    #endif
    #if defined(loco_light)
    #define sb_shape_name light_t
    #define sb_shape_var_name light
    #define sb_is_light
    #include _FAN_PATH(graphics/opengl/2D/objects/light.h)
    sb_shape_name sb_shape_var_name;
    #undef sb_shape_var_name
    #undef sb_shape_name
    #undef sb_is_light
    #endif
    #if defined(loco_sprite_sheet)
    #define sb_shape_var_name sprite_sheet;
    #define sb_sprite_sheet_name sprite_sheet_t
    #include _FAN_PATH(graphics/sprite_sheet.h)
    sb_sprite_sheet_name sb_shape_var_name;
    #endif
    #if defined(loco_letter)
    #if !defined(loco_font)
    #define loco_font "fonts/bitter"
    #endif
    #define sb_shape_var_name letter
    #include _FAN_PATH(graphics/opengl/2D/objects/letter.h)
    letter_t sb_shape_var_name;
    #undef sb_shape_var_name
    #endif
    #if defined(loco_text)
    #define sb_shape_var_name text
    #include _FAN_PATH(graphics/opengl/2D/objects/text.h)
    using text_t = text_renderer_t;
    text_t sb_shape_var_name;
    #undef sb_shape_var_name
    #endif
    #if defined(loco_responsive_text)
    #include _FAN_PATH(graphics/gui/responsive_text.h)
    responsive_text_t responsive_text;
    #endif
    #if defined(loco_button)
    #define sb_shape_var_name button
    #include _FAN_PATH(graphics/gui/button.h)
    button_t sb_shape_var_name;
    #undef sb_shape_var_name
    #endif
   
    #include _FAN_PATH(graphics/custom_shapes.h)

    #if defined(loco_rectangle_3d)
    #define sb_shape_var_name rectangle_3d
    #include _FAN_PATH(graphics/opengl/3D/objects/rectangle.h)
    rectangle_3d_t sb_shape_var_name;
    #undef sb_shape_var_name
    #endif
  };

  fan::mp_t<shapes_t> shapes;

  #if defined(loco_letter)
  #include _FAN_PATH(graphics/font.h)
  font_t font;
  #endif

  static constexpr uint8_t pixel_data[] = {
    1, 0, 0, 1,
    1, 0, 0, 1
  };

  #if defined(loco_custom_id_t_types)
  using custom_id_t_types_t = std::tuple<loco_custom_id_t_types>;
  #endif

  loco_t() : loco_t(properties_t()){

  }

  loco_t(properties_t p)
    #ifdef loco_window
    :
  gloco_dummy(this),
    window(p.window_size, fan::window_t::default_window_name, p.window_flags),
    #endif
    #if defined(loco_context)
    context(
      #if defined(loco_window)
      &window
      #endif
    )
    #endif
    #if defined(loco_window)
    , unloaded_image(fan::webp::image_info_t{ (void*)pixel_data, 1 })
    #endif
  {
    m_time.start();
    #if defined(loco_window)

    #if defined(loco_opengl)
    initialize_fb_vaos(fb_vao, fb_vbo);
    initialize_fb_vaos(post_fb_vao, post_fb_vbo);
    #endif

    root = loco_bdbt_NewNode(&bdbt);

    // set_vsync(p.vsync);
    #if defined(loco_vfi)
    window.add_buttons_callback([this](const mouse_buttons_cb_data_t& d) {
      fan::vec2 window_size = window.get_size();
      feed_mouse_button(d.button, d.state, get_mouse_position());
      });

    window.add_keys_callback([&](const keyboard_keys_cb_data_t& d) {
      feed_keyboard(d.key, d.state);
      });

    window.add_mouse_move_callback([&](const mouse_move_cb_data_t& d) {
      feed_mouse_move(get_mouse_position());
      });

    window.add_text_callback([&](const fan::window_t::text_cb_data_t& d) {
      feed_text(d.character);
      });
    #endif
    #endif
  #if fan_verbose_print_level >= 1
    #if defined(loco_opengl)
    fan::print("RENDERER BACKEND: OPENGL");
    #elif defined(loco_vulkan)
    fan::print("RENDERER BACKEND: VULKAN");
    #endif
  #endif

    #if defined(loco_letter)
    font.open(loco_font);
    #endif

    #if defined(loco_opengl)
    loco_t::image_t::load_properties_t lp;
    lp.visual_output = fan::opengl::GL_CLAMP_TO_EDGE;
    #if defined(loco_framebuffer)
    m_framebuffer.open(get_context());
    m_framebuffer.bind(get_context());
    #endif
    #endif

    #if defined(loco_opengl)

    #if defined(loco_framebuffer)

    static auto load_texture = [&](fan::webp::image_info_t& image_info, auto& color_buffer, fan::opengl::GLenum attachment) {
      loco_t::image_t::load_properties_t load_properties;
      load_properties.internal_format = fan::opengl::GL_RGBA;
      load_properties.format = fan::opengl::GL_RGBA;
      load_properties.type = fan::opengl::GL_FLOAT;
      load_properties.min_filter = fan::opengl::GL_LINEAR_MIPMAP_LINEAR;
      load_properties.mag_filter = fan::opengl::GL_LINEAR_MIPMAP_LINEAR;

      color_buffer.load(image_info, load_properties);
      get_context().opengl.call(get_context().opengl.glGenerateMipmap, fan::opengl::GL_TEXTURE_2D);

      color_buffer.bind_texture();
      fan::opengl::core::framebuffer_t::bind_to_texture(get_context(), color_buffer.get_texture(), attachment);
    };

    fan::webp::image_info_t image_info;
    image_info.data = nullptr;
    image_info.size = window.get_size();

    load_texture(image_info, color_buffers[0], fan::opengl::GL_COLOR_ATTACHMENT0);
    load_texture(image_info, color_buffers[1], fan::opengl::GL_COLOR_ATTACHMENT1);

    window.add_resize_callback([&](const auto& d) {
    fan::webp::image_info_t image_info;
    image_info.data = nullptr;
    image_info.size = window.get_size();

    load_texture(image_info, color_buffers[0], fan::opengl::GL_COLOR_ATTACHMENT0);
    load_texture(image_info, color_buffers[1], fan::opengl::GL_COLOR_ATTACHMENT1);

    fan::opengl::core::renderbuffer_t::properties_t renderbuffer_properties;
    m_framebuffer.bind(get_context());
    renderbuffer_properties.size = image_info.size;
    renderbuffer_properties.internalformat = fan::opengl::GL_DEPTH_COMPONENT;
    m_rbo.set_storage(get_context(), renderbuffer_properties);

    fan::vec2 window_size = gloco->window.get_size();
    default_camera->viewport.set(fan::vec2(0, 0), d.size, d.size);
  });

  fan::opengl::core::renderbuffer_t::properties_t renderbuffer_properties;
  m_framebuffer.bind(get_context());
  renderbuffer_properties.size = image_info.size;
  renderbuffer_properties.internalformat = fan::opengl::GL_DEPTH_COMPONENT;
  m_rbo.open(get_context());
  m_rbo.set_storage(get_context(), renderbuffer_properties);
  renderbuffer_properties.internalformat = fan::opengl::GL_DEPTH_ATTACHMENT;
  m_rbo.bind_to_renderbuffer(get_context(), renderbuffer_properties);

  unsigned int attachments[sizeof(color_buffers) / sizeof(color_buffers[0])];

  for (uint8_t i = 0; i < std::size(color_buffers); ++i) {
    attachments[i] = fan::opengl::GL_COLOR_ATTACHMENT0 + i;
  }

  get_context().opengl.call(get_context().opengl.glDrawBuffers, std::size(attachments), attachments);
  // finally check if framebuffer is complete
  if (!m_framebuffer.ready(get_context())) {
    fan::throw_error("framebuffer not ready");
  }

  m_framebuffer.unbind(gloco->get_context());

  m_fbo_final_shader.open();

  m_fbo_final_shader.set_vertex(
    loco_t::read_shader(_FAN_PATH_QUOTE(graphics/glsl/opengl/2D/effects/loco_fbo.vs))
  );
  m_fbo_final_shader.set_fragment(
    loco_t::read_shader(_FAN_PATH_QUOTE(graphics/glsl/opengl/2D/effects/loco_fbo.fs))
  );
  m_fbo_final_shader.compile();


  m_fbo_post_gui_shader.open();


  m_fbo_post_gui_shader.set_vertex(
    loco_t::read_shader(_FAN_PATH_QUOTE(graphics/glsl/opengl/2D/effects/loco_fbo.vs))
  );
  m_fbo_post_gui_shader.set_fragment(
    loco_t::read_shader(_FAN_PATH_QUOTE(graphics/glsl/opengl/2D/effects/loco_post_fbo.fs))
  );
  m_fbo_post_gui_shader.compile();

    #endif
    #endif




  #if defined(loco_vulkan) && defined(loco_window)
  fan::vulkan::pipeline_t::properties_t pipeline_p;

  render_fullscreen_shader.open(context, &m_write_queue);
  render_fullscreen_shader.set_vertex(
    context,
    "graphics/glsl/vulkan/2D/objects/loco_fbo.vert",
    #include _FAN_PATH(graphics/glsl/vulkan/2D/objects/loco_fbo.vert))
    );
    render_fullscreen_shader.set_fragment(
      context,
      "graphics/glsl/vulkan/2D/objects/loco_fbo.frag",
      #include _FAN_PATH(graphics/glsl/vulkan/2D/objects/loco_fbo.frag))
      );
      // NOTE order of the layouts (descriptor binds) depends about draw order of shape specific
      auto layouts = std::to_array({
      #if defined(loco_line)
        gloco->shapes.line.m_ssbo.m_descriptor.m_layout,
      #endif
      #if defined(loco_rectangle)
        gloco->shapes.rectangle.m_ssbo.m_descriptor.m_layout,
      #endif
      #if defined(loco_sprite)
        gloco->shapes.sprite.m_ssbo.m_descriptor.m_layout,
      #endif
      #if defined(loco_letter)
        gloco->shapes.letter.m_ssbo.m_descriptor.m_layout,
      #endif
      #if defined(loco_button)
        gloco->shapes.button.m_ssbo.m_descriptor.m_layout,
      #endif
      #if defined(loco_text_box)
        gloco->shapes.text_box.m_ssbo.m_descriptor.m_layout,
      #endif
      #if defined(loco_yuv420p)
        gloco->shapes.yuv420p.m_ssbo.m_descriptor.m_layout,
      #endif
      });
      // NOTE THIS
      std::reverse(layouts.begin(), layouts.end());
      pipeline_p.descriptor_layout_count = layouts.size();
      pipeline_p.descriptor_layout = layouts.data();
      pipeline_p.shader = &render_fullscreen_shader;
      pipeline_p.push_constants_size = sizeof(loco_t::push_constants_t);
      pipeline_p.subpass = 1;
      VkDescriptorImageInfo imageInfo{};

      VkPipelineColorBlendAttachmentState color_blend_attachment[1]{};
      color_blend_attachment[0].colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT |
        VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT |
        VK_COLOR_COMPONENT_A_BIT
        ;
      color_blend_attachment[0].blendEnable = VK_TRUE;
      color_blend_attachment[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
      color_blend_attachment[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      color_blend_attachment[0].colorBlendOp = VK_BLEND_OP_ADD;
      color_blend_attachment[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
      color_blend_attachment[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      color_blend_attachment[0].alphaBlendOp = VK_BLEND_OP_ADD;
      pipeline_p.color_blend_attachment_count = std::size(color_blend_attachment);
      pipeline_p.color_blend_attachment = color_blend_attachment;
      pipeline_p.enable_depth_test = false;
      context.render_fullscreen_pl.open(context, pipeline_p);
      #endif

  #if defined(loco_opengl)
    default_texture.create_missing_texture();
    transparent_texture.create_transparent_texture();
    #endif

    fan::vec2 window_size = window.get_size();

    default_camera = add_camera(fan::graphics::direction_e::right);

    {

      default_camera_3d = new camera_impl_t;

      fan::vec2 window_size = gloco->window.get_size();
      default_camera_3d->viewport = default_camera->viewport;
      static constexpr f32_t fov = 90.f;
      gloco->open_camera(&default_camera_3d->camera, fov);
    }
    open_camera(&default_camera->camera,
      fan::vec2(0, window_size.x),
      fan::vec2(0, window_size.y)
    );


  #if defined(loco_physics)
    fan::graphics::open_bcol();
  #endif

    #if defined(loco_imgui)

    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable | ImGuiConfigFlags_ViewportsEnable;

    static auto update_key_modifiers = [&] {
      io.AddKeyEvent(ImGuiMod_Ctrl, window.key_pressed(fan::key_left_control) || window.key_pressed(fan::key_right_control));
      io.AddKeyEvent(ImGuiMod_Shift, window.key_pressed(fan::key_left_shift) || window.key_pressed(fan::key_right_shift));
      io.AddKeyEvent(ImGuiMod_Alt, window.key_pressed(fan::key_left_alt) || window.key_pressed(fan::key_right_alt));
      io.AddKeyEvent(ImGuiMod_Super, window.key_pressed(fan::key_left_super) || window.key_pressed(fan::key_right_super));
    };

    window.add_buttons_callback([&](const auto& d) {

      update_key_modifiers();

      if (d.button != fan::mouse_scroll_up && d.button != fan::mouse_scroll_down) {
        io.AddMouseButtonEvent(d.button - fan::mouse_left, (bool)d.state);
      }
      else {
        if (d.button == fan::mouse_scroll_up) {
          io.AddMouseWheelEvent(0, 1);
        }
        else if (d.button == fan::mouse_scroll_down) {
          io.AddMouseWheelEvent(0, -1);
        }
      }
    });
    window.add_keys_callback([&](const auto& d) {
      update_key_modifiers();
      ImGuiKey imgui_key = fan::window_input::fan_to_imguikey(d.key);
      io.AddKeyEvent(imgui_key, (int)d.state);
    });
    window.add_text_callback([&](const auto& d) {
      io.AddInputCharacter(d.character);
    });

    static bool init = false;
    if (init == false) {
      init = true;

      #if defined(loco_imgui)
      loco_t::imgui_themes::dark();

      #if defined(fan_platform_windows)
      ImGui_ImplWin32_Init(window.get_handle());
      #elif defined(fan_platform_linux)
      imgui_xorg_init();
      #endif
      ImGui_ImplOpenGL3_Init();

      auto& style = ImGui::GetStyle();
      auto& io = ImGui::GetIO();

      static constexpr const char* font_name = "fonts/SourceCodePro-Regular.ttf";
      static constexpr f32_t font_size = 36;
      if (io.Fonts->AddFontFromFileTTF(font_name, font_size) == nullptr) {
        fan::throw_error(fan::string("failed to load font") + font_name);
      }
      io.Fonts->Build();
    }
    #endif
    #endif
  }

  #if defined(loco_vfi)
  void push_back_input_hitbox(loco_t::shapes_t::vfi_t::shape_id_t& id, const loco_t::shapes_t::vfi_t::properties_t& p) {
    shapes.vfi.push_back(id, p);
  }
  #endif
  /* uint32_t push_back_keyboard_event(uint32_t depth, const fan_2d::graphics::gui::ke_t::properties_t& p) {
     return element_depth[depth].keyboard_event.push_back(p);
   }*/

  #if defined(loco_vfi)
  void feed_mouse_move(const fan::vec2& mouse_position) {
    shapes.vfi.feed_mouse_move(mouse_position);
  }

  void feed_mouse_button(uint16_t button, fan::mouse_state button_state, const fan::vec2& mouse_position) {
    shapes.vfi.feed_mouse_button(button, button_state);
  }

  void feed_keyboard(uint16_t key, fan::keyboard_state keyboard_state) {
    shapes.vfi.feed_keyboard(key, keyboard_state);
  }

  void feed_text(uint32_t key) {
    shapes.vfi.feed_text(key);
  }
  #endif

  void process_frame() {

    #if defined(loco_opengl)
    #if defined(loco_framebuffer)
    get_context().opengl.glActiveTexture(fan::opengl::GL_TEXTURE0);
    color_buffers[0].bind_texture();

    get_context().opengl.glActiveTexture(fan::opengl::GL_TEXTURE1);
    color_buffers[1].bind_texture();


    #endif
    #endif

    #if defined(loco_opengl)
    #if defined(loco_framebuffer)
    m_framebuffer.bind(get_context());
    //float clearColor[] = { 0.0f, 0.0f, 0.0f, 1.0f };
    //auto buffers = fan::opengl::GL_COLOR_ATTACHMENT0 + 2;
    //get_context().opengl.glClearBufferfv(fan::opengl::GL_COLOR, 0, clearColor);
    //get_context().opengl.glClearBufferfv(fan::opengl::GL_COLOR, 1, clearColor);
    //get_context().opengl.glClearBufferfv(fan::opengl::GL_COLOR, 2, clearColor);
    get_context().opengl.glDrawBuffer(fan::opengl::GL_COLOR_ATTACHMENT1);
    get_context().opengl.glClearColor(0, 0, 0, 1);
    get_context().opengl.glClear(fan::opengl::GL_COLOR_BUFFER_BIT);
    get_context().opengl.glDrawBuffer(fan::opengl::GL_COLOR_ATTACHMENT0);
    #endif
    get_context().opengl.glClearColor(clear_color.r, clear_color.g, clear_color.b, clear_color.a);
    get_context().opengl.call(get_context().opengl.glClear, fan::opengl::GL_COLOR_BUFFER_BIT | fan::opengl::GL_DEPTH_BUFFER_BIT);
    #endif

    auto it = m_update_callback.GetNodeFirst();
    while (it != m_update_callback.dst) {
      m_update_callback.StartSafeNext(it);
      m_update_callback[it](this);
      it = m_update_callback.EndSafeNext();
    }

    m_write_queue.process(get_context());

    #ifdef loco_window
    #if defined(loco_opengl)

    #include "draw_shapes.h"

    #if defined(loco_framebuffer)

    m_framebuffer.unbind(get_context());

    get_context().opengl.glClearColor(0, 0, 0, 1);
    get_context().opengl.call(get_context().opengl.glClear, fan::opengl::GL_COLOR_BUFFER_BIT | fan::opengl::GL_DEPTH_BUFFER_BIT);
    fan::vec2 window_size = window.get_size();
    fan::opengl::viewport_t::set_viewport(0, window_size, window_size);

    m_fbo_final_shader.use();
    m_fbo_final_shader.set_int("_t00", 0);
    m_fbo_final_shader.set_int("_t01", 1);

    get_context().opengl.glActiveTexture(fan::opengl::GL_TEXTURE0);
    color_buffers[0].bind_texture();

    get_context().opengl.glActiveTexture(fan::opengl::GL_TEXTURE1);
    color_buffers[1].bind_texture();

    render_final_fb();

    #endif
    #if defined(loco_imgui)

    {
      auto it = m_imgui_draw_cb.GetNodeFirst();
      while (it != m_imgui_draw_cb.dst) {
        m_imgui_draw_cb.StartSafeNext(it);
        m_imgui_draw_cb[it]();
        it = m_imgui_draw_cb.EndSafeNext();
      }
    }

    #if defined(loco_framebuffer)
    m_framebuffer.bind(get_context());
    #endif

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    #if defined(loco_framebuffer)
    m_framebuffer.unbind(get_context());

    get_context().opengl.glClearColor(0, 0, 0, 1);
    get_context().opengl.call(get_context().opengl.glClear, fan::opengl::GL_COLOR_BUFFER_BIT | fan::opengl::GL_DEPTH_BUFFER_BIT);
    fan::opengl::viewport_t::set_viewport(0, window_size, window_size);

    m_fbo_post_gui_shader.use();
    m_fbo_post_gui_shader.set_int("_t00", 0);
    m_fbo_post_gui_shader.set_int("_t01", 1);
    m_fbo_post_gui_shader.set_float("m_time", gloco->m_time.elapsed() / 1e+9);

    get_context().opengl.glActiveTexture(fan::opengl::GL_TEXTURE0);
    color_buffers[0].bind_texture();

    get_context().opengl.glActiveTexture(fan::opengl::GL_TEXTURE1);
    color_buffers[1].bind_texture();

    render_post_fb();

    #endif

    #endif
    get_context().render(get_window());

    #elif defined(loco_vulkan)
    get_context().begin_render(get_window());

    #include "draw_shapes.h"

    get_context().end_render(get_window());
    #endif
    #endif
  }
  #if defined(loco_window)
  bool window_open(uint32_t event) {
    return event != fan::window_t::events::close;
  }
  uint32_t get_fps() {
    return window.get_fps();
  }

  void set_vsync(bool flag) {
    get_context().set_vsync(get_window(), flag);
  }

  fan::vec2 transform_matrix(const fan::vec2& position) {
    fan::vec2 window_size = window.get_size();
    // not custom ortho friendly - made for -1 1
    return position / window_size * 2 - 1;
  }

  fan::vec2 screen_to_ndc(const fan::vec2& screen_pos) {
    fan::vec2 window_size = window.get_size();
    return screen_pos / window_size * 2 - 1;
  }

  fan::vec2 ndc_to_screen(const fan::vec2& ndc_position) {
    fan::vec2 window_size = window.get_size();
    fan::vec2 normalized_position = (ndc_position + 1) / 2;
    return normalized_position * window_size;
  }

  //  behaving oddly
  fan::vec2 get_mouse_position(const loco_t::camera_t& camera, const loco_t::viewport_t& viewport) {
    fan::vec2 mouse_pos = window.get_mouse_position();
    fan::vec2 translated_pos;
    translated_pos.x = fan::math::map(mouse_pos.x, viewport.get_position().x, viewport.get_position().x + viewport.get_size().x, camera.coordinates.left, camera.coordinates.right);
    translated_pos.y = fan::math::map(mouse_pos.y, viewport.get_position().y, viewport.get_position().y + viewport.get_size().y, camera.coordinates.up, camera.coordinates.down);
    return translated_pos;
  }

  fan::vec2 get_mouse_position() {
    return window.get_mouse_position();
    //return get_mouse_position(gloco->default_camera->camera, gloco->default_camera->viewport); behaving oddly
  }

  static fan::vec2 translate_position(const fan::vec2 & p, fan::graphics::viewport_t * viewport, loco_t::camera_t * camera) {

    fan::vec2 viewport_position = viewport->get_position();
    fan::vec2 viewport_size = viewport->get_size();

    f32_t l = camera->coordinates.left;
    f32_t r = camera->coordinates.right;
    f32_t t = camera->coordinates.up;
    f32_t b = camera->coordinates.down;

    fan::vec2 tp = p - viewport_position;
    fan::vec2 d = viewport_size;
    tp /= d;
    tp = fan::vec2(r * tp.x - l * tp.x + l, b * tp.y - t * tp.y + t);
    return tp;
  }
  fan::vec2 translate_position(const fan::vec2& p) {
    return translate_position(p, &default_camera->viewport, &default_camera->camera);
  }

  #if defined(loco_vulkan)
  loco_t::shader_t render_fullscreen_shader;
  #endif


#if defined(loco_framebuffer)
  #if defined(loco_opengl)

    fan::opengl::core::framebuffer_t m_framebuffer;
    fan::opengl::core::renderbuffer_t m_rbo;
    loco_t::image_t color_buffers[2];
    loco_t::shader_t m_fbo_final_shader;
    loco_t::shader_t m_fbo_post_gui_shader;

  #endif
#endif

  bool process_loop(const auto& lambda = []{}) {

    // enables drawing while resizing, not required for x11
    #if defined(fan_platform_windows)
    auto it = window.add_resize_callback([this, &lambda](const auto& d) {
      gloco->process_loop(lambda);
    });
    #endif

    uint32_t window_event = window.handle_events();
    if (window_event & fan::window_t::events::close) {
      window.destroy_window();
      return 1;
    }

    #if defined(fan_platform_windows)
    window.remove_resize_callback(it);
    #endif


    #if defined(loco_imgui)
    ImGui_ImplOpenGL3_NewFrame();
    #if defined(fan_platform_windows)
    ImGui_ImplWin32_NewFrame();
    #elif defined(fan_platform_linux)
    imgui_xorg_new_frame();
    #endif
    ImGui::NewFrame();

    auto& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;
    const ImVec4 bgColor = ImVec4(0.0, 0.0, 0.0, 0.4);
    colors[ImGuiCol_WindowBg] = bgColor;
    colors[ImGuiCol_ChildBg] = bgColor;
    colors[ImGuiCol_TitleBg] = bgColor;

    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_DockingEmptyBg, ImVec4(0, 0, 0, 0));
    ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());
    ImGui::PopStyleColor(2);
    #endif

    lambda();

    ev_timer.process();
    process_frame();
    return 0;
  }

  void loop(const auto& lambda) {
    while (1) {
      if (process_loop(lambda)) {
        break;
      }
    }
  }

  static loco_t* get_loco(fan::window_t* window) {
    return OFFSETLESS(window, loco_t, window);
  }
  #endif


  #if defined(loco_cuda)

  struct cuda_textures_t {

    cuda_textures_t() {
      inited = false;
    }
    ~cuda_textures_t() {
    }
    void close(loco_t* loco, loco_t::shape_t& cid) {
      uint8_t image_amount = fan::pixel_format::get_texture_amount(loco->shapes.pixel_format_renderer.sb_get_ri(cid).format);
      auto& ri = loco->shapes.pixel_format_renderer.sb_get_ri(cid);
      for (uint32_t i = 0; i < image_amount; ++i) {
        wresources[i].close();
        ri.images[i].unload();
      }
    }

    void resize(loco_t* loco, loco_t::cid_nt_t& id, uint8_t format, fan::vec2ui size, uint32_t filter = loco_t::image_t::filter::linear) {
      auto& ri = loco->shapes.pixel_format_renderer.sb_get_ri(id);
      uint8_t image_amount = fan::pixel_format::get_texture_amount(format);
      if (inited == false) {
        // purge cid's images here
        // update cids images
        loco->shapes.pixel_format_renderer.reload(id, format, size, filter);
        for (uint32_t i = 0; i < image_amount; ++i) {
          wresources[i].open(loco->image_list[ri.images[i].texture_reference].texture_id);
        }
        inited = true;
      }
      else {

        if (ri.images[0].size == size) {
          return;
        }

        // update cids images
        for (uint32_t i = 0; i < fan::pixel_format::get_texture_amount(ri.format); ++i) {
          wresources[i].close();
        }

        loco->shapes.pixel_format_renderer.reload(id, format, size, filter);

        for (uint32_t i = 0; i < image_amount; ++i) {
          wresources[i].open(loco->image_list[ri.images[i].texture_reference].texture_id);
        }
      }
    }

    cudaArray_t& get_array(uint32_t index) {
      return wresources[index].cuda_array;
    }

    struct graphics_resource_t {
      void open(int texture_id) {
        fan::cuda::check_error(cudaGraphicsGLRegisterImage(&resource, texture_id, fan::opengl::GL_TEXTURE_2D, cudaGraphicsMapFlagsNone));
        map();
      }
      void close() {
        unmap();
        fan::cuda::check_error(cudaGraphicsUnregisterResource(resource));
        resource = nullptr;
      }
      void map() {
        fan::cuda::check_error(cudaGraphicsMapResources(1, &resource, 0));
        fan::cuda::check_error(cudaGraphicsSubResourceGetMappedArray(&cuda_array, resource, 0, 0));
        fan::print("+", resource);
      }
      void unmap() {
        fan::print("-", resource);
        fan::cuda::check_error(cudaGraphicsUnmapResources(1, &resource));
        //fan::cuda::check_error(cudaGraphicsResourceSetMapFlags(resource, 0));
      }
      //void reload(int texture_id) {
      //  close();
      //  open(texture_id);
      //}
      cudaGraphicsResource_t resource = nullptr;
      cudaArray_t cuda_array = nullptr;
    };

    bool inited = false;
    graphics_resource_t wresources[4];
  };

  #endif


protected:
  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_SafeNext 1
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_prefix update_callback
  #include _FAN_PATH(fan_bll_preset.h)
  #define BLL_set_Link 1
  #define BLL_set_type_node uint16_t
  #define BLL_set_NodeDataType fan::function_t<void(loco_t*)>
  #include _FAN_PATH(BLL/BLL.h)
public:

  using update_callback_nr_t = update_callback_NodeReference_t;

  update_callback_t m_update_callback;

  image_t default_texture;
  image_t transparent_texture;

  #if defined(loco_vfi)
  // used in fl
  struct vfi_id_t {
    using properties_t = loco_t::shapes_t::vfi_t::properties_t;
    operator loco_t::shapes_t::vfi_t::shape_id_t* () {
      return &cid;
    }
    vfi_id_t() = default;
    vfi_id_t(const properties_t& p) {
      gloco->shapes.vfi.push_back(cid, *(properties_t*)&p);
    }
    vfi_id_t& operator[](const properties_t& p) {
      gloco->shapes.vfi.push_back(cid, *(properties_t*)&p);
      return *this;
    }
    ~vfi_id_t() {
      gloco->shapes.vfi.erase(cid);
    }

    loco_t::shapes_t::vfi_t::shape_id_t cid;
  };

  #endif

  template <typename T>
  void push_shape(loco_t::cid_nt_t& id, T properties) {
    shapes.iterate([&]<auto i>(auto & shape) {
      fan_if_has_function(&shape, push_back, (id, properties));
    });
  }

  void shape_get_properties(loco_t::cid_nt_t& id, auto lambda) {
    shapes.iterate([&]<auto i, typename T>(T& shape) {
      if (shape.shape_type != (loco_t::shape_type_t)id->shape_type) {
        return;
      }
      typename T::properties_t properties;
      fan_if_has_function_get(&shape, get_properties, (id), properties);
      else fan_if_has_function_get(&shape, sb_get_properties, (id), properties);
      lambda(properties);
    });
  }

  void shape_draw(
    loco_t::shape_type_t shape_type,
    const loco_t::redraw_key_t& redraw_key,
    loco_bdbt_NodeReference_t nr
  ) {
    shapes.iterate([&]<auto i>(auto& shape) {
      if (shape_type != shape.shape_type) {
        return;
      }

      fan_if_has_function(&shape, draw, (redraw_key, nr));
    });
  }

  void shape_erase(loco_t::cid_nt_t& id) {
    bool erased = false;
    shapes.iterate([&]<auto i>(auto & shape) {
      if (erased) {
        return;
      }
      if (shape.shape_type != (loco_t::shape_type_t)id->shape_type) {
        return;
      }
      fan_if_has_function(&shape, erase, (id));
      erased = true;
    });
  }

  #define fan_build_set_shape_property(property_type, property_name) \
  void shape_set_##property_name(loco_t::cid_nt_t& id, property_type value) { \
    shapes.iterate([&]<auto i, typename T>(T& shape) { \
      if (shape.shape_type != (loco_t::shape_type_t)id->shape_type) { \
          return; \
      } \
      fan_if_has_function(&shape, set_##property_name, (id, value)); \
      else if constexpr(fan_requires_rule(T, typename T::vi_t)){ \
        if constexpr(fan_has_variable(typename T::vi_t, property_name)) { \
          fan_if_has_function(&shape, set, (id, &T::vi_t::property_name, value)); \
        } \
      }\
    }); \
  }

  #define fan_build_get_shape_property(property_type, property_name) \
  std::remove_const_t<std::remove_reference_t<property_type>> shape_get_##property_name(loco_t::cid_nt_t& id) { \
    std::remove_const_t<std::remove_reference_t<property_type>> ret; \
    shapes.iterate([&]<auto i, typename T>(T& shape) { \
      if (shape.shape_type != (loco_t::shape_type_t)id->shape_type) { \
          return; \
      } \
      fan_if_has_function_get(&shape, get_##property_name, (id), ret); \
      else if constexpr(fan_requires_rule(T, typename T::vi_t)){ \
        if constexpr(fan_has_variable(typename T::vi_t, property_name)) { \
          fan_if_has_function_get(&shape, get, (id, &T::vi_t::property_name), ret); \
        } \
      } \
    }); \
    return ret; \
  }


  #define fan_build_get_set_shape_property(property_type, property_name) \
    fan_build_get_shape_property(property_type, property_name); \
    fan_build_set_shape_property(property_type, property_name);
  

  fan_build_get_set_shape_property(const fan::vec3&, position);
  fan_build_get_set_shape_property(const fan::vec3&, size);
  fan_build_get_set_shape_property(const fan::color&, color);
  fan_build_get_set_shape_property(const fan::vec3&, angle);
  fan_build_get_set_shape_property(const fan::string&, text);
  fan_build_get_set_shape_property(const fan::vec2&, rotation_point);
  fan_build_get_set_shape_property(f32_t, font_size);
  fan_build_get_set_shape_property(const fan::vec2&, text_size);

  fan_build_get_set_shape_property(const fan::vec2&, tc_position);
  fan_build_get_set_shape_property(const fan::vec2&, tc_size);

  fan_build_get_set_shape_property(fan::color, outline_color);
  fan_build_get_set_shape_property(f32_t, outline_size);

  fan_build_get_set_shape_property(f32_t, depth);

  fan_build_get_set_shape_property(loco_t::viewport_t*, viewport);
  fan_build_get_set_shape_property(loco_t::camera_t*, camera);
  fan_build_get_set_shape_property(loco_t::image_t*, image);

  /*bool shape_append_letter(
    loco_t::cid_nt_t& id,
      wchar_t wc,
      bool force = false
  ) {
    bool ret = 0;
    shapes.iterate([&]<auto i, typename T>(T& shape) { 
      if (shape.shape_type != (loco_t::shape_type_t)id->shape_type) { 
          return; 
      } 
      fan_if_has_function_get(&shape, append_letter, (id), ret);
    }); 
    return ret;
  }*/

  void shape_set_line(
  loco_t::cid_nt_t& id,
    const fan::vec3& src,
    fan::vec2 dst
  ) {
    shapes.iterate([&]<auto i, typename T>(T & shape) {
      if (shape.shape_type != (loco_t::shape_type_t)id->shape_type) {
        return;
      }
      fan_if_has_function(&shape, set_line, (id, src, dst));
    });
  }

  fan::vec2 get_camera_view_size(loco_t::camera_t camera) {
    return fan::vec2(
      std::abs(camera.coordinates.right) + std::abs(camera.coordinates.left),
      std::abs(camera.coordinates.down) + std::abs(camera.coordinates.up)
    );
  }

  static inline std::vector<camera_impl_t*> viewport_handler;
  camera_impl_t* add_camera(fan::graphics::direction_e split_direction) {
    viewport_handler.push_back(new camera_impl_t(split_direction));
    int index = 0;
    fan::vec2 window_size = gloco->window.get_size();
    gloco->viewport_divider.iterate([&index, window_size](auto& node) {
      viewport_handler[index]->viewport.set(
        (node.position - node.size / 2) * window_size,
        ((node.size) * window_size), window_size
      );
      index++;
    });
    return viewport_handler.back();
  }

  #if defined(loco_button)
  loco_t::theme_t default_theme = loco_t::themes::gray();
  #endif
  camera_impl_t* default_camera;
  camera_impl_t* default_camera_3d;

  fan::graphics::viewport_divider_t viewport_divider;

  #undef make_global_function
  #undef fan_build_get
  #undef fan_build_set

  fan::time::clock m_time;

  using image_info_t = fan::webp::image_info_t;

  int begin = 0;

  #if defined(loco_sprite)

  fan::string get_sprite_vertex_shader() {
    return 
      #if defined(loco_opengl)
      loco_t::read_shader(_FAN_PATH_QUOTE(graphics/glsl/opengl/2D/objects/sprite.vs))
      #else
      "";
    ;
      #endif
      ;
  }

  loco_t::shader_t create_sprite_shader(const fan::string& fragment) {
    loco_t::shader_t shader;
    #if defined(loco_opengl)
    shader.open();
    shader.set_vertex(
      get_sprite_vertex_shader()
    );
    shader.set_fragment(fragment);
    shader.compile();
    #else
    assert(0);
    #endif
    return shader;
  }

  #endif
  #if defined(loco_light)
  loco_t::shader_t create_light_shader(const fan::string& fragment) {
    loco_t::shader_t shader;
    shader.open();
    shader.set_vertex(
      loco_t::read_shader(_FAN_PATH_QUOTE(graphics/glsl/opengl/2D/objects/light.vs))
    );
    shader.set_fragment(fragment);
    shader.compile();
    return shader;
  }

  #endif

  static fan::vec2 convert_mouse_to_ndc(const fan::vec2& mouse_position, const fan::vec2i& window_size) {
    return fan::vec2((2.0f * mouse_position.x) / window_size.x - 1.0f, 1.0f - (2.0f * mouse_position.y) / window_size.y);
  }
  fan::vec2 convert_mouse_to_ndc(const fan::vec2& mouse_position) const {
    return convert_mouse_to_ndc(mouse_position, gloco->window.get_size());
  }
  fan::vec2 convert_mouse_to_ndc() const {
    return convert_mouse_to_ndc(gloco->get_mouse_position(), gloco->window.get_size());
  }

  static fan::ray3_t convert_mouse_to_ray(const fan::vec2i& mouse_position, const fan::vec3& camera_position, const fan::mat4& projection, const fan::mat4& view) {
    fan::vec2i screen_size = gloco->window.get_size();

    fan::vec4 ray_ndc((2.0f * mouse_position.x) / screen_size.x - 1.0f, 1.0f - (2.0f * mouse_position.y) / screen_size.y, 1.0f, 1.0f);

    fan::mat4 inverted_projection = projection.inverse();

    fan::vec4 ray_clip = inverted_projection * ray_ndc;

    ray_clip.z = -1.0f;
    ray_clip.w = 0.0f;

    fan::mat4 inverted_view = view.inverse();

    fan::vec4 ray_world = inverted_view * ray_clip;

    fan::vec3 ray_dir = fan::vec3(ray_world.x, ray_world.y, ray_world.z).normalize();

    fan::vec3 ray_origin = camera_position;
    return fan::ray3_t(ray_origin, ray_dir);
  }

  fan::ray3_t convert_mouse_to_ray(const fan::vec3& camera_position, const fan::mat4& projection, const fan::mat4& view) {
    return convert_mouse_to_ray(gloco->get_mouse_position(), camera_position, projection, view);
  }
  fan::ray3_t convert_mouse_to_ray(const fan::mat4& projection, const fan::mat4& view) {
    return convert_mouse_to_ray(gloco->get_mouse_position(), default_camera_3d->camera.position, projection, view);
  }

  static bool is_ray_intersecting_cube(const fan::ray3_t& ray, const fan::vec3& position, const fan::vec3& size) {
    fan::vec3 min_bounds = position - size;
    fan::vec3 max_bounds = position + size;

    fan::vec3 t_min = (min_bounds - ray.origin) / (ray.direction + fan::vec3(1e-6f));
    fan::vec3 t_max = (max_bounds - ray.origin) / (ray.direction + fan::vec3(1e-6f));

    fan::vec3 t1 = t_min.min(t_max);
    fan::vec3 t2 = t_min.max(t_max);

    float t_near = fan::max(t1.x, fan::max(t1.y, t1.z));
    float t_far = fan::min(t2.x, fan::min(t2.y, t2.z));

    return t_near <= t_far && t_far >= 0.0f;
  }
  #if defined(loco_compute_shader)
  #include _FAN_PATH(graphics/vulkan/compute_shader.h)
  #endif
};

#if defined(loco_pixel_format_renderer)
template <typename T>
constexpr std::array<T, 4> fan::pixel_format::get_image_properties(uint8_t format) {
  switch (format) {
    case yuv420p: {
      return std::array<loco_t::image_t::load_properties_t, 4>{
        loco_t::image_t::load_properties_t{
          .internal_format = loco_t::image_t::format::r8_unorm,
          .format = loco_t::image_t::format::r8_unorm
        },
          loco_t::image_t::load_properties_t{
            .internal_format = loco_t::image_t::format::r8_unorm,
            .format = loco_t::image_t::format::r8_unorm
        },
          loco_t::image_t::load_properties_t{
            .internal_format = loco_t::image_t::format::r8_unorm,
            .format = loco_t::image_t::format::r8_unorm
        }
      };
    }
    case nv12: {
      return std::array<loco_t::image_t::load_properties_t, 4>{
        loco_t::image_t::load_properties_t{
          .internal_format = loco_t::image_t::format::r8_unorm,
          .format = loco_t::image_t::format::r8_unorm
        },
          loco_t::image_t::load_properties_t{
            .internal_format = loco_t::image_t::format::rg8_unorm,
            .format = loco_t::image_t::format::rg8_unorm
        }
      };
    }
    default: {
      fan::throw_error("invalid format");
      return std::array<loco_t::image_t::load_properties_t, 4>{};
    }
  }
}

#endif

namespace fan {
  namespace graphics {

    using camera_t = loco_t::camera_impl_t;
    // use bll to avoid 'new'
    static auto add_camera(fan::graphics::direction_e split_direction) {
      return gloco->add_camera(split_direction);
    }

    #include _FAN_PATH(graphics/common_types.h)
  }
}

#if defined(loco_opengl)

inline void fan::opengl::viewport_t::open() {
  viewport_reference = gloco->get_context().viewport_list.NewNode();
  gloco->get_context().viewport_list[viewport_reference].viewport_id = this;
}

inline void fan::opengl::viewport_t::close() {
  gloco->get_context().viewport_list.Recycle(viewport_reference);
}

inline void fan::opengl::viewport_t::set_viewport(const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
  gloco->get_context().opengl.call(
    gloco->get_context().opengl.glViewport,
    viewport_position_.x,
    window_size.y - viewport_size_.y - viewport_position_.y,
    viewport_size_.x, viewport_size_.y
  );
}

inline void fan::opengl::viewport_t::set(const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
  viewport_position = viewport_position_;
  viewport_size = viewport_size_;

  gloco->get_context().opengl.call(
    gloco->get_context().opengl.glViewport,
    viewport_position.x, window_size.y - viewport_size.y - viewport_position.y,
    viewport_size.x, viewport_size.y
  );
}

#elif defined(loco_vulkan)
inline void fan::vulkan::viewport_t::open() {
  auto& context = gloco->get_context();
  viewport_reference = context.viewport_list.NewNode();
  context.viewport_list[viewport_reference].viewport_id = this;
}

inline void fan::vulkan::viewport_t::close() {
  auto& context = gloco->get_context();
  context.viewport_list.Recycle(viewport_reference);
}

inline void fan::vulkan::viewport_t::set(const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
  viewport_position = viewport_position_;
  viewport_size = viewport_size_;

  VkViewport viewport{};
  viewport.x = viewport_position.x;
  viewport.y = viewport_position.y;
  viewport.width = viewport_size.x;
  viewport.height = viewport_size.y;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

  auto& context = gloco->get_context();

  if (!context.command_buffer_in_use) {
    VkResult result = vkGetFenceStatus(context.device, context.inFlightFences[context.currentFrame]);
    if (result == VK_NOT_READY) {
      vkDeviceWaitIdle(context.device);
    }

    if (vkBeginCommandBuffer(context.commandBuffers[context.currentFrame], &beginInfo) != VK_SUCCESS) {
      fan::throw_error("failed to begin recording command buffer!");
    }
  }
  vkCmdSetViewport(context.commandBuffers[context.currentFrame], 0, 1, &viewport);

  if (!context.command_buffer_in_use) {
    if (vkEndCommandBuffer(context.commandBuffers[context.currentFrame]) != VK_SUCCESS) {
      fan::throw_error("failed to record command buffer!");
    }
    context.command_buffer_in_use = false;
  }
}

inline void fan::vulkan::viewport_t::set_viewport(const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
  auto& context = gloco->get_context();
  VkViewport viewport{};
  viewport.x = viewport_position_.x;
  viewport.y = viewport_position_.y;
  viewport.width = viewport_size_.x;
  viewport.height = viewport_size_.y;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

  if (!context.command_buffer_in_use) {
    VkResult result = vkGetFenceStatus(context.device, context.inFlightFences[context.currentFrame]);
    if (result == VK_NOT_READY) {
      vkDeviceWaitIdle(context.device);
    }

    if (vkBeginCommandBuffer(context.commandBuffers[context.currentFrame], &beginInfo) != VK_SUCCESS) {
      fan::throw_error("failed to begin recording command buffer!");
    }
  }
  vkCmdSetViewport(context.commandBuffers[context.currentFrame], 0, 1, &viewport);

  if (!context.command_buffer_in_use) {
    if (vkEndCommandBuffer(context.commandBuffers[context.currentFrame]) != VK_SUCCESS) {
      fan::throw_error("failed to record command buffer!");
    }
    context.command_buffer_in_use = false;
  }
}
#endif



inline loco_t::image_list_NodeReference_t::image_list_NodeReference_t(loco_t::image_t* image) {
  NRI = image->texture_reference.NRI;
}

inline loco_t::camera_list_NodeReference_t::camera_list_NodeReference_t(loco_t::camera_t* camera) {
  NRI = camera->camera_reference.NRI;
}

inline loco_t::shader_list_NodeReference_t::shader_list_NodeReference_t(loco_t::shader_t* shader) {
  NRI = shader->shader_reference.NRI;
}

inline void fan::camera::move(f32_t movement_speed, f32_t friction) {
  this->velocity /= friction * gloco->window.get_delta_time() + 1;
  static constexpr auto minimum_velocity = 0.001;
  if (this->velocity.x < minimum_velocity && this->velocity.x > -minimum_velocity) {
    this->velocity.x = 0;
  }
  if (this->velocity.y < minimum_velocity && this->velocity.y > -minimum_velocity) {
    this->velocity.y = 0;
  }
  if (this->velocity.z < minimum_velocity && this->velocity.z > -minimum_velocity) {
    this->velocity.z = 0;
  }
  if (gloco->window.key_pressed(fan::input::key_w)) {
    this->velocity += this->m_front * (movement_speed * gloco->window.get_delta_time());
  }
  if (gloco->window.key_pressed(fan::input::key_s)) {
    this->velocity -= this->m_front * (movement_speed * gloco->window.get_delta_time());
  }
  if (gloco->window.key_pressed(fan::input::key_a)) {
    this->velocity -= this->m_right * (movement_speed * gloco->window.get_delta_time());
  }
  if (gloco->window.key_pressed(fan::input::key_d)) {
    this->velocity += this->m_right * (movement_speed * gloco->window.get_delta_time());
  }

  if (gloco->window.key_pressed(fan::input::key_space)) {
    this->velocity.y += movement_speed * gloco->window.get_delta_time();
  }
  if (gloco->window.key_pressed(fan::input::key_left_shift)) {
    this->velocity.y -= movement_speed * gloco->window.get_delta_time();
  }

  if (gloco->window.key_pressed(fan::input::key_left)) {
    this->set_yaw(this->get_yaw() - sensitivity * 100 * gloco->window.get_delta_time());
  }
  if (gloco->window.key_pressed(fan::input::key_right)) {
    this->set_yaw(this->get_yaw() + sensitivity * 100 * gloco->window.get_delta_time());
  }
  if (gloco->window.key_pressed(fan::input::key_up)) {
    this->set_pitch(this->get_pitch() + sensitivity * 100 * gloco->window.get_delta_time());
  }
  if (gloco->window.key_pressed(fan::input::key_down)) {
    this->set_pitch(this->get_pitch() - sensitivity * 100 * gloco->window.get_delta_time());
  }

  this->position += this->velocity * gloco->window.get_delta_time();
  this->update_view();
}

#if defined(loco_button)
namespace fan::opengl {
  // Primary template for the constructor
  inline theme_list_NodeReference_t::theme_list_NodeReference_t(void* theme) {
    //static_assert(std::is_same_v<decltype(theme), loco_t::theme_t*>, "invalid parameter passed to theme");
    NRI = ((loco_t::theme_t*)theme)->theme_reference.NRI;
  }
}
#endif

#if defined(loco_opengl)
inline fan::opengl::viewport_list_NodeReference_t::viewport_list_NodeReference_t(fan::opengl::viewport_t* viewport) {
  NRI = viewport->viewport_reference.NRI;
}
#elif defined(loco_vulkan)

inline void fan::vulkan::context_t::begin_render(fan::window_t* window) {
  vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

  VkResult result = vkAcquireNextImageKHR(
    device,
    swapChain,
    UINT64_MAX,
    imageAvailableSemaphores[currentFrame],
    VK_NULL_HANDLE,
    &image_index
  );

  if (result == VK_ERROR_OUT_OF_DATE_KHR) {
    recreateSwapChain(window->get_size());
    return;
  }
  else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
    fan::throw_error("failed to acquire swap chain image!");
  }

  vkResetFences(device, 1, &inFlightFences[currentFrame]);

  vkResetCommandBuffer(commandBuffers[currentFrame], /*VkCommandBufferResetFlagBits*/ 0);

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

  if (vkBeginCommandBuffer(commandBuffers[currentFrame], &beginInfo) != VK_SUCCESS) {
    fan::throw_error("failed to begin recording command buffer!");
  }

  command_buffer_in_use = true;

  VkRenderPassBeginInfo renderPassInfo{};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderPassInfo.renderPass = renderPass;
  renderPassInfo.framebuffer = swapChainFramebuffers[image_index];
  renderPassInfo.renderArea.offset = { 0, 0 };
  renderPassInfo.renderArea.extent.width = swap_chain_size.x;
  renderPassInfo.renderArea.extent.height = swap_chain_size.y;

  // TODO

  #if defined(loco_wboit)
  VkClearValue clearValues[4]{};
  clearValues[2].color = { { 0.0f, 0.0f, 0.0f, 0.0f} };
  clearValues[3].depthStencil = { 1.0f, 0 };

  clearValues[0].color.float32[0] = 0.0f;
  clearValues[0].color.float32[1] = 0.0f;
  clearValues[0].color.float32[2] = 0.0f;
  clearValues[0].color.float32[3] = 0.0f;
  clearValues[1].color.float32[0] = 1.f;  // Initially, all pixels show through all the way (reveal = 100%)

  #else
  VkClearValue clearValues[
    5
  ]{};
    clearValues[0].color = { { gloco->clear_color.r, gloco->clear_color.g, gloco->clear_color.b, gloco->clear_color.a} };
    clearValues[1].color = { {gloco->clear_color.r, gloco->clear_color.g, gloco->clear_color.b, gloco->clear_color.a} };
    clearValues[2].color = { {gloco->clear_color.r, gloco->clear_color.g, gloco->clear_color.b, gloco->clear_color.a} };
    clearValues[3].color = { {gloco->clear_color.r, gloco->clear_color.g, gloco->clear_color.b, gloco->clear_color.a} };
    clearValues[4].depthStencil = { 1.0f, 0 };
    #endif

    renderPassInfo.clearValueCount = std::size(clearValues);
    renderPassInfo.pClearValues = clearValues;

    vkCmdBeginRenderPass(commandBuffers[currentFrame], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
}

inline fan::vulkan::viewport_list_NodeReference_t::viewport_list_NodeReference_t(fan::vulkan::viewport_t* viewport) {
  NRI = viewport->viewport_reference.NRI;
}

inline void fan::vulkan::pipeline_t::open(fan::vulkan::context_t& context, const properties_t& p) {
  VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
  vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

  VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
  inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology = p.shape_type;
  inputAssembly.primitiveRestartEnable = VK_FALSE;

  VkPipelineViewportStateCreateInfo viewportState{};
  viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.scissorCount = 1;

  VkPipelineRasterizationStateCreateInfo rasterizer{};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizer.lineWidth = 1.0f;
  rasterizer.cullMode = VK_CULL_MODE_NONE;
  rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterizer.depthBiasEnable = VK_FALSE;

  VkPipelineMultisampleStateCreateInfo multisampling{};
  multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineDepthStencilStateCreateInfo depthStencil{};
  depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depthStencil.depthTestEnable = VK_FALSE;//p.enable_depth_test;
  depthStencil.depthWriteEnable = VK_TRUE;
  depthStencil.depthCompareOp = p.depth_test_compare_op;
  depthStencil.depthBoundsTestEnable = VK_FALSE;
  depthStencil.stencilTestEnable = VK_FALSE;

  VkPipelineColorBlendStateCreateInfo colorBlending{};
  colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlending.logicOpEnable = VK_FALSE;
  colorBlending.logicOp = VK_LOGIC_OP_NO_OP;
  colorBlending.attachmentCount = p.color_blend_attachment_count;
  colorBlending.pAttachments = p.color_blend_attachment;
  colorBlending.blendConstants[0] = 1.0f;
  colorBlending.blendConstants[1] = 1.0f;
  colorBlending.blendConstants[2] = 1.0f;
  colorBlending.blendConstants[3] = 1.0f;

  std::vector<VkDynamicState> dynamicStates = {
      VK_DYNAMIC_STATE_VIEWPORT,
      VK_DYNAMIC_STATE_SCISSOR
  };
  VkPipelineDynamicStateCreateInfo dynamicState{};
  dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamicState.dynamicStateCount = dynamicStates.size();
  dynamicState.pDynamicStates = dynamicStates.data();

  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = p.descriptor_layout_count;
  pipelineLayoutInfo.pSetLayouts = p.descriptor_layout;

  VkPushConstantRange push_constant;
  push_constant.offset = 0;
  push_constant.size = p.push_constants_size;
  push_constant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

  pipelineLayoutInfo.pPushConstantRanges = &push_constant;
  pipelineLayoutInfo.pushConstantRangeCount = 1;

  if (vkCreatePipelineLayout(context.device, &pipelineLayoutInfo, nullptr, &m_layout) != VK_SUCCESS) {
    fan::throw_error("failed to create pipeline layout!");
  }

  VkGraphicsPipelineCreateInfo pipelineInfo{};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineInfo.stageCount = 2;
  pipelineInfo.pStages = ((loco_t::shader_t*)p.shader)->get_shader().shaderStages;
  pipelineInfo.pVertexInputState = &vertexInputInfo;
  pipelineInfo.pInputAssemblyState = &inputAssembly;
  pipelineInfo.pViewportState = &viewportState;
  pipelineInfo.pRasterizationState = &rasterizer;
  pipelineInfo.pMultisampleState = &multisampling;
  pipelineInfo.pDepthStencilState = &depthStencil;
  pipelineInfo.pColorBlendState = &colorBlending;
  pipelineInfo.pDynamicState = &dynamicState;
  pipelineInfo.layout = m_layout;
  pipelineInfo.renderPass = context.renderPass;
  pipelineInfo.subpass = p.subpass;
  pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

  if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline) != VK_SUCCESS) {
    fan::throw_error("failed to create graphics pipeline");
  }
}
#endif

#if defined(loco_imgui) && defined(fan_platform_linux)
static void imgui_xorg_init() {
  ImGuiIO& io = ImGui::GetIO();
  io.DisplaySize = gloco->window.get_size();
  gloco->window.add_mouse_move_callback([](const auto& d) {
    auto& io = ImGui::GetIO();
    if (!io.WantSetMousePos) {
      io.AddMousePosEvent(d.position.x, d.position.y);
    }
  });
}
static void imgui_xorg_new_frame() {
  ImGuiIO& io = ImGui::GetIO();
  io.DisplaySize = gloco->window.get_size();
}
#endif

#if defined(loco_imgui)
namespace ImGui {
  inline IMGUI_API void Image(loco_t::image_t& img, const ImVec2& size, const ImVec2& uv0 = ImVec2(0, 0), const ImVec2& uv1 = ImVec2(1, 1), const ImVec4& tint_col = ImVec4(1, 1, 1, 1), const ImVec4& border_col = ImVec4(0, 0, 0, 0)) {
    ImGui::Image((void*)img.get_texture(), size, uv0, uv1, tint_col, border_col);
  }
  inline IMGUI_API bool ImageButton(loco_t::image_t& img, const ImVec2& size, const ImVec2& uv0 = ImVec2(0, 0), const ImVec2& uv1 = ImVec2(1, 1), int frame_padding = -1, const ImVec4& bg_col = ImVec4(0, 0, 0, 0), const ImVec4& tint_col = ImVec4(1, 1, 1, 1)) {
    return ImGui::ImageButton((void*)img.get_texture(), size, uv0, uv1, frame_padding, bg_col, tint_col);
  }
}
#endif

#include _FAN_PATH(graphics/collider.h)

#if defined(loco_model_3d)
  #include _FAN_PATH(graphics/opengl/3D/objects/model.h)
#endif