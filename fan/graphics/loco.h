#pragma once

#if !defined(loco_vulkan) && !defined(loco_opengl)
#define loco_opengl
#endif

struct loco_t;

// to set new loco use gloco = new_loco;
inline struct global_loco_t {

  loco_t* loco = nullptr;

  operator loco_t* ();
  global_loco_t& operator=(loco_t* l);
  loco_t* operator->();
}thread_local gloco;

struct loco_t;
#ifndef loco_legacy
  #define loco_framebuffer
#endif

#include <fan/graphics/graphics.h>
#include <fan/system.h>
#include <fan/time/timer.h>
#include <fan/physics/collision/circle.h>
#include <fan/io/directory.h>
#include <fan/event/event.h>
#include <fan/trees/quad_tree.h>
#include <fan/graphics/divider.h>

#if defined(loco_opengl)
  #include <fan/graphics/opengl/gl_image.h>
#elif defined(loco_vulkan)
  #include <fan/graphics/vulkan/vk_image.h>
#endif

#if defined(loco_imgui)
#include <fan/graphics/console.h>
#endif

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

#include <fan/graphics/loco_auto_config.h>

#if defined(loco_letter)
  #include <fan/graphics/font.h>
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

    static uint8_t get_texture_amount(uint8_t format);
    static std::array<fan::vec2ui, 4> get_image_sizes(uint8_t format, const fan::vec2ui& image_size);
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

  #if defined(loco_tp)
  using font_t = fan::graphics::gl_font_impl::font_t;
  #endif

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

  void use();

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

  uint32_t fb_vao;
  uint32_t fb_vbo;

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

  #endif

public:
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

  using image_t = fan::graphics::image_t;

  template <uint8_t n_>
  struct textureid_t : fan::graphics::gl_image_impl::image_list_NodeReference_t {
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
  fan::graphics::gl_image_impl::image_list_t image_list;
  //#endif

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
      resize_callback_id_t();
      resize_callback_id_t(const inherit_t& i);
      resize_callback_id_t(resize_callback_id_t&& i);

      resize_callback_id_t& operator=(const resize_callback_id_t& i) = delete;

      resize_callback_id_t& operator=(resize_callback_id_t&& i);

      operator loco_t::viewport_resize_callback_NodeReference_t();
      ~resize_callback_id_t();
    };

    resize_callback_id_t add_resize_callback(resize_cb_t function) {
      auto nr = gloco->m_viewport_resize_callback.NewNodeLast();
      gloco->m_viewport_resize_callback[nr].data = function;
      return resize_callback_id_t(nr);
    }

    camera_t() {
      camera_reference.sic();
    }
    camera_t(const camera_t& camera) {
      m_view = camera.m_view;
      m_projection = camera.m_projection;
      coordinates = camera.coordinates;
      open();
    }
    camera_t(camera_t&& camera) {
      m_view = camera.m_view;
      m_projection = camera.m_projection;
      coordinates = camera.coordinates;
      camera_reference = camera.camera_reference;
      gloco->camera_list[camera_reference].camera_id = this;
      camera.camera_reference.sic();
    }
    camera_t& operator=(const camera_t& t) {
      if (this != &t) {
        m_view = t.m_view;
        m_projection = t.m_projection;
        coordinates = t.coordinates;
        open();
      }
      return *this;
    }
    camera_t& operator=(camera_t&& t) {
      if (this != &t) {
        if (!camera_reference.iic()) {
          close();
        }
        camera_reference = t.camera_reference;
        m_view = t.m_view;
        m_projection = t.m_projection;
        coordinates = t.coordinates;
        gloco->camera_list[camera_reference].camera_id = this;
        t.camera_reference.sic();
      }
      return *this;
    }

    void link(const camera_t& t) {
      m_view = t.m_view;
      m_projection = t.m_projection;
      coordinates = t.coordinates;
      camera_reference = t.camera_reference;
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
        znearfar
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
  #include <fan/fan_bll_preset.h>
  #define BLL_set_prefix cid_list
  #define BLL_set_type_node uint32_t
  #define BLL_set_NodeData fan::graphics::cid_t cid;
  #define BLL_set_Link 1
  #include _FAN_PATH(BLL/BLL.h)
public:

  struct cid_nt_t : cid_list_NodeReference_t {
    loco_t::cid_t* operator->() const;
    using base_t = cid_list_NodeReference_t;
    void init();

    bool is_invalid() const;

    void invalidate_soft();

    void invalidate();

    uint32_t* gdp4();
  };

  struct cid_nr_t : cid_nt_t {

    cid_nr_t();
    cid_nr_t(const cid_nt_t& c);

    cid_nr_t(const cid_nr_t& nr);

    cid_nr_t(cid_nr_t&& nr);

    loco_t::cid_nr_t& operator=(const cid_nr_t& id);

    loco_t::cid_nr_t& operator=(cid_nr_t&& id);
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
  f64_t& get_delta_time() {
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

    inline shape_t(const shape_t& id);
    inline shape_t(shape_t&& id);
    
    loco_t::shape_t& operator=(const shape_t& id);
    
    loco_t::shape_t& operator=(shape_t&& id);

    ~shape_t();

    void erase();
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
    // might return nullptr image
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
  #include <fan/fan_bll_preset.h>
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

    imgui_element_nr_t() = default;

    imgui_element_nr_t(const imgui_element_nr_t& nr);

    imgui_element_nr_t(imgui_element_nr_t&& nr);
    ~imgui_element_nr_t();


    imgui_element_nr_t& operator=(const imgui_element_nr_t& id);

    imgui_element_nr_t& operator=(imgui_element_nr_t&& id);

    void init();

    bool is_invalid() const;

    void invalidate_soft();

    void invalidate();

    void set(const auto& lambda);
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

  #if defined(loco_post_process)
  #include _FAN_PATH(graphics/opengl/2D/effects/blur.h)
  blur_t blur[1];
  #include _FAN_PATH(graphics/opengl/2D/effects/bloom.h)
  bloom_t bloom;
  #endif

  #if defined(loco_letter)
  font_t font;
  #endif

  static constexpr uint8_t pixel_data[] = {
    1, 0, 0, 1,
    1, 0, 0, 1
  };

  #if defined(loco_custom_id_t_types)
  using custom_id_t_types_t = std::tuple<loco_custom_id_t_types>;
  #endif

  loco_t() : loco_t(properties_t()) {

  }

  loco_t(const properties_t& p);

  #if defined(loco_vfi)
  void push_back_input_hitbox(loco_t::shapes_t::vfi_t::shape_id_t& id, const loco_t::shapes_t::vfi_t::properties_t& p);
  /* uint32_t push_back_keyboard_event(uint32_t depth, const fan_2d::graphics::gui::ke_t::properties_t& p) {
     return element_depth[depth].keyboard_event.push_back(p);
   }*/

  void feed_mouse_move(const fan::vec2& mouse_position);

  void feed_mouse_button(uint16_t button, fan::mouse_state button_state, const fan::vec2& mouse_position);

  void feed_keyboard(int key, fan::keyboard_state keyboard_state);

  void feed_text(uint32_t key);
  #endif

  void process_frame();
  #if defined(loco_window)
  uint32_t get_fps();

  void set_vsync(bool flag);

  fan::vec2 transform_matrix(const fan::vec2& position);

  fan::vec2 screen_to_ndc(const fan::vec2& screen_pos);

  fan::vec2 ndc_to_screen(const fan::vec2& ndc_position);

  //  behaving oddly
  fan::vec2d get_mouse_position(const loco_t::camera_t& camera, const loco_t::viewport_t& viewport);

  fan::vec2d get_mouse_position();

  static fan::vec2 translate_position(const fan::vec2 & p, fan::graphics::viewport_t * viewport, loco_t::camera_t * camera);
  fan::vec2 translate_position(const fan::vec2& p);

  #if defined(loco_vulkan)
  loco_t::shader_t render_fullscreen_shader;
  #endif


#if defined(loco_framebuffer)
  #if defined(loco_opengl)

    fan::opengl::core::framebuffer_t m_framebuffer;
    fan::opengl::core::renderbuffer_t m_rbo;
    loco_t::image_t color_buffers[4];
    loco_t::shader_t m_fbo_final_shader;

  #endif
#endif

  bool process_loop(const fan::function_t<void()>& lambda = []{});

  void loop(const fan::function_t<void()>& lambda);

  #endif


  #if defined(loco_cuda)

  struct cuda_textures_t {

    cuda_textures_t();
    ~cuda_textures_t();
    void close(loco_t* loco, loco_t::shape_t& cid);

    void resize(loco_t* loco, loco_t::cid_nt_t& id, uint8_t format, fan::vec2ui size, uint32_t filter = loco_t::image_t::filter::linear);

    cudaArray_t& get_array(uint32_t index);

    struct graphics_resource_t {
      void open(int texture_id);
      void close();
      void map();
      void unmap();
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
  #include <fan/fan_bll_preset.h>
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
    vfi_id_t(const properties_t& p);
    vfi_id_t& operator[](const properties_t& p);
    ~vfi_id_t();

    loco_t::shapes_t::vfi_t::shape_id_t cid;
  };

  #endif

  template <typename T>
  void push_shape(loco_t::cid_nt_t& id, T properties) {
    shapes.iterate([&]<auto i>(auto & shape) {
      fan_if_has_function(&shape, push_back, (id, properties));
    });
  }

  void shape_get_properties(loco_t::cid_nt_t& id, const auto& lambda) {
    shapes.iterate([&]<auto i, typename T>(T & shape) {
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
  );

  void shape_erase(loco_t::cid_nt_t& id);

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

  void shape_set_line(
  loco_t::cid_nt_t& id,
    const fan::vec3& src,
    fan::vec2 dst
  );

  std::vector<camera_impl_t*> viewport_handler;
  camera_impl_t* add_camera(fan::graphics::direction_e split_direction);

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

  fan::string get_sprite_vertex_shader();

  loco_t::shader_t create_sprite_shader(const fan::string& fragment);

  #endif
  #if defined(loco_light)
  loco_t::shader_t create_light_shader(const fan::string& fragment);

  #endif

  static fan::vec2 convert_mouse_to_ndc(const fan::vec2& mouse_position, const fan::vec2i& window_size);
  fan::vec2 convert_mouse_to_ndc(const fan::vec2& mouse_position) const;
  fan::vec2 convert_mouse_to_ndc() const;

  static fan::ray3_t convert_mouse_to_ray(const fan::vec2i& mouse_position, const fan::vec3& camera_position, const fan::mat4& projection, const fan::mat4& view);

  fan::ray3_t convert_mouse_to_ray(const fan::vec3& camera_position, const fan::mat4& projection, const fan::mat4& view);
  fan::ray3_t convert_mouse_to_ray(const fan::mat4& projection, const fan::mat4& view);

  static bool is_ray_intersecting_cube(const fan::ray3_t& ray, const fan::vec3& position, const fan::vec3& size);


  #if defined(loco_sprite)
  void add_fragment_shader_reload(int key, const fan::string& vs_path, const fan::string& fs_path);
  #endif
  #if defined(loco_imgui)

  #define fan_imgui_dragfloat_named(name, variable, speed, m_min, m_max) \
  [=] <typename T5>(T5& var){ \
    if constexpr(std::is_same_v<f32_t, T5>)  { \
      return ImGui::DragFloat(fan::string(std::move(name)).c_str(), &var, (f32_t)speed, (f32_t)m_min, (f32_t)m_max); \
    } \
    else if constexpr(std::is_same_v<fan::vec2, T5>)  { \
      return ImGui::DragFloat2(fan::string(std::move(name)).c_str(), var.data(), (f32_t)speed, (f32_t)m_min, (f32_t)m_max); \
    } \
    else if constexpr(std::is_same_v<fan::vec3, T5>)  { \
      return ImGui::DragFloat3(fan::string(std::move(name)).c_str(), var.data(), (f32_t)speed, (f32_t)m_min, (f32_t)m_max); \
    } \
    else if constexpr(std::is_same_v<fan::vec4, T5>)  { \
      return ImGui::DragFloat4(fan::string(std::move(name)).c_str(), var.data(), (f32_t)speed, (f32_t)m_min, (f32_t)m_max); \
    } \
  }(variable)

  #define fan_imgui_dragfloat(variable, speed, m_min, m_max) \
    fan_imgui_dragfloat_named(STRINGIFY(variable), variable, speed, m_min, m_max)

  static std::string extract_variable_type(const std::string& string_data, const std::string& varName);

  template <typename T>
  struct imgui_fs_var_t {
    loco_t::imgui_element_t ie;

    imgui_fs_var_t() = default;

    imgui_fs_var_t(
      loco_t::shader_t* shader,
      const fan::string& var_name,
      T initial = 0,
      T speed = 1,
      T min = -100000,
      T max = 100000
    ) {
      auto fs = shader->get_shader().sfragment;
      auto found = fs.find(var_name);
      if (found == std::string::npos) {
        fan::throw_error(var_name, "not found");
      }

      fan::string type = extract_variable_type(fs, var_name);
      if (type.empty()) {
        fan::throw_error(var_name, "failed to find type of variable");
      }

      switch (fan::get_hash(type)) {
        case fan::get_hash(std::string_view("float")): {
          shader->set_float(var_name, initial);
          break;
        }
        case fan::get_hash(std::string_view("int")): {
          shader->set_int(var_name, initial);
          break;
        }
        case fan::get_hash(std::string_view("vec2")): {
          shader->set_vec2(var_name, initial);
          break;
        }
        case fan::get_hash(std::string_view("vec3")): {
          shader->set_vec3(var_name, initial);
          break;
        }
        case fan::get_hash(std::string_view("vec4")): {
          shader->set_vec4(var_name, initial);
          break;
        }
      }

      ie = [shader, var_name, speed, min, max, type, data = initial]() mutable {
        switch (fan::get_hash(type)) {
          case fan::get_hash(std::string_view("float")): {
            if (fan_imgui_dragfloat_named(var_name, data, speed, min, max)) {
              shader->set_float(var_name, data);
            }
            break;
          }
          case fan::get_hash(std::string_view("int")): {
            if (fan_imgui_dragfloat_named(var_name, data, speed, min, max)) {
              shader->set_int(var_name, data);
            }
            break;
          }
          case fan::get_hash(std::string_view("vec2")): {
            if (fan_imgui_dragfloat_named(var_name, data, speed, min, max)) {
              shader->set_vec2(var_name, data);
            }
            break;
          }
          case fan::get_hash(std::string_view("vec3")): {
            if (fan_imgui_dragfloat_named(var_name, data, speed, min, max)) {
              shader->set_vec3(var_name, data);
            }
            break;
          }
          case fan::get_hash(std::string_view("vec4")): {
            if (fan_imgui_dragfloat_named(var_name, data, speed, min, max)) {
              shader->set_vec4(var_name, data);
            }
            break;
          }
        }
      };
    }
  };
  #endif

  #if defined(loco_compute_shader)
  #include _FAN_PATH(graphics/vulkan/compute_shader.h)
  #endif

  #if defined(loco_imgui)
  void set_imgui_viewport(loco_t::viewport_t& viewport);
  #endif

  #if defined(loco_imgui)
  fan::console_t console;
  bool toggle_console = false;
  bool toggle_fps = false;


  ImFont* fonts[6];
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

inline loco_t::camera_list_NodeReference_t::camera_list_NodeReference_t(loco_t::camera_t* camera) {
  NRI = camera->camera_reference.NRI;
}

inline loco_t::shader_list_NodeReference_t::shader_list_NodeReference_t(loco_t::shader_t* shader) {
  NRI = shader->shader_reference.NRI;
}

#if defined(loco_opengl)
inline fan::opengl::viewport_list_NodeReference_t::viewport_list_NodeReference_t(fan::opengl::viewport_t* viewport) {
  NRI = viewport->viewport_reference.NRI;
}
#endif

#if defined(loco_button)
namespace fan::opengl {
  // Primary template for the constructor
  inline theme_list_NodeReference_t::theme_list_NodeReference_t(void* theme) {
    //static_assert(std::is_same_v<decltype(theme), loco_t::theme_t*>, "invalid parameter passed to theme");
    NRI = ((loco_t::theme_t*)theme)->theme_reference.NRI;
  }
}
#endif

#if defined(loco_imgui)
namespace ImGui {
  IMGUI_API void Image(loco_t::image_t& img, const ImVec2& size, const ImVec2& uv0 = ImVec2(0, 0), const ImVec2& uv1 = ImVec2(1, 1), const ImVec4& tint_col = ImVec4(1, 1, 1, 1), const ImVec4& border_col = ImVec4(0, 0, 0, 0));
  IMGUI_API bool ImageButton(loco_t::image_t& img, const ImVec2& size, const ImVec2& uv0 = ImVec2(0, 0), const ImVec2& uv1 = ImVec2(1, 1), int frame_padding = -1, const ImVec4& bg_col = ImVec4(0, 0, 0, 0), const ImVec4& tint_col = ImVec4(1, 1, 1, 1));
}
#endif
#include _FAN_PATH(graphics/collider.h)

#if defined(loco_model_3d)
  #include _FAN_PATH(graphics/opengl/3D/objects/model.h)
#endif