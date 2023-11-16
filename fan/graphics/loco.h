#ifndef loco_vulkan
#define loco_opengl
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

#define loco_framebuffer

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
#define loco_texture_pack
#define loco_unlit_sprite
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
  }

  std::vector<fan::function_t<void()>> m_draw_queue_light;

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
    custom
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

  #if defined(loco_opengl) && defined(loco_context)

public:
  using viewport_t = fan::graphics::viewport_t;
protected:

  unsigned int fb_vao;
  unsigned int fb_vbo;

  void initialize_final_fb() {
    static constexpr f32_t quad_vertices[] = {
      -1.0f, 1.0f, 0, 0.0f, 1.0f,
      -1.0f, -1.0f, 0, 0.0f, 0.0f,
      1.0f, 1.0f, 0, 1.0f, 1.0f,
      1.0f, -1.0f, 0, 1.0f, 0.0f,
    };
    auto& context = get_context();
    context.opengl.glGenVertexArrays(1, &fb_vao);
    context.opengl.glGenBuffers(1, &fb_vbo);
    context.opengl.glBindVertexArray(fb_vao);
    context.opengl.glBindBuffer(fan::opengl::GL_ARRAY_BUFFER, fb_vbo);
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
  struct image_t;

  #if defined(loco_window)

  #if defined(loco_opengl)
  #include _FAN_PATH(graphics/opengl/image_list_builder_settings.h)
  #endif
  #include _FAN_PATH(BLL/BLL.h)

  image_list_t image_list;

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

  #if defined(loco_opengl)
  #include _FAN_PATH(graphics/opengl/gl_image.h)
  #endif

  struct camera_t;

  #define BLL_set_declare_NodeReference 1
  #define BLL_set_declare_rest 0
  #if defined(loco_opengl)
  #include _FAN_PATH(graphics/opengl/camera_list_builder_settings.h)
  #endif
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

  struct camera_t {

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
      camera_position = 0;
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
      return camera_position;
    }
    void set_position(const fan::vec3& cp) {
      camera_position = cp;

      m_view[3][0] = 0;
      m_view[3][1] = 0;
      m_view[3][2] = 0;
      m_view = m_view.translate(camera_position);
      fan::vec3 position = m_view.get_translation();
      constexpr fan::vec3 front(0, 0, 1);

      m_view = fan::math::look_at_left<fan::mat4, fan::vec3>(position, position + front, fan::camera::world_up);
    }

    fan::vec2 get_camera_size() const {
      return fan::vec2(std::abs(coordinates.right - coordinates.left), std::abs(coordinates.down - coordinates.up));
    }

    bool calculate_aspect_ratio = false;

    fan::vec2 some_function(fan::vec2 d, fan::vec2 c) {
      return c / (c / d).min();
    }

    void set_ortho(fan::vec2 x, fan::vec2 y, loco_t::viewport_t* aspect_ratio_viewport = nullptr) {

      if (aspect_ratio_viewport) {
        fan::vec2 desired_res = { 1, 1 };
        fan::vec2 current_res = aspect_ratio_viewport->get_size();

        auto ortho = some_function(desired_res, current_res);

        x = { -ortho.x, +ortho.x };
        y = { -ortho.y, +ortho.y };

        calculate_aspect_ratio = true;
      }

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
        #endif


      );

      m_view[3][0] = 0;
      m_view[3][1] = 0;
      m_view[3][2] = 0;
      m_view = m_view.translate(camera_position);
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

    fan::mat4 m_projection;
    // temporary
    fan::mat4 m_view;

    fan::vec3 camera_position;

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
    loco_t::viewport_t viewport;
  };

  #define BLL_set_declare_NodeReference 0
  #define BLL_set_declare_rest 1
  #if defined(loco_opengl)
  #include _FAN_PATH(graphics/opengl/camera_list_builder_settings.h)
  #endif
  #include _FAN_PATH(BLL/BLL.h)

  camera_list_t camera_list;

  uint32_t camera_index = 0;

  image_t unloaded_image;
  fan::color clear_color = {0.10, 0.10, 0.131, 1};

  #endif

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
  #include _FAN_PATH(fan_bll_present.h)
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
    fan::vec2 window_size = 1300;
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
    shape->m_current_shader->use(get_context());
    shape->m_current_shader->set_camera(get_context(), camera, &m_write_queue);
    shape->m_current_shader->set_vec2(get_context(), "matrix_size",
      fan::vec2(camera->coordinates.right - camera->coordinates.left, camera->coordinates.down - camera->coordinates.up).abs()
    );
    shape->m_current_shader->set_vec2(get_context(), "camera_position", camera->get_position());
    #endif
  }
  void process_block_properties_element(auto* shape, fan::graphics::viewport_list_NodeReference_t viewport_id) {
    fan::graphics::viewport_t* viewport = get_context().viewport_list[viewport_id].viewport_id;
    viewport->set(
      viewport->get_position(),
      viewport->get_size(),
      window.get_size()
    );
    shape->m_current_shader->set_vec4(get_context(), "viewport", fan::vec4(viewport->get_position(), viewport->get_size()));
  }

  template <uint8_t n>
  void process_block_properties_element(auto* shape, textureid_t<n> tid) {
    #if defined(loco_opengl)
    if (tid.NRI == (decltype(tid.NRI))-1) {
      return;
    }
    shape->m_current_shader->use(get_context());
    shape->m_current_shader->set_int(get_context(), tid.name, n);
    get_context().opengl.call(get_context().opengl.glActiveTexture, fan::opengl::GL_TEXTURE0 + n);
    get_context().opengl.call(get_context().opengl.glBindTexture, fan::opengl::GL_TEXTURE_2D, image_list[tid].texture_id);
    #endif
  }

  void process_block_properties_element(auto* shape, uint16_t depth) {

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

    bool append_letter(wchar_t wc, bool force = false) {
      return gloco->shape_append_letter(*this, wc, force);
    }

    fan_build_get_set_cref(fan::vec3, position);

    void set_position(const fan::vec2& data) {
      gloco->shape_set_position(*this, fan::vec3(data, get_position().z));
    }

    fan_build_get_set_cref(fan::vec2, size);
    fan_build_get_set_cref(fan::color, color);
    fan_build_get_set_cref(f32_t, angle);
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

    operator fan::opengl::cid_t* () {
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
  #include _FAN_PATH(fan_bll_present.h)
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
    #define vk_shape_wboit
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

    #include _FAN_PATH(graphics/opengl/2D/effects/particles.h)
    particles_t particles;

    #endif
    #if defined(loco_unlit_sprite)
    #define sb_shape_var_name unlit_sprite
    #define sb_sprite_name unlit_sprite_t
    #define sb_custom_shape_type loco_t::shape_type_t::unlit_sprite
    #define sb_shader_fragment_path _FAN_PATH_QUOTE(graphics/glsl/opengl/2D/objects/unlit_sprite.fs)
    #include _FAN_PATH(graphics/opengl/2D/objects/sprite.h)
    unlit_sprite_t unlit_sprite;
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
    #if defined(loco_model_3d)
    #define sb_shape_var_name model
    #include _FAN_PATH(graphics/opengl/3D/objects/model.h)
    model_t sb_shape_var_name;
    #undef sb_shape_var_name
    #endif
    #if defined(loco_post_process)
    #define sb_post_process_var_name post_process
    #include _FAN_PATH(graphics/opengl/2D/effects/post_process.h)
    post_process_t sb_post_process_var_name;
    #undef sb_post_process_var_name
    #endif
   
    #include _FAN_PATH(graphics/custom_shapes.h)
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
    window(p.window_size),
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
    #if defined(loco_window)

    initialize_final_fb();

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
    #if defined(loco_opengl)
    fan::print("RENDERER BACKEND: OPENGL");
    #elif defined(loco_vulkan)
    fan::print("RENDERER BACKEND: VULKAN");
    #endif

    #if defined(loco_letter)
    font.open(loco_font);
    #endif

    #if defined(loco_post_process)
    fan::opengl::core::renderbuffer_t::properties_t rp;
    rp.size = window.get_size();
    if (post_process.open(rp)) {
      fan::throw_error("failed to initialize frame buffer");
    }
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

    fan::webp::image_info_t ii;
    ii.data = nullptr;
    ii.size = window.get_size();

    lp.internal_format = fan::opengl::GL_RGBA;
    lp.format = fan::opengl::GL_RGBA;
    lp.min_filter = fan::opengl::GL_LINEAR_MIPMAP_LINEAR;
    lp.mag_filter = fan::opengl::GL_LINEAR_MIPMAP_LINEAR;
    lp.type = fan::opengl::GL_FLOAT;

    color_buffers[0].load(ii, lp);
    get_context().opengl.call(get_context().opengl.glGenerateMipmap, fan::opengl::GL_TEXTURE_2D);

    color_buffers[0].bind_texture();
    fan::opengl::core::framebuffer_t::bind_to_texture(
      get_context(),
      color_buffers[0].get_texture(),
      fan::opengl::GL_COLOR_ATTACHMENT0
    );

    lp.internal_format = fan::opengl::GL_RGBA;
    lp.format = fan::opengl::GL_RGBA;

    color_buffers[1].load(ii, lp);

    color_buffers[1].bind_texture();
    fan::opengl::core::framebuffer_t::bind_to_texture(
      get_context(),
      color_buffers[1].get_texture(),
      fan::opengl::GL_COLOR_ATTACHMENT1
    );

    get_context().opengl.call(get_context().opengl.glGenerateMipmap, fan::opengl::GL_TEXTURE_2D);

    window.add_resize_callback([this](const auto& d) {
      loco_t::image_t::load_properties_t lp;
      lp.visual_output = fan::opengl::GL_CLAMP_TO_EDGE;

      fan::webp::image_info_t ii;
      ii.data = nullptr;
      ii.size = window.get_size();

      lp.internal_format = fan::opengl::GL_RGBA;
      lp.format = fan::opengl::GL_RGBA;
      lp.type = fan::opengl::GL_FLOAT;
      lp.min_filter = fan::opengl::GL_LINEAR_MIPMAP_LINEAR;
      lp.mag_filter = fan::opengl::GL_LINEAR_MIPMAP_LINEAR;

      color_buffers[0].reload_pixels(ii, lp);

      color_buffers[0].bind_texture();
      fan::opengl::core::framebuffer_t::bind_to_texture(
        get_context(),
        color_buffers[0].get_texture(),
        fan::opengl::GL_COLOR_ATTACHMENT0
      );

      get_context().opengl.call(get_context().opengl.glGenerateMipmap, fan::opengl::GL_TEXTURE_2D);

      lp.internal_format = fan::opengl::GL_RGBA;
      lp.format = fan::opengl::GL_RGBA;

      color_buffers[1].reload_pixels(ii, lp);

      color_buffers[1].bind_texture();
      fan::opengl::core::framebuffer_t::bind_to_texture(
        get_context(),
        color_buffers[1].get_texture(),
        fan::opengl::GL_COLOR_ATTACHMENT1
      );

      get_context().opengl.call(get_context().opengl.glGenerateMipmap, fan::opengl::GL_TEXTURE_2D);

      fan::opengl::core::renderbuffer_t::properties_t rp;
      m_framebuffer.bind(get_context());
      rp.size = ii.size;
      rp.internalformat = fan::opengl::GL_DEPTH_COMPONENT;
      m_rbo.set_storage(get_context(), rp);

      fan::vec2 window_size = gloco->window.get_size();

      default_camera->viewport.set(fan::vec2(0, 0), d.size, d.size);
  });

  fan::opengl::core::renderbuffer_t::properties_t rp;
  m_framebuffer.bind(get_context());
  rp.size = ii.size;
  rp.internalformat = fan::opengl::GL_DEPTH_COMPONENT;
  m_rbo.open(get_context());
  m_rbo.set_storage(get_context(), rp);
  rp.internalformat = fan::opengl::GL_DEPTH_ATTACHMENT;
  m_rbo.bind_to_renderbuffer(get_context(), rp);

  unsigned int attachments[sizeof(color_buffers) / sizeof(color_buffers[0])];

  for (uint8_t i = 0; i < std::size(color_buffers); ++i) {
    attachments[i] = fan::opengl::GL_COLOR_ATTACHMENT0 + i;
  }

  get_context().opengl.call(get_context().opengl.glDrawBuffers, std::size(attachments), attachments);
  // finally check if framebuffer is complete
  if (!m_framebuffer.ready(get_context())) {
    fan::throw_error("framebuffer not ready");
  }

  m_framebuffer.unbind(get_context());

  m_fbo_final_shader.open(get_context());

  m_fbo_final_shader.set_vertex(
    get_context(),
    fan::graphics::read_shader(_FAN_PATH_QUOTE(graphics/glsl/opengl/2D/effects/loco_fbo.vs))
  );
  m_fbo_final_shader.set_fragment(
    get_context(),
    fan::graphics::read_shader(_FAN_PATH_QUOTE(graphics/glsl/opengl/2D/effects/loco_fbo.fs))
  );
  m_fbo_final_shader.compile(get_context());

    #endif
    #endif
    default_texture.create_missing_texture();
    transparent_texture.create_transparent_texture();

    fan::vec2 window_size = window.get_size();

    default_camera = add_camera(fan::graphics::direction_e::right);

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

    window.add_buttons_callback([&](const auto& d) {
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
      ImGuiKey imgui_key = fan::window_input::fan_to_imguikey(d.key);
      io.AddKeyEvent(imgui_key, (int)d.state);
    });
    window.add_text_callback([&](const auto& d) {
      io.AddInputCharacter(d.character);
    });

    #if defined(loco_imgui)
    loco_t::imgui_themes::dark();

    #if defined(fan_platform_windows)
    ImGui_ImplWin32_Init(window.get_handle());
    #elif defined(fan_platform_linux)
    imgui_xorg_init();
    #endif
    ImGui_ImplOpenGL3_Init();
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

    #ifdef loco_post_process
    post_process.start_capture();
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

    m_fbo_final_shader.use(get_context());
    m_fbo_final_shader.set_int(get_context(), "_t00", 0);
    m_fbo_final_shader.set_int(get_context(), "_t01", 1);

    get_context().opengl.glActiveTexture(fan::opengl::GL_TEXTURE0);
    color_buffers[0].bind_texture();

    get_context().opengl.glActiveTexture(fan::opengl::GL_TEXTURE1);
    color_buffers[1].bind_texture();

    render_final_fb();

    #if defined(loco_imgui)
    ImGui_ImplOpenGL3_NewFrame();
    #if defined(fan_platform_windows)
    ImGui_ImplWin32_NewFrame();
    #elif defined(fan_platform_linux)
    imgui_xorg_new_frame();
    #endif
    ImGui::NewFrame();

    {
      auto it = m_imgui_draw_cb.GetNodeFirst();
      while (it != m_imgui_draw_cb.dst) {
        m_imgui_draw_cb.StartSafeNext(it);
        m_imgui_draw_cb[it]();
        it = m_imgui_draw_cb.EndSafeNext();
      }
    }

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    #endif

    #endif
    get_context().render(get_window());

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

#if defined(loco_framebuffer)
  #if defined(loco_opengl)

    fan::opengl::core::framebuffer_t m_framebuffer;
    fan::opengl::core::renderbuffer_t m_rbo;
    loco_t::image_t color_buffers[2];
    fan::opengl::shader_t m_fbo_final_shader;

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
      uint8_t image_amount = fan::pixel_format::get_texture_amount(loco->pixel_format_renderer.sb_get_ri(cid).format);
      auto& ri = loco->pixel_format_renderer.sb_get_ri(cid);
      for (uint32_t i = 0; i < image_amount; ++i) {
        wresources[i].close();
        ri.images[i].unload();
      }
    }

    void resize(loco_t* loco, loco_t::cid_nt_t& id, uint8_t format, fan::vec2ui size, uint32_t filter = loco_t::image_t::filter::linear) {
      auto& ri = loco->pixel_format_renderer.sb_get_ri(id);
      uint8_t image_amount = fan::pixel_format::get_texture_amount(format);
      if (inited == false) {
        // purge cid's images here
        // update cids images
        loco->pixel_format_renderer.reload(id, format, size, filter);
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

        loco->pixel_format_renderer.reload(id, format, size, filter);

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
  #include _FAN_PATH(fan_bll_present.h)
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
    shapes.iterate([&]<auto i>(auto & shape) {
      if (shape.shape_type != (loco_t::shape_type_t)id->shape_type) {
        return;
      }
      fan_if_has_function(&shape, erase, (id));
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
  fan_build_get_set_shape_property(const fan::vec2&, size);
  fan_build_get_set_shape_property(const fan::color&, color);
  fan_build_get_set_shape_property(f32_t, angle);
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

  bool shape_append_letter(
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
  }

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

  fan::graphics::viewport_divider_t viewport_divider;

  #undef make_global_function
  #undef fan_build_get
  #undef fan_build_set

  using image_info_t = fan::webp::image_info_t;
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

inline loco_t::image_list_NodeReference_t::image_list_NodeReference_t(loco_t::image_t* image) {
  NRI = image->texture_reference.NRI;
}

inline loco_t::camera_list_NodeReference_t::camera_list_NodeReference_t(loco_t::camera_t* camera) {
  NRI = camera->camera_reference.NRI;
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

inline fan::opengl::viewport_list_NodeReference_t::viewport_list_NodeReference_t(fan::opengl::viewport_t* viewport) {
  NRI = viewport->viewport_reference.NRI;
}

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

#include _FAN_PATH(graphics/collider.h)